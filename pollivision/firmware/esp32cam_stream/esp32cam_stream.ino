/*
 * PolliVision ESP32-CAM node - the rover's eye.
 *
 * Serves an MJPEG stream that pollivision.io.esp32 consumes, plus a still
 * capture endpoint for camera calibration and a small control endpoint for
 * runtime sensor tuning.
 *
 * Target: AI-Thinker ESP32-CAM (OV2640). Select "AI Thinker ESP32-CAM" as the
 * board, and note this module has no USB - program it through an FTDI adapter
 * with IO0 tied to GND, then remove that link to run.
 *
 * Endpoints
 *   GET  /            status page with the stream URL and current settings
 *   GET  :81/stream   multipart MJPEG stream (the perception input)
 *   GET  /capture     single JPEG still (use for `pollivision calibrate`)
 *   GET  /control?var=<name>&val=<n>   set a sensor parameter at runtime
 *   GET  /status      JSON of the current settings
 *
 * Notes that matter for perception quality, not just for the stream working:
 *
 *  - SVGA (800x600) is the sweet spot. UXGA looks better but the frame interval
 *    roughly triples and the rover's control loop cares far more about latency
 *    than about resolution; the flowers it acts on are large in frame anyway.
 *
 *  - Auto white balance is left ON but auto gain control is capped. The pollen
 *    estimator compares the anther against the petals *in the same frame*, so a
 *    consistent white balance matters less than a stable exposure - and an
 *    uncapped AGC hunts badly on bright yellow petals against dark foliage,
 *    which shows up downstream as a flickering pollen reading.
 *
 *  - Set a fixed frame size before streaming and do not change it mid-run: the
 *    perception stack scales camera intrinsics from the configured resolution,
 *    and a mid-stream change silently invalidates every range estimate.
 */

#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include <WiFi.h>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Leave STA credentials empty to run as an access point the rover connects to,
// which is the more robust arrangement in a field with no infrastructure.
static const char *WIFI_SSID     = "";
static const char *WIFI_PASSWORD = "";

static const char *AP_SSID     = "PolliVision-Cam";
static const char *AP_PASSWORD = "pollinate";   // >= 8 characters

// AI-Thinker ESP32-CAM pin map.
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#define LED_GPIO_NUM       4    // on-board illuminator

#define PART_BOUNDARY "123456789000000000000987654321"
static const char *STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *STREAM_PART =
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

httpd_handle_t camera_httpd = NULL;
httpd_handle_t stream_httpd = NULL;

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

static bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;   config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;   config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;   config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;   config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;   config.pin_pclk  = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href  = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;   config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Two framebuffers when PSRAM is present: the sensor can fill one while the
  // HTTP task is still sending the other, which roughly doubles frame rate.
  if (psramFound()) {
    config.frame_size   = FRAMESIZE_SVGA;   // 800x600, matches configs/esp32cam.yaml
    config.jpeg_quality = 12;               // 10-15 is the useful band; lower = better
    config.fb_count     = 2;
    config.grab_mode    = CAMERA_GRAB_LATEST;  // freshness over throughput
    config.fb_location  = CAMERA_FB_IN_PSRAM;
  } else {
    // Without PSRAM the module cannot hold an SVGA JPEG; drop to VGA. Update
    // camera.width/height in your config to match, or ranging will be wrong.
    config.frame_size   = FRAMESIZE_VGA;    // 640x480
    config.jpeg_quality = 14;
    config.fb_count     = 1;
    config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location  = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }

  sensor_t *s = esp_camera_sensor_get();
  // The OV2640 ships with a vertical flip on most AI-Thinker boards.
  s->set_vflip(s, 1);
  s->set_hmirror(s, 0);

  // Auto exposure and white balance on, but with the gain ceiling capped.
  // An uncapped AGC hunts on high-contrast flower-against-foliage scenes, and
  // that oscillation reaches the pollen estimator as spurious frame-to-frame
  // change - which the verification stage would otherwise read as a transfer.
  s->set_whitebal(s, 1);
  s->set_awb_gain(s, 1);
  s->set_exposure_ctrl(s, 1);
  s->set_gain_ctrl(s, 1);
  s->set_gainceiling(s, GAINCEILING_4X);
  s->set_brightness(s, 0);
  s->set_contrast(s, 1);      // slight lift helps petal/anther separation
  s->set_saturation(s, 1);    // the pollen and sex cues are chroma-based
  s->set_lenc(s, 1);          // lens shading correction, flattens vignetting
  return true;
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

static esp_err_t stream_handler(httpd_req_t *req) {
  char part_buf[64];
  esp_err_t res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "X-Framerate", "30");

  while (true) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Frame capture failed");
      res = ESP_FAIL;
      break;
    }

    size_t jpg_len = 0;
    uint8_t *jpg_buf = NULL;
    bool converted = false;
    if (fb->format != PIXFORMAT_JPEG) {
      converted = frame2jpg(fb, 80, &jpg_buf, &jpg_len);
      esp_camera_fb_return(fb);
      fb = NULL;
      if (!converted) { res = ESP_FAIL; break; }
    } else {
      jpg_len = fb->len;
      jpg_buf = fb->buf;
    }

    res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if (res == ESP_OK) {
      size_t hlen = snprintf(part_buf, sizeof(part_buf), STREAM_PART, jpg_len);
      res = httpd_resp_send_chunk(req, part_buf, hlen);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)jpg_buf, jpg_len);
    }

    if (fb) {
      esp_camera_fb_return(fb);
    } else if (converted && jpg_buf) {
      free(jpg_buf);
    }

    // The client disconnected (the perception loop stopped or the link
    // dropped). Return so the framebuffer is released rather than spinning.
    if (res != ESP_OK) break;
  }
  return res;
}

static esp_err_t capture_handler(httpd_req_t *req) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }
  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return res;
}

static esp_err_t control_handler(httpd_req_t *req) {
  char query[128], variable[32], value[32];
  if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK ||
      httpd_query_key_value(query, "var", variable, sizeof(variable)) != ESP_OK ||
      httpd_query_key_value(query, "val", value, sizeof(value)) != ESP_OK) {
    httpd_resp_send_404(req);
    return ESP_FAIL;
  }

  int val = atoi(value);
  sensor_t *s = esp_camera_sensor_get();
  int res = 0;

  if (!strcmp(variable, "quality"))          res = s->set_quality(s, val);
  else if (!strcmp(variable, "contrast"))    res = s->set_contrast(s, val);
  else if (!strcmp(variable, "brightness"))  res = s->set_brightness(s, val);
  else if (!strcmp(variable, "saturation"))  res = s->set_saturation(s, val);
  else if (!strcmp(variable, "gainceiling")) res = s->set_gainceiling(s, (gainceiling_t)val);
  else if (!strcmp(variable, "awb"))         res = s->set_whitebal(s, val);
  else if (!strcmp(variable, "aec"))         res = s->set_exposure_ctrl(s, val);
  else if (!strcmp(variable, "agc"))         res = s->set_gain_ctrl(s, val);
  else if (!strcmp(variable, "vflip"))       res = s->set_vflip(s, val);
  else if (!strcmp(variable, "hmirror"))     res = s->set_hmirror(s, val);
  else if (!strcmp(variable, "led"))         { pinMode(LED_GPIO_NUM, OUTPUT);
                                               digitalWrite(LED_GPIO_NUM, val ? HIGH : LOW); }
  // Changing framesize mid-run invalidates the camera intrinsics the rover is
  // using for range estimation, so it is deliberately not exposed here.
  else res = -1;

  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  if (res) { httpd_resp_send_500(req); return ESP_FAIL; }
  httpd_resp_set_type(req, "application/json");
  return httpd_resp_send(req, "{\"ok\":true}", HTTPD_RESP_USE_STRLEN);
}

static esp_err_t status_handler(httpd_req_t *req) {
  sensor_t *s = esp_camera_sensor_get();
  char json[320];
  snprintf(json, sizeof(json),
           "{\"framesize\":%u,\"quality\":%u,\"brightness\":%d,\"contrast\":%d,"
           "\"saturation\":%d,\"awb\":%u,\"aec\":%u,\"agc\":%u,"
           "\"psram\":%s,\"heap\":%u}",
           s->status.framesize, s->status.quality, s->status.brightness,
           s->status.contrast, s->status.saturation, s->status.awb,
           s->status.aec, s->status.agc,
           psramFound() ? "true" : "false", (unsigned)ESP.getFreeHeap());
  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, json, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t index_handler(httpd_req_t *req) {
  char page[560];
  String ip = (WiFi.getMode() & WIFI_MODE_AP) ? WiFi.softAPIP().toString()
                                              : WiFi.localIP().toString();
  snprintf(page, sizeof(page),
           "<!doctype html><meta name=viewport content='width=device-width'>"
           "<title>PolliVision Camera</title>"
           "<body style='font-family:system-ui;margin:2rem;max-width:40rem'>"
           "<h2>PolliVision ESP32-CAM</h2>"
           "<p>Stream URL for the perception stack:</p>"
           "<pre>http://%s:81/stream</pre>"
           "<p>Run it with:</p>"
           "<pre>pollivision stream --esp32 --url http://%s:81/stream</pre>"
           "<p><img src='http://%s:81/stream' style='max-width:100%%'></p>"
           "<p><a href='/status'>status</a> &middot; "
           "<a href='/capture'>still capture</a></p></body>",
           ip.c_str(), ip.c_str(), ip.c_str());
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, page, HTTPD_RESP_USE_STRLEN);
}

static void startServers() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.ctrl_port   = 32768;
  config.max_uri_handlers = 8;

  httpd_uri_t index_uri   = {"/",        HTTP_GET, index_handler,   NULL};
  httpd_uri_t status_uri  = {"/status",  HTTP_GET, status_handler,  NULL};
  httpd_uri_t capture_uri = {"/capture", HTTP_GET, capture_handler, NULL};
  httpd_uri_t control_uri = {"/control", HTTP_GET, control_handler, NULL};

  if (httpd_start(&camera_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &index_uri);
    httpd_register_uri_handler(camera_httpd, &status_uri);
    httpd_register_uri_handler(camera_httpd, &capture_uri);
    httpd_register_uri_handler(camera_httpd, &control_uri);
  }

  // The stream lives on its own server so a slow or stalled stream client
  // cannot block the control endpoints.
  config.server_port = 81;
  config.ctrl_port   = 32769;
  httpd_uri_t stream_uri = {"/stream", HTTP_GET, stream_handler, NULL};
  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}

// ---------------------------------------------------------------------------

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);
  Serial.println("\nPolliVision ESP32-CAM starting");

  pinMode(LED_GPIO_NUM, OUTPUT);
  digitalWrite(LED_GPIO_NUM, LOW);

  if (!initCamera()) {
    Serial.println("Camera init failed; halting.");
    while (true) delay(1000);
  }
  Serial.printf("PSRAM: %s\n", psramFound() ? "yes (SVGA)" : "no (VGA fallback)");

  if (strlen(WIFI_SSID) > 0) {
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.print("Joining ");
    Serial.print(WIFI_SSID);
    for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) {
      delay(500);
      Serial.print('.');
    }
    Serial.println();
  }

  if (WiFi.status() != WL_CONNECTED) {
    // Access-point mode: no infrastructure needed in the field, and the rover
    // keeps its link to the camera even with no network for kilometres.
    WiFi.mode(WIFI_AP);
    WiFi.softAP(AP_SSID, AP_PASSWORD);
    Serial.printf("Access point '%s' at %s\n", AP_SSID,
                  WiFi.softAPIP().toString().c_str());
  } else {
    Serial.printf("Connected, IP %s\n", WiFi.localIP().toString().c_str());
  }

  startServers();
  String ip = (WiFi.getMode() & WIFI_MODE_AP) ? WiFi.softAPIP().toString()
                                              : WiFi.localIP().toString();
  Serial.printf("Stream:  http://%s:81/stream\n", ip.c_str());
  Serial.printf("Capture: http://%s/capture\n", ip.c_str());
}

void loop() {
  // Everything runs in the HTTP server tasks.
  delay(2000);
}
