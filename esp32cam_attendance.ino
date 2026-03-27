/*
 * ESP32-CAM Facial Recognition Attendance System
 * 
 * Board: AI-Thinker ESP32-CAM
 * 
 * Wiring:
 *   Buzzer positive → GPIO 12
 *   Buzzer negative → GND
 * 
 * How it works:
 *   1. Connects to WiFi
 *   2. Captures a JPEG frame every 500ms
 *   3. POSTs it to Railway backend /api/recognize-and-mark/
 *   4. If recognized → 1 short beep
 *   5. If already marked → 2 short beeps
 *   6. If unknown → 3 rapid beeps
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"
#include "Arduino.h"

// ── WiFi credentials ──────────────────────────────────────
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// ── Backend ───────────────────────────────────────────────
const char* SERVER_URL = "https://facial-recognition-attendance-backend-production.up.railway.app/api/recognize-and-mark/";
const char* CLASSROOM  = "301";  // Change to your classroom

// ── Pins ──────────────────────────────────────────────────
#define BUZZER_PIN 12

// ── AI-Thinker ESP32-CAM pin map ─────────────────────────
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


// ── Buzzer helpers ────────────────────────────────────────

void beep(int times, int duration_ms, int pause_ms = 100) {
  for (int i = 0; i < times; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(duration_ms);
    digitalWrite(BUZZER_PIN, LOW);
    if (i < times - 1) delay(pause_ms);
  }
}

void beep_recognized()    { beep(1, 400); }           // Single long beep
void beep_already_marked(){ beep(2, 150, 80); }       // Two short beeps
void beep_unknown()       { beep(3, 80, 60); }        // Three rapid beeps
void beep_error()         { beep(1, 100); delay(100); beep(1, 100); } // Two quick


// ── Camera setup ──────────────────────────────────────────

bool init_camera() {
  camera_config_t config;
  config.ledc_channel  = LEDC_CHANNEL_0;
  config.ledc_timer    = LEDC_TIMER_0;
  config.pin_d0        = Y2_GPIO_NUM;
  config.pin_d1        = Y3_GPIO_NUM;
  config.pin_d2        = Y4_GPIO_NUM;
  config.pin_d3        = Y5_GPIO_NUM;
  config.pin_d4        = Y6_GPIO_NUM;
  config.pin_d5        = Y7_GPIO_NUM;
  config.pin_d6        = Y8_GPIO_NUM;
  config.pin_d7        = Y9_GPIO_NUM;
  config.pin_xclk      = XCLK_GPIO_NUM;
  config.pin_pclk      = PCLK_GPIO_NUM;
  config.pin_vsync     = VSYNC_GPIO_NUM;
  config.pin_href      = HREF_GPIO_NUM;
  config.pin_sscb_sda  = SIOD_GPIO_NUM;
  config.pin_sscb_scl  = SIOC_GPIO_NUM;
  config.pin_pwdn      = PWDN_GPIO_NUM;
  config.pin_reset     = RESET_GPIO_NUM;
  config.xclk_freq_hz  = 20000000;
  config.pixel_format  = PIXFORMAT_JPEG;
  config.frame_size    = FRAMESIZE_QVGA;  // 320x240 — good balance of speed/quality
  config.jpeg_quality  = 12;
  config.fb_count      = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }
  Serial.println("Camera ready.");
  return true;
}


// ── Setup ─────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  // Connect WiFi
  Serial.print("Connecting to WiFi");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\nWiFi failed. Restarting...");
    beep_error();
    delay(3000);
    ESP.restart();
  }

  Serial.println("\nWiFi connected: " + WiFi.localIP().toString());
  beep(1, 200);  // Single short beep = WiFi connected

  if (!init_camera()) {
    beep_error();
    delay(3000);
    ESP.restart();
  }

  Serial.println("System ready. Starting recognition...");
}


// ── Loop ──────────────────────────────────────────────────

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi lost, reconnecting...");
    WiFi.reconnect();
    delay(3000);
    return;
  }

  // Capture frame
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Frame capture failed");
    delay(500);
    return;
  }

  // POST to backend
  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("X-Classroom", CLASSROOM);
  http.setTimeout(8000);

  int httpCode = http.POST(fb->buf, fb->len);
  esp_camera_fb_return(fb);

  if (httpCode == 200) {
    String response = http.getString();
    Serial.println("Response: " + response);

    // Parse response
    if (response.indexOf("\"recognized\":true") >= 0) {
      if (response.indexOf("\"already_marked\":true") >= 0) {
        Serial.println("→ Already marked");
        beep_already_marked();
      } else if (response.indexOf("\"marked\":true") >= 0) {
        Serial.println("→ Attendance marked!");
        beep_recognized();
      } else {
        // Recognized but not marked (not enrolled / no active lecture)
        Serial.println("→ Recognized but not marked");
        beep_error();
      }
    } else {
      Serial.println("→ Unknown face");
      beep_unknown();
    }
  } else {
    Serial.printf("HTTP error: %d\n", httpCode);
    beep_error();
  }

  http.end();
  delay(500);  // ~2 fps
}
