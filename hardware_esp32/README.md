# NeuroHome physical appliance bridge — ESP32

This bridge receives real HTTP commands from the NeuroHome FastAPI backend.
It is intended for a 5-channel relay board driving appropriate **isolated, correctly rated loads**.

## Commands
- LIGHT_ON
- FAN_ON
- TV_ON
- AC_ON
- DOOR_OPEN
- ALL_OFF

## Setup
1. Install the ESP32 board package in Arduino IDE.
2. Install the ArduinoJson library.
3. Put your Wi-Fi SSID/password in `NeuroHome_ESP32.ino`.
4. Wire the relay inputs to GPIO 16,17,18,19,21 (or change the constants).
5. Upload the sketch. Open Serial Monitor at 115200 and note the ESP32 IP address.
6. On the NeuroHome backend PC, set:
   `$env:NEUROHOME_ESP32_URL = "http://ESP32_IP"`
7. Start the backend. Check `GET /hardware/status`.

IMPORTANT: Do not connect mains voltage directly to an ESP32. Use an appropriate, enclosed, isolated relay/contactor and qualified electrical installation. For a college demo, use low-voltage loads/LEDs if possible.
