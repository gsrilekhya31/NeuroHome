#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>

// Set these for your Wi-Fi.
const char* WIFI_SSID = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

WebServer server(80);

// Use a 5-channel relay board. Change pins to match your wiring.
const int RELAY_LIGHT = 16;
const int RELAY_FAN   = 17;
const int RELAY_TV    = 18;
const int RELAY_AC    = 19;
const int RELAY_DOOR  = 21;
const int RELAY_ON = LOW;   // Most relay modules are active-LOW.
const int RELAY_OFF = HIGH;

void allOff(){
  digitalWrite(RELAY_LIGHT, RELAY_OFF); digitalWrite(RELAY_FAN, RELAY_OFF);
  digitalWrite(RELAY_TV, RELAY_OFF); digitalWrite(RELAY_AC, RELAY_OFF);
  digitalWrite(RELAY_DOOR, RELAY_OFF);
}
void command(String c){
  if(c=="LIGHT_ON") digitalWrite(RELAY_LIGHT,RELAY_ON);
  else if(c=="FAN_ON") digitalWrite(RELAY_FAN,RELAY_ON);
  else if(c=="TV_ON") digitalWrite(RELAY_TV,RELAY_ON);
  else if(c=="AC_ON") digitalWrite(RELAY_AC,RELAY_ON);
  else if(c=="DOOR_OPEN") digitalWrite(RELAY_DOOR,RELAY_ON);
  else if(c=="ALL_OFF") allOff();
}
void handleCommand(){
  if(!server.hasArg("plain")){ server.send(400,"application/json","{\"ok\":false,\"error\":\"missing body\"}"); return; }
  JsonDocument doc; if(deserializeJson(doc,server.arg("plain"))){ server.send(400,"application/json","{\"ok\":false,\"error\":\"invalid json\"}"); return; }
  String c=doc["command"] | ""; command(c);
  JsonDocument out; out["ok"]=true; out["command"]=c; String body; serializeJson(out,body); server.send(200,"application/json",body);
}
void handleStatus(){ server.send(200,"application/json","{\"device\":\"NeuroHome ESP32\",\"status\":\"online\"}"); }
void setup(){
  Serial.begin(115200);
  pinMode(RELAY_LIGHT,OUTPUT); pinMode(RELAY_FAN,OUTPUT); pinMode(RELAY_TV,OUTPUT); pinMode(RELAY_AC,OUTPUT); pinMode(RELAY_DOOR,OUTPUT);
  allOff();
  WiFi.begin(WIFI_SSID,WIFI_PASSWORD);
  while(WiFi.status()!=WL_CONNECTED){ delay(500); Serial.print("."); }
  Serial.println(); Serial.print("ESP32 IP: "); Serial.println(WiFi.localIP());
  server.on("/status",HTTP_GET,handleStatus); server.on("/command",HTTP_POST,handleCommand); server.begin();
}
void loop(){ server.handleClient(); }
