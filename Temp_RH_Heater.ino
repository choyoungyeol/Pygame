#define Pump 7

#include "DHT.h"
#define DHTPIN 2
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

void setup() {
  // put your setup code here, to run once:
  pinMode(Pump, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  delay(2000);
  int h = dht.readHumidity();
  int t = dht.readTemperature();
  Serial.print(t);
  Serial.print(", ");
  Serial.println(h);

  if (Serial.available()) {
    delay(3);
    char c = Serial.read();

    if (c == 'a') {
      digitalWrite(Pump, HIGH);
    }
    if (c == 'b') {
      digitalWrite(Pump, LOW);
    }
  }
}
