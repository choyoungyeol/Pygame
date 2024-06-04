import pygame
import serial
import time
import cv2
import numpy as np

# Pygame initialization
pygame.init()

# Screen setup
screen_size = (800, 600)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Greenhouse DHT11 Sensor Readings and Camera Feed')

# Font and color setup
font = pygame.font.Font(None, 30)
button_font = pygame.font.Font(None, 30)
white = (255, 255, 255)
black = (0, 0, 0)
blue = (0, 0, 255)
red = (255, 0, 0)
green = (0, 255, 0)
gray = (128, 128, 128)

# Arduino serial communication setup
try:
    arduino = serial.Serial('COM25', 9600)  # Change to the actual connected port
    time.sleep(2)
except serial.SerialException as e:
    print(f"Error opening the serial port: {e}")
    pygame.quit()
    exit()

# Button class definition
class Button:
    def __init__(self, text, pos, color, action):
        self.text = text
        self.pos = pos
        self.color = color
        self.action = action
        self.rect = pygame.Rect(pos[0], pos[1], 100, 50)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        text_surf = button_font.render(self.text, True, white)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, event):
        return self.rect.collidepoint(event.pos)

# Heater frame class definition
class HeaterFrame:
    def __init__(self):
        self.rect = pygame.Rect(570, 400, 100, 100)
        self.color = gray

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        pygame.draw.rect(screen, black, self.rect, 2)
        label_text = font.render("Heater", True, black)
        label_rect = label_text.get_rect(center=(self.rect.centerx, self.rect.top - 20))
        screen.blit(label_text, label_rect)

# Button instantiation
button_on = Button("ON", (570, 400), blue, 'a')
button_off = Button("OFF", (570, 450), red, 'b')

# OpenCV camera setup
cap = cv2.VideoCapture(1)

# Main loop
running = True
temperature = "N/A"
humidity = "N/A"

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button_on.is_clicked(event):
                arduino.write(b'a')
            elif button_off.is_clicked(event):
                arduino.write(b'b')

    if arduino.in_waiting > 0:
        line = arduino.readline().decode('utf-8').strip()
        if ", " in line:
            temperature, humidity = line.split(", ")

    screen.fill(white)

    # Draw greenhouse structure
    pygame.draw.rect(screen, white, (100, 200, 600, 300))
    pygame.draw.rect(screen, black, (100, 200, 600, 300), 2)
    pygame.draw.polygon(screen, white, [(100, 200), (400, 100), (700, 200)])
    pygame.draw.polygon(screen, black, [(100, 200), (400, 100), (700, 200)], 2)

    # Draw heater frame
    heater_frame = HeaterFrame()
    heater_frame.draw(screen)

    # Draw buttons
    button_on.draw(screen)
    button_off.draw(screen)

    # Display temperature and humidity
    temp_text = font.render(f"Temperature: {temperature} °C", True, black)
    hum_text = font.render(f"Humidity: {humidity} %", True, black)
    screen.blit(temp_text, (150, 250))
    screen.blit(hum_text, (150, 300))

    # Capture camera frame
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (200, 150))  # 작은 크기로 조정
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)

        # 온실 내부에 화면 표시
        screen.blit(frame, (530, 50))  # 원하는 위치로 조정

    pygame.display.flip()
    pygame.time.Clock().tick(60)

# Pygame cleanup
pygame.quit()
