import pygame
import serial
import time

# Pygame 초기화
pygame.init()

# 화면 설정
screen_size = (800, 600)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Greenhouse DHT11 Sensor Readings')

# 폰트 설정
font = pygame.font.Font(None, 30)
button_font = pygame.font.Font(None, 30)

# 색상 설정
white = (255, 255, 255)
black = (0, 0, 0)
blue = (0, 0, 255)
red = (255, 0, 0)
green = (0, 255, 0)
gray = (128, 128, 128)

# Arduino와 시리얼 통신 설정
try:
    arduino = serial.Serial('COM25', 9600)  # COM 포트는 실제 연결된 포트로 변경해야 합니다.
    time.sleep(2)  # 시리얼 통신이 안정될 때까지 대기
except serial.SerialException as e:
    print(f"Error opening the serial port: {e}")
    pygame.quit()
    exit()


# 버튼 정의
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


# 난방기 프레임 정의
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


# 버튼 생성
button_on = Button("ON", (570, 400), blue, 'a')
button_off = Button("OFF", (570, 450), red, 'b')

# 메인 루프
running = True
temperature = "N/A"
humidity = "N/A"
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button_on.is_clicked(event):
                arduino.write(b'a')  # 버튼 'a' 클릭 시 'a'를 Arduino로 전송
            elif button_off.is_clicked(event):
                arduino.write(b'b')  # 버튼 'b' 클릭 시 'b'를 Arduino로 전송

    # 시리얼 데이터 읽기
    if arduino.in_waiting > 0:
        line = arduino.readline().decode('utf-8').strip()
        if ", " in line:
            temperature, humidity = line.split(", ")

    # 화면 그리기
    screen.fill(white)

    # 온실 그리기
    pygame.draw.rect(screen, white, (100, 200, 600, 300))
    pygame.draw.rect(screen, black, (100, 200, 600, 300), 2)
    pygame.draw.polygon(screen, white, [(100, 200), (400, 100), (700, 200)])
    pygame.draw.polygon(screen, black, [(100, 200), (400, 100), (700, 200)], 2)

    # 난방기 프레임 그리기
    heater_frame = HeaterFrame()
    heater_frame.draw(screen)

    # 버튼 그리기
    button_on.draw(screen)
    button_off.draw(screen)

    # 온도와 습도 표시
    temp_text = font.render(f"Temperature: {temperature} °C", True, black)
    hum_text = font.render(f"Humidity: {humidity} %", True, black)
    screen.blit(temp_text, (150, 250))
    screen.blit(hum_text, (150, 300))

    pygame.display.flip()

    # 프레임 속도 설정
    pygame.time.Clock().tick(60)

# Pygame 종료
pygame.quit()
# 시리얼 통신 종료
arduino.close()

