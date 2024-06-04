import pygame
import serial
import time

# Pygame 초기화
pygame.init()

# 화면 설정
screen_size = (800, 600)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Pygame Arduino Example')

# 폰트 설정
font = pygame.font.Font(None, 74)
button_font = pygame.font.Font(None, 50)

# 색상 설정
white = (255, 255, 255)
black = (0, 0, 0)
blue = (0, 0, 255)
red = (255, 0, 0)

# Arduino와 시리얼 통신 설정
try:
    arduino = serial.Serial('COM25', 9600)  # COM 포트는 실제 연결된 포트로 변경해야 합니다.
    time.sleep(2)  # 시리얼 통신이 안정될 때까지 대기
except serial.SerialException as e:
    print(f"Error opening the serial port: {e}")
    pygame.quit()
    exit()

# 버튼 클래스 정의
class Button:
    def __init__(self, text, pos, color, action):
        self.text = text
        self.pos = pos
        self.color = color
        self.action = action
        self.rect = pygame.Rect(pos[0], pos[1], 200, 100)
    
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        text_surf = button_font.render(self.text, True, white)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
    
    def is_clicked(self, event):
        return self.rect.collidepoint(event.pos)

# 버튼 생성
button_a = Button("ON", (150, 450), blue, 'a')
button_b = Button("OFF", (450, 450), red, 'b')

# 메인 루프
running = True
temperature = "N/A"
humidity = "N/A"

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button_a.is_clicked(event):
                arduino.write(b'a')
            elif button_b.is_clicked(event):
                arduino.write(b'b')
    
    # 시리얼 데이터 읽기
    if arduino.in_waiting > 0:
        line = arduino.readline().decode('utf-8').strip()
        if ", " in line:
            temperature, humidity = line.split(", ")
    
    # 화면 그리기
    screen.fill(white)
    temp_text = font.render(f"Temperature: {temperature} °C", True, black)
    hum_text = font.render(f"Humidity: {humidity} %", True, black)
    screen.blit(temp_text, (50, 150))
    screen.blit(hum_text, (50, 300)
    
    # 버튼 그리기
    button_a.draw(screen)
    button_b.draw(screen)
    
    pygame.display.flip()
    
    # 프레임 속도 설정
    pygame.time.Clock().tick(60)

# Pygame 종료
pygame.quit()

# 시리얼 통신 종료
arduino.close()
