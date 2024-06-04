import pygame
import serial
import time

# Pygame 초기화
pygame.init()

# 화면 설정
screen_size = (800, 600)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Greenhouse Environment')

# 폰트 설정
font = pygame.font.Font(None, 72)  # 폰트 크기 36으로 설정

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

# 게임이 동작하는 동안 이벤트
running = True
temperature = "N/A"
humidity = "N/A"
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
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
    screen.blit(hum_text, (50, 300))

    pygame.display.flip()

    # 프레임 속도 설정
    pygame.time.Clock().tick(60)

# Pygame 종료
pygame.quit()

# 시리얼 통신 종료
arduino.close()
