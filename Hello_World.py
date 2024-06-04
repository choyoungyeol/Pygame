import pygame

# Pygame 초기화
pygame.init()

# 창 크기 설정
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# 창 설정
display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Hello World!")

# 폰트 설정
font = pygame.font.Font(None, 72)  # 폰트 크기 36으로 설정

# 게임이 동작하는 동안 이벤트
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 화면에 "Hello World!" 텍스트 출력
    text = font.render("Hello World!", True, (0, 0, 0))  # 검은색 텍스트 생성
    text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))  # 텍스트 위치 설정
    display_surface.fill((255, 255, 255))  # 화면을 하얀색으로 채우기
    display_surface.blit(text, text_rect)  # 텍스트를 화면에 그리기

    pygame.display.update()  # 화면 업데이트

pygame.quit()
