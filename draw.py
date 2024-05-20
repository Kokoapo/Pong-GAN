import sys
import pygame as pg
import pygame.freetype
from pong import Pong, StepCondition


WIDTH = 800
HEIGHT = 600
BLACK = pg.Color(0, 0, 0)
WHITE = pg.Color(255, 255, 255)
PAD_WIDTH = 5


def main():
    pg.init()
    pygame.display.set_caption("Pong!")
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    clock = pg.time.Clock()
    pong = Pong(WIDTH, HEIGHT)
    font = pg.freetype.Font("./F77MinecraftRegular-0VYv.ttf", 24)

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        keys = pg.key.get_pressed()
        if keys[pg.K_q] or keys[pg.K_ESCAPE]:
            running = False

        if keys[pg.K_w]:
            pong.play1(-1)
        if keys[pg.K_s]:
            pong.play1(1)
        if keys[pg.K_UP] or keys[pg.K_k]:
            pong.play2(-1)
        if keys[pg.K_DOWN] or keys[pg.K_j]:
            pong.play2(1)

        condition = pong.step()

        screen.fill(BLACK)
        # elementos do game
        ball = pg.Rect(
            pong.ball_pos,
            (pong.ball_radius, pong.ball_radius),
        )
        pad1 = pg.Rect(0, pong.p1_pos, PAD_WIDTH, pong.pad_size)
        pad2 = pg.Rect(WIDTH - PAD_WIDTH, pong.p2_pos, PAD_WIDTH, pong.pad_size)
        pg.draw.rect(screen, WHITE, ball)
        pg.draw.rect(screen, WHITE, pad1)
        pg.draw.rect(screen, WHITE, pad2)
        # decoracoes
        pg.draw.rect(screen, WHITE, pg.Rect(0, 0, WIDTH, HEIGHT), 1)
        pg.draw.line(screen, WHITE, (WIDTH / 2, 0), (WIDTH / 2, HEIGHT))
        font.render_to(screen, (10, 10), f"{pong.p1_score:>2} : {pong.p2_score}", WHITE)
        elapsed = pg.time.get_ticks() // 1000
        font.render_to(screen, (WIDTH - 120, 10), f" {elapsed // 60:02} : {elapsed % 60:02}", WHITE)
        pg.display.flip()

        if condition == StepCondition.Player1Score or condition == StepCondition.Player2Score:
            pg.time.delay(500)

        clock.tick(60)

    pg.quit()


if __name__ == "__main__":
    main()
