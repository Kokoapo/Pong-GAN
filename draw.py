import sys
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pygame as pg
import pygame.freetype
DeepQNetwork = None
from pong import Pong, StepCondition


WIDTH = 800
HEIGHT = 600
BLACK = pg.Color(0, 0, 0)
WHITE = pg.Color(255, 255, 255)
PAD_WIDTH = 5


def main(p1_use_ai, p2_use_ai):
    pg.init()
    pygame.display.set_caption("Pong!")
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    clock = pg.time.Clock()
    pong = Pong(WIDTH, HEIGHT)
    font = pg.freetype.Font("./F77MinecraftRegular-0VYv.ttf", 24)

    if p1_use_ai:
        p1_model = DeepQNetwork((6,), 3)
        p1_model.load("modelo_p1.weights.h5")
        p1_model.epsilon = 0.0

    if p2_use_ai:
        p2_model = DeepQNetwork((6,), 3)
        p2_model.load("modelo_p2.weights.h5")
        p2_model.epsilon = 0.0

    p1_action = 0
    p2_action = 0

    frame = 0
    action_interval = 8
    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        keys = pg.key.get_pressed()
        if keys[pg.K_q] or keys[pg.K_ESCAPE]:
            running = False

        state = pong.state()

        if not p1_use_ai:
            p1_action = -keys[pg.K_w] + keys[pg.K_s]
        elif frame % action_interval == 0:
            p1_action = p1_model.agir(state) - 1

        if not p2_use_ai:
            p2_action = -keys[pg.K_k] + keys[pg.K_j]
        elif (frame + action_interval//2) % action_interval == 0:
            p2_action = p2_model.agir(state) - 1

        condition = pong.step(p1_action, p2_action)

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
        frame += 1

    pg.quit()


if __name__ == "__main__":
    msg = """\
Modo de Jogo:\n\
(1) CPU vs CPU\n\
(2) CPU vs P2\n\
(3) P1 vs CPU\n\
(4) P1 vs P2\n\
> \
"""
    match input(msg):
        case "1":
            p1_use_ai = True
            p2_use_ai = True
        case "2":
            p1_use_ai = True
            p2_use_ai = False
        case "3":
            p1_use_ai = False
            p2_use_ai = True
        case "4":
            p1_use_ai = False
            p2_use_ai = False
        case _:
            p1_use_ai = True
            p2_use_ai = True
    if p1_use_ai or p2_use_ai:
        DeepQNetwork = __import__("rede_neural").DeepQNetwork
    main(p1_use_ai, p2_use_ai)
