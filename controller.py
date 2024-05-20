from rede_neural import DeepQNetwork
from pong import Pong, StepCondition

def main():
    n_episodios = 1000
    c_save = 10
    c_copia = 20
    t_batch = 32

    p1 = DeepQNetwork((6,), 3)
    p2 = DeepQNetwork((6,), 3)
    env = Pong(800, 600)

    for ep in range(n_episodios):
        while True:
            estado = env.reset()
            pass

if __name__ == "__main__":
    main()