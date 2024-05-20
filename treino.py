from rede_neural import DeepQNetwork
from pong import Pong, StepCondition

def main():
    n_episodios = 1000
    c_save = 50
    c_copia = 20
    t_batch = 32

    p1 = DeepQNetwork((1,), 3)
    p2 = DeepQNetwork((1,), 3)
    env = Pong(800, 600)

    for ep in range(n_episodios):
        env.reset()
        estado = env.state()
        fim = False
        while not fim:
            acao_p1 = p1.agir(estado)
            acao_p2 = p2.agir(estado)
            condition = env.step(acao_p1, acao_p2)
            proximo_estado = env.state()
            recompensa_p1 = recompensa_p2 = 0
            match condition:
                case StepCondition.Player1Hit:
                    recompensa_p1 = 5
                case StepCondition.Player2Hit:
                    recompensa_p2 = 5
                case StepCondition.Player1Score:
                    recompensa_p1 = 10
                    recompensa_p2 = -10
                    fim = True
                case StepCondition.Player2Score:
                    recompensa_p2 = 10
                    recompensa_p1 = -10
                    fim = True
            p1.memorizar(estado, acao_p1, recompensa_p1, proximo_estado, fim)
            p2.memorizar(estado, acao_p2, recompensa_p2, proximo_estado, fim)
            estado = proximo_estado
        mse_p1 = p1.replay(t_batch)
        mse_p2 = p2.replay(t_batch)

        if ep % c_save == 0:
            p1.save("modelo_p1.weights.h5")
            p2.save("modelo_p2.weights.h5")
        if ep % c_copia == 0:
            p1.update_alvo()
            p2.update_alvo()

        print("Episodio {}: P1 = {} P2 = {}".format(ep, mse_p1, mse_p2))
        

if __name__ == "__main__":
    main()