import time
from rede_neural import DeepQNetwork
from pong import Pong, StepCondition

def main():
    n_episodios = 1000
    epsilon_decay = 1/n_episodios
    c_save = 50
    t_batch = 32

    p1 = DeepQNetwork((6,), 3)
    p2 = DeepQNetwork((6,), 3)
    env = Pong(800, 600)

    for ep in range(n_episodios):
        env.reset()
        estado = env.state()
        
        fim = False
        tempo_init = time.time()
        while not fim:
            acao_p1 = p1.agir(estado)-1
            acao_p2 = p2.agir(estado)-1
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
        p1.update_alvo()
        p2.update_alvo()
        menor_mse_p1, maior_mse_p1 = p1.replay(t_batch)
        menor_mse_p2, maior_mse_p2 = p2.replay(t_batch)

        p1.update_epsilon(epsilon_decay)
        p2.update_epsilon(epsilon_decay)

        if ep % c_save == 0:
            p1.save("modelo_p1.weights.h5")
            p2.save("modelo_p2.weights.h5")

        tempo_fim = time.time() - tempo_init
        print("Episodio {} : \t\t P1 - {} \t + {} \t\t P2 - {} \t + {} \t\t Tempo : {}".format(ep, menor_mse_p1, maior_mse_p1, menor_mse_p2, maior_mse_p2, tempo_fim))
        

if __name__ == "__main__":
    main()