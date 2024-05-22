import time
from rede_neural import DeepQNetwork
from pong import Pong, StepCondition

def main():
    n_episodios = 7000
    epsilon_decay = (1-0.1)/n_episodios
    c_save = 50
    c_copy = 100
    t_batch = 20

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
                    recompensa_p1 = 100
                case StepCondition.Player2Hit:
                    recompensa_p2 = 100
                case StepCondition.Player1Score:
                    recompensa_p1 = 50
                    recompensa_p2 = -50
                    fim = True
                case StepCondition.Player2Score:
                    recompensa_p2 = 50
                    recompensa_p1 = -50
                    fim = True
            
            p1.memorizar(estado, acao_p1, recompensa_p1, proximo_estado, fim)
            p2.memorizar(estado, acao_p2, recompensa_p2, proximo_estado, fim)
            estado = proximo_estado
        if ep % c_copy == 0:
            p1.update_alvo()
            p2.update_alvo()
        
        media_loss_p1 = p1.replay(t_batch)
        media_loss_p2 = p2.replay(t_batch)

        p1.update_epsilon(epsilon_decay)
        p2.update_epsilon(epsilon_decay)

        if ep % c_save == 0:
            p1.save("savesp2/modelo_{}.weights.h5".format(ep))
            p2.save("savesp1/modelo_{}.weights.h5".format(ep))

        tempo_fim = time.time() - tempo_init
        print("Episodio {} : \t\t P1 - {:.10f} \t\t\t P2 - {:.10f} \t\t\t Tempo : {:.5f}".format(ep, media_loss_p1, media_loss_p2, tempo_fim))
        

if __name__ == "__main__":
    main()