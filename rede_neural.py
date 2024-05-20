import numpy as np;
import tensorflow as tf;
from tensorflow import keras;
import random;
from collections import deque

class DeepQNetwork:
    def __init__(self, n_entradas, n_saidas):
        self.n_entradas = n_entradas
        self.n_saidas = n_saidas
        self.memoria = deque(maxlen=2000)

        self.gamma = 0.95   # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.001  # learning rate

        self.modelo_main = self.criar_modelo()
        self.modelo_alvo = self.criar_modelo()

    def criar_modelo(self):
        modelo = keras.models.Sequential()
        modelo.add(keras.layers.InputLayer(shape=self.n_entradas))
        modelo.add(keras.layers.Dense(512, activation='relu'))
        modelo.add(keras.layers.Dense(self.n_saidas, activation='linear'))
        modelo.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(), metrics=['mse'])
        return modelo

    def update_alvo(self):
        self.modelo_alvo.set_weights(self.modelo_main.get_weights())

    def memorizar(self, estado_atual, acao, recompensa, proximo_estado, fim):
        self.memoria.append((estado_atual, acao, recompensa, proximo_estado, fim))

    def agir(self, estado):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_saidas)
        q_values = self.modelo_main.predict(estado, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, t_batch):
        minibatch = random.sample(self.memoria, t_batch)
        maior_loss = -9999.99
        menor_loss = 99999.99
        for estado_atual, acao, recompensa, proximo_estado, fim in minibatch:
            q_values = self.modelo_main.predict(estado_atual, verbose=0)
            if fim:
                q_values[0][acao] = recompensa
            else:
                t = self.modelo_alvo.predict(proximo_estado, verbose=0)[0]
                q_values[0][acao] = recompensa + self.gamma * np.amax(t)
            history = self.modelo_main.fit(estado_atual, q_values, epochs=1, verbose=0)

            if history.history['mse'][0] > maior_loss:
                maior_loss = history.history['mse'][0]
            elif history.history['mse'][0] < menor_loss:
                menor_loss = history.history['mse'][0]

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        return menor_loss, maior_loss

    def load(self, name):
        self.modelo_main.load_weights(name)
        self.modelo_alvo.load_weights(name)

    def save(self, name):
        self.modelo_main.save_weights(name)