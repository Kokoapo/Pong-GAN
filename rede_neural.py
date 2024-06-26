import numpy as np;
import tensorflow as tf;
import keras;
import random;
from collections import deque

class DeepQNetwork:
    def __init__(self, n_entradas, n_saidas):
        self.n_entradas = n_entradas
        self.n_saidas = n_saidas
        self.memoria = deque(maxlen=8000)

        self.gamma = 0.99   # discount rate
        self.epsilon = 1.0  # exploration rate
        self.alpha = 0.01  # learning rate

        self.modelo_main = self.criar_modelo()
        self.modelo_alvo = self.criar_modelo()

    def criar_modelo(self):
        modelo = keras.models.Sequential()
        modelo.add(keras.layers.InputLayer(shape=self.n_entradas))
        modelo.add(keras.layers.Dense(512, activation='relu'))
        modelo.add(keras.layers.Dense(256, activation='relu'))
        modelo.add(keras.layers.Dense(256, activation='relu'))
        modelo.add(keras.layers.Dense(self.n_saidas, activation='linear'))
        modelo.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.alpha), metrics=['mse'])
        return modelo
    
    def update_epsilon(self, decay):
        if self.epsilon > 0.0001:
            self.epsilon -= decay

    def update_alvo(self):
        self.modelo_alvo.set_weights(self.modelo_main.get_weights())

    def memorizar(self, estado_atual, acao, recompensa, proximo_estado, fim):
        estado_atual = self.reshape_estado(estado_atual)
        proximo_estado = self.reshape_estado(proximo_estado)
        self.memoria.append((estado_atual, acao, recompensa, proximo_estado, fim))

    def reshape_estado(self, estado):
        return np.reshape(estado, [1] + list(self.n_entradas))

    def agir(self, estado):
        estado = self.reshape_estado(estado)
        if random.random() < self.epsilon:
            return random.randint(0, self.n_saidas-1)
        q_values = self.modelo_main.predict(estado, verbose=0)[0]
        return np.argmax(q_values)

    def replay(self, t_batch):
        minibatch = random.sample(self.memoria, t_batch)
        media_loss = 0.0
        for estado_atual, acao, recompensa, proximo_estado, fim in minibatch:
            q_values = self.modelo_main.predict(estado_atual, verbose=0)
            if fim:
                q_values[0][acao] = recompensa
            else:
                t = self.modelo_alvo.predict(proximo_estado, verbose=0)[0]
                q_values[0][acao] = recompensa + self.gamma * np.amax(t)
            history = self.modelo_main.fit(estado_atual, q_values, epochs=1, verbose=0)

            media_loss += history.history['mse'][0]

        media_loss /= t_batch
        return media_loss

    def load(self, name):
        self.modelo_main.load_weights(name)
        self.modelo_alvo.load_weights(name)

    def save(self, name):
        self.modelo_main.save_weights(name)