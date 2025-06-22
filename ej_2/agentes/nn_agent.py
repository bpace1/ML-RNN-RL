# nn_agent.py

from agentes.base import Agent
import numpy as np
import tensorflow as tf
import random
from typing import List, Any, Dict

DEFAULT_MODEL_PATH = 'models/flappy_q_nn_model.keras'

class NNAgent(Agent):
    """
    DQN-based Agent for Flappy Bird that uses a pretrained neural network to approximate Q-values.
    Supports optional epsilon-greedy exploration during training.
    """

    def __init__(
        self,
        actions: List[Any],
        game: Any = None,
        model_path: str = DEFAULT_MODEL_PATH,
        epsilon: float = 0.1,
        min_epsilon: float = 0.01,
        decay_rate: float = 0.995
    ):
        """
        :param actions: list of possible actions
        :param game:       reference to the game environment (optional)
        :param model_path: path to the saved TensorFlow model
        :param epsilon:    initial exploration probability
        :param min_epsilon:minimum exploration probability
        :param decay_rate: rate at which epsilon decays after each action
        """
        super().__init__(actions, game)
        # Cargar el modelo entrenado
        self.model: tf.keras.Model = tf.keras.models.load_model(model_path)
        # Parámetros de exploración
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

    def _extract_features(self, state: Dict[str, float]) -> np.ndarray:
        """
        Convierte el estado crudo en un vector de características continuas para la red.
        En lugar de discretizar, utilizamos:
          - delta_y: distancia vertical del jugador al centro del próximo tubo
          - player_vel: velocidad actual del jugador

        :param state: diccionario con llaves 'player_y', 'next_pipe_top_y', 'next_pipe_bottom_y', 'player_vel'
        :return: np.array de shape (2,) dtype float32
        """
        player_y = state['player_y']
        top_y = state['next_pipe_top_y']
        bot_y = state['next_pipe_bottom_y']
        pipe_center = (top_y + bot_y) / 2.0
        delta_y = player_y - pipe_center
        player_vel = state['player_vel']
        return np.array([delta_y, player_vel], dtype=np.float32)

    def act(self, state: Dict[str, float], training: bool = False) -> Any:
        """
        Elige una acción dado el estado actual.
        - Si training=True, aplica epsilon-greedy.
        - En modo evaluación, siempre aprovecha la Q máxima.

        :param state:    estado actual del juego
        :param training: si es True, permite exploración aleatoria
        :return:         acción seleccionada de self.actions
        """
        features = self._extract_features(state)
        # Formatear batch de tamaño 1
        batch = np.expand_dims(features, axis=0)

        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_values = self.model.predict(batch, verbose=0)[0]
            idx = int(np.argmax(q_values))
            action = self.actions[idx]

        # Decaer epsilon tras cada paso de entrenamiento
        if training:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

        return action
