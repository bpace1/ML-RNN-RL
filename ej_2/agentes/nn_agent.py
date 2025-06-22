from agentes.base import Agent
import numpy as np
import tensorflow as tf

model_path = 'models/flappy_q_nn_model.keras'

class NNAgent(Agent):
    """
    Agente que utiliza una red neuronal entrenada para aproximar la Q-table.
    La red debe estar guardada como TensorFlow SavedModel.
    """
    def __init__(self, actions, game=None, model_path=model_path):
        super().__init__(actions, game)
        
        self.model = tf.keras.models.load_model(model_path)
    
    def _discretize_state(self, state):
        """Convierte el estado continuo en características discretas"""
        player_y = state['player_y']
        pipe_top_y = state['next_pipe_top_y']
        pipe_bottom_y = state['next_pipe_bottom_y']
        pipe_center_y = (pipe_bottom_y - pipe_top_y) / 2 + pipe_top_y
        
        # Discretizar posición vertical del jugador (0-3)
        if player_y < pipe_top_y:
            player_y_bin = 0    # Por encima del tubo superior
        elif player_y > pipe_bottom_y:
            player_y_bin = 1    # Por debajo del tubo inferior
        else:
            if player_y < pipe_center_y:    
                player_y_bin = 2    # Entre tubos, cerca del superior
            else:
                player_y_bin = 3    # Entre tubos, cerca del inferior
        
        # Discretizar velocidad del jugador (-2 a 2)
        player_vel = state['player_vel']
        if player_vel > 8:
            player_vel_bin = -2   # Muy rápido descendiendo
        elif player_vel > 3:
            player_vel_bin = -1   # Rápido descendiendo
        elif player_vel > -3:
            player_vel_bin = 0    # Estable
        elif player_vel > -8:
            player_vel_bin = 1    # Rápido ascendiendo
        else:
            player_vel_bin = 2    # Muy rápido ascendiendo
            
        return np.array([player_y_bin, player_vel_bin], dtype=np.float32)


    
    def act(self, state):
        """
        COMPLETAR: Implementar la función de acción.
        Debe transformar el estado al formato de entrada de la red y devolver la acción con mayor Q.
        """
        #print(state)
        player_y = state['player_y']
        pipe_bottom_y = state['next_pipe_bottom_y']
        pipe_top_y = state['next_pipe_top_y']
        pipe_center_y = (pipe_bottom_y - pipe_top_y) / 2 + pipe_top_y
        player_y_bin = 0  # Inicializar variable de discretización de la posición del jugador
        if player_y < pipe_top_y:
            player_y_bin = 0    # Por encima del tubo superior
        elif player_y > pipe_bottom_y:
            player_y_bin = 1    # Por debajo del tubo inferior
        elif player_y > pipe_top_y and player_y < pipe_bottom_y:
            if player_y < pipe_center_y:    
                player_y_bin = 2    # Entre los tubos, más cerca del tubo superior
            else:
                player_y_bin = 3    # Entre los tubos, más cerca del tubo inferior
        player_vel = state['player_vel']
        
        if player_vel > 8:
            player_vel_bin = -2   # Muy rápido descendiendo
        elif player_vel > 3:
            player_vel_bin = -1   # Rápido descendiendo
        elif player_vel > -3:
            player_vel_bin = 0    # Estable
        elif player_vel > -8:
            player_vel_bin = 1    # Rápido ascendiendo
        else:
            player_vel_bin = 2    # Muy rápido ascendiendo
        action_prob = self.model.predict(np.array([player_y_bin,player_vel_bin],), verbose=0)
        print(action_prob)
        predicted_action_idx = np.argmax(action_prob)
       
        return self.actions[predicted_action_idx]