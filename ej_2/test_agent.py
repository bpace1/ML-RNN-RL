from ple.games.flappybird import FlappyBird
from ple import PLE
import time
import argparse
import importlib
import sys
import numpy as np

# --- Configuración del Entorno y Agente ---
# Inicializar el juego
game = FlappyBird()  # Usar FlappyBird en vez de Pong
env = PLE(game, display_screen=True, fps=30) # fps=30 es más normal, display_screen=True para ver

# Inicializar el entorno
env.init()

# Obtener acciones posibles
actions = env.getActionSet()

# --- Argumentos ---
parser = argparse.ArgumentParser(description="Test de agentes para FlappyBird (PLE)")
parser.add_argument('--agent', type=str, required=True, help='Ruta completa del agente, ej: agentes.random_agent.RandomAgent')
parser.add_argument('--qtable', type=str, help='Ruta al archivo de Q-table para cargar (solo para QAgent)')
args = parser.parse_args()

# --- Carga dinámica del agente usando path completo ---
try:
    module_path, class_name = args.agent.rsplit('.', 1)
    agent_module = importlib.import_module(module_path)
    AgentClass = getattr(agent_module, class_name)
except (ValueError, ModuleNotFoundError, AttributeError):
    print(f"No se pudo encontrar la clase {args.agent}")
    sys.exit(1)

# Inicializar el agente
agent = AgentClass(actions, game)

# Si es un QAgent y se proporcionó una Q-table, cargarla y desactivar exploración
if hasattr(agent, 'load_q_table') and args.qtable:
    try:
        q_table = np.load(args.qtable)
        agent.load_q_table(q_table)
        print(f"Q-table cargada desde {args.qtable}")
        
        # Desactivar exploración si el agente tiene el atributo
        if hasattr(agent, 'epsilon'):
            agent.epsilon = 0  # Sin exploración, solo explotación
            print("Exploración desactivada (epsilon=0)")
    except Exception as e:
        print(f"Error al cargar la Q-table: {e}")
        sys.exit(1)

# Agente con acciones aleatorias
while True:
    env.reset_game()
    agent.reset()
    state_dict = env.getGameState()
    done = False
    total_reward_episode = 0
    print("\n--- Ejecutando agente ---")
    while not done:
        action = agent.act(state_dict)
        reward = env.act(action)
        state_dict = env.getGameState()
        done = env.game_over()
        total_reward_episode += reward
        time.sleep(0.03)
        #print(f"Estado: {state_dict}")
    print(f"Recompensa episodio: {total_reward_episode}")