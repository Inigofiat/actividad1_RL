import gymnasium as gym
from deustorl.helpers import DiscretizedObservationWrapper

env_name = "Acrobot-v1"

PRESUPUESTOS= {
    "bajo": 4,
    "medio": 6,
    "alto": 8
}

def crear_entorno(presupuesto, render_mode=None):
    if render_mode is not None:
        return DiscretizedObservationWrapper(gym.make(env_name, render_mode=render_mode), n_bins=PRESUPUESTOS[presupuesto])
    else:
        return DiscretizedObservationWrapper(gym.make(env_name), n_bins=PRESUPUESTOS[presupuesto])