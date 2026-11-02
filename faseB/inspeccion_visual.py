import os
import sys
import random

ACTIVIDAD = os.path.dirname(os.path.dirname(os.path.abspath(_file_)))
sys.path[0:0] = [ACTIVIDAD, os.path.dirname(ACTIVIDAD)]

from deustorl.common import *
from config import crear_entorno, PRESUPUESTOS
from tabla_q import cargar_tabla_q

if _name_ == "_main_":
    os.chdir(os.path.dirname(os.path.abspath(_file_)))

    for presupuesto in PRESUPUESTOS:
        q_table, metadatos = cargar_tabla_q(f"politicas_finales/mejor_{presupuesto}.npz")
        print(f"\n{presupuesto}: {metadatos['algoritmo']} (semilla {metadatos['semilla']})")

        visual_env = crear_entorno(presupuesto, render_mode="human")
        seed = 3
        random.seed(seed)
        visual_env.reset(seed=seed)
        evaluate_policy(visual_env, q_table, max_policy, n_episodes=10, verbose=False)
        visual_env.close()