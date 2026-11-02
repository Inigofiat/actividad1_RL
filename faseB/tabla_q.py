"""Fase B, punto 1: guardar y cargar la tabla Q en un archivo."""
import os
import sys
import json
import numpy as np

ACTIVIDAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[0:0] = [ACTIVIDAD, os.path.dirname(ACTIVIDAD)]  # para config.py y deustorl/

from deustorl.common import QTable


def guardar_tabla_q(ruta, q_table, metadatos):
    """Guarda la tabla Q y sus metadatos en un archivo .npz."""
    carpeta = os.path.dirname(ruta)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
    np.savez_compressed(ruta, tabla=q_table.table, metadatos=json.dumps(metadatos))


def cargar_tabla_q(ruta):
    """Carga un archivo guardado con guardar_tabla_q. Devuelve (q_table, metadatos)."""
    with np.load(ruta) as datos:
        tabla = datos["tabla"]
        metadatos = json.loads(str(datos["metadatos"]))
    q_table = QTable(tabla.shape[0], tabla.shape[1])
    q_table.table = tabla
    return q_table, metadatos


if __name__ == "__main__":
    # Prueba: entrenar una tabla pequeña, guardarla, cargarla y comprobar que es idéntica
    import random
    from deustorl.common import EpsilonGreedyPolicy
    from deustorl.qlearning import QLearning
    from config import crear_entorno

    # Trabajar dentro de faseB/ para que tablas_q/ y logs/ se creen aquí
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    env = crear_entorno("bajo")
    seed = 3
    random.seed(seed)
    env.reset(seed=seed)

    algo = QLearning(env)
    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=0.1)
    algo.learn(epsilon_greedy_policy, n_steps=10_000, lr=0.1)

    metadatos = {"algoritmo": "qlearning", "presupuesto": "bajo", "semilla": seed}
    guardar_tabla_q("tablas_q/prueba.npz", algo.q_table, metadatos)

    q_cargada, metadatos_cargados = cargar_tabla_q("tablas_q/prueba.npz")
    print("Metadatos cargados:", metadatos_cargados)
    print("Tamaño de la tabla:", q_cargada.table.shape)
    print("¿Es idéntica a la original?", np.array_equal(algo.q_table.table, q_cargada.table))