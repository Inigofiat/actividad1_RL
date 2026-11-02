import os
import sys
import random
import json
import time
import gc

ACTIVIDAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[0:0] = [ACTIVIDAD, os.path.dirname(ACTIVIDAD)]

from deustorl.common import *
from deustorl.sarsa import Sarsa
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.qlearning import QLearning
from config import crear_entorno, ALGORITMOS, PRESUPUESTOS

from tabla_q import guardar_tabla_q

from tensorboard.summary.writer.event_file_writer import EventFileWriter

MIS_PRESUPUESTOS = [arg for arg in sys.argv[1:] if arg in PRESUPUESTOS] 
MIS_SEMILLAS = [int(arg) for arg in sys.argv[1:] if arg.isdigit()]
if not MIS_PRESUPUESTOS or not MIS_SEMILLAS:
    sys.exit("Uso: python faseB/entrenar.py <presupuestos> <semillas>  (ej.: python faseB/entrenar.py bajo 1 2)")
    
n_steps = 2_000_000

if __name__ == "__main__":
    os.chdir(os.path.join(ACTIVIDAD, "faseB"))

    json_file = open(os.path.join(ACTIVIDAD, "faseA", "resultados", "mejores_configuraciones.json"))
    mejores_configuraciones = json.load(json_file)
    json_file.close()

    for presupuesto in MIS_PRESUPUESTOS:
        for algo_name in ALGORITMOS:
            for seed in MIS_SEMILLAS:
                params = mejores_configuraciones[f"{algo_name}_{presupuesto}"]
                print(f"\nEntrenando {algo_name} con presupuesto {presupuesto} y semilla {seed}...")
                env = crear_entorno(presupuesto)
                random.seed(seed)
                env.reset(seed=seed)

                if algo_name == "sarsa":
                    algo = Sarsa(env)
                elif algo_name == "esarsa":
                    algo = ExpectedSarsa(env)
                elif algo_name == "qlearning":
                    algo = QLearning(env)

                carpeta_logs = f"logs/{presupuesto}/{algo_name}/semilla_{seed}"
                os.makedirs(carpeta_logs, exist_ok=True)
                os.chdir(carpeta_logs)
                
                epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=params["epsilon"])
                start_time = time.time()
                algo.learn(epsilon_greedy_policy, n_steps=n_steps, discount_rate=params["discount_rate"], lr=params["learning_rate"], lrdecay=params["learning_rate_decay"], n_episodes_decay=params["lr_episodes_decay"])
                elapsed_time = time.time() - start_time
                print("----- {:0.4f} secs. -----".format(elapsed_time))
                for objeto in gc.get_objects():
                    if type(objeto) is EventFileWriter:
                        objeto.close()
                
                os.chdir(os.path.join(ACTIVIDAD, "faseB"))
                env.close()
                
                metadatos = {
                    "algoritmo": algo_name,
                    "presupuesto": presupuesto,
                    "n_bins": PRESUPUESTOS[presupuesto],
                    "semilla": seed,
                    "n_steps": n_steps,
                    "hiperparámetros": params
                    }
                
                ruta = f"tablas_q/{algo_name}_{presupuesto}_semilla_{seed}.npz"
                guardar_tabla_q(ruta, algo.q_table, metadatos)
                print(f"Tabla Q guardada en faseB/{ruta} con metadatos: {metadatos}")
                