# Evaluar cada una 50 episodios y seleccionar la mejor por cada tamaño de discretización (total: 3).

import os 
import sys
import glob
import random
import shutil

ACTIVIDAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[0:0] = [ACTIVIDAD, os.path.dirname(ACTIVIDAD)]

from deustorl.common import *
from config import crear_entorno, PRESUPUESTOS
from tabla_q import cargar_tabla_q

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    resultados = []
    
    for ruta in sorted(glob.glob("tablas_q/*_semilla*.npz")):
        q_table, metadatos = cargar_tabla_q(ruta)
        print(f"\n{ruta}")
        
        env = crear_entorno(metadatos["presupuesto"])
        seed = 3
        random.seed(seed)
        env.reset(seed=seed)
        avg_reward, avg_steps = evaluate_policy(env, q_table, max_policy, n_episodes=50)
        
        resultados.append((avg_reward, metadatos["presupuesto"], ruta))
    
    os.makedirs("politicas_finales", exist_ok=True)
    print("\nPoliticas finales (la mejor de cada presupuesto):")
    for presupuesto in PRESUPUESTOS.keys():
        candidatas = [r for r in resultados if r[1] == presupuesto]
        if len(candidatas) == 0:
            continue
        mejor_reward, _, mejor_ruta = max(candidatas)
        shutil.copy(mejor_ruta, f"politicas_finales/mejor_{presupuesto}.npz")
        print(f"  {presupuesto:6s} -> {mejor_ruta}   recompensa media = {mejor_reward:.2f}")