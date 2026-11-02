# Seleccionar la mejor combinación de hiperparámetros para cada algoritmo 
# y presupuesto de discretización, obteniendo un total de nueve configuraciones.

import os
import sys
import glob
import json 
import csv 
import optuna

ACTIVIDAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[0:0] = [ACTIVIDAD, os.path.dirname(ACTIVIDAD)]  

from config import PRESUPUESTOS, ALGORITMOS

if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    storage_file = f"sqlite:///optuna/optuna.db"

    estudios_existentes = []
    for s in optuna.get_all_study_summaries(storage=storage_file):
        estudios_existentes.append(s.study_name)

    for archivo in glob.glob("optuna/optuna_*.db"):
        storage_companero = "sqlite:///" + archivo.replace("\\", "/")

        for resumen in optuna.get_all_study_summaries(storage=storage_companero):
            study_name = resumen.study_name

            if study_name in estudios_existentes:
                continue

            optuna.copy_study(from_study_name=study_name, from_storage=storage_companero, to_storage=storage_file)
            estudios_existentes.append(study_name)
            print(f"Copiado {study_name} desde {archivo}")

    mejores_configuraciones = {}   #dicc vacío 
    filas_tabla = []    #lista vacía

    for presupuesto in PRESUPUESTOS:
        for algo_name in ALGORITMOS:
            study_name = f"{algo_name}_{presupuesto}"
            study = optuna.load_study(study_name=study_name, storage=storage_file)
            best_trial = study.best_trial

            mejores_configuraciones[study_name] = best_trial.params
            filas_tabla.append({"algoritmo": algo_name, "presupuesto": presupuesto, "n_bins": PRESUPUESTOS[presupuesto], "n_trials": len(study.trials), "mejor_recompensa": round(best_trial.value, 2), **best_trial.params})

            print(f"{study_name}: recompensa = {best_trial.value:.2f} con {best_trial.params}")

    os.makedirs("resultados", exist_ok=True)

    best_params = json.dumps(mejores_configuraciones, sort_keys=True, indent=4) #json.dumps convierte el diccionario en texto json, sort_keys ordena las claves y indent=4 da formato legible
    json_file = open("resultados/mejores_configuraciones.json", "w")
    json_file.write(best_params)
    json_file.close()

    csv_file = open("resultados/mejores_configuraciones.csv", "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=filas_tabla[0].keys())
    writer.writeheader()
    writer.writerows(filas_tabla)
    csv_file.close()

    print("Guardado en faseA/resultados/mejores_configuraciones.json y .csv")