import os 
import sys
import glob
import numpy as np 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ACTIVIDAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path[0:0] = [ACTIVIDAD, os.path.dirname(ACTIVIDAD)] 

from config import PRESUPUESTOS, ALGORITMOS

def leer_curva(carpeta_semilla):
    pasos = []
    recompensas = []
    for carpeta_log in glob.glob(f"{carpeta_semilla}/logs/*"):
        eventos = EventAccumulator(carpeta_log, size_guidance={"scalars": 0})
        eventos.Reload()
        for e in eventos.Scalars("train/reward"):
            pasos.append(e.step)
            recompensas.append(e.value)
    return np.array(pasos), np.array(recompensas)


if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("resultados", exist_ok=True)

    rejilla = np.linspace(0, 2_000_000, 200)

    curvas = {}
    for presupuesto in PRESUPUESTOS:
        for algo_name in ALGORITMOS:
            semillas = []
            for carpeta_semilla in sorted(glob.glob(f"logs/{presupuesto}/{algo_name}/semilla*")):
                pasos, recompensas = leer_curva(carpeta_semilla)
                if len(pasos) > 0:
                    semillas.append(np.interp(rejilla, pasos, recompensas))
            if len(semillas) > 0:
                curvas[(presupuesto, algo_name)] = (np.mean(semillas, axis=0), np.std(semillas, axis=0))
                print(f"{algo_name} {presupuesto}: {len(semillas)} semillas, recompensa final = {np.mean(semillas, axis=0) [-1]:.2f}")

    fig, paneles = plt.subplots(1, len(PRESUPUESTOS), figsize=(15, 4), sharey=True)
    for panel, presupuesto in zip(paneles, PRESUPUESTOS):
        for algo_name in ALGORITMOS:
            if (presupuesto, algo_name) in curvas:
                media, desviacion = curvas[(presupuesto, algo_name)]
                panel.plot(rejilla, media, label=algo_name)
                panel.fill_between(rejilla, media - desviacion, media + desviacion, alpha=0.2)

        panel.set_title(f"{presupuesto} ({PRESUPUESTOS[presupuesto]} bins)")
        panel.set_xlabel("Pasos de entrenamiento")
        panel.grid(alpha=0.3)
    paneles[0].set_ylabel("Recompensa media por episodio")
    paneles[0].legend()
    fig.suptitle("Curvas de aprendizaje por presupuesto (media ± desviación de las semillas)")
    fig.tight_layout()
    fig.savefig("resultados/curvas_aprendizaje.png", dpi=150)

    print("\nGráfica guardada en faseB/resultados/curvas_aprendizaje.png")

        