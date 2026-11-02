import os 
import sys
import glob
import csv
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
        if "train/reward" not in eventos.Tags()["scalars"]:
            print(f"  (sin datos, se ignoran: {carpeta_log}")
            continue
        for e in eventos.Scalars("train/reward"):
            pasos.append(e.step)
            recompensas.append(e.value)
    return np.array(pasos), np.array(recompensas)


if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("resultados", exist_ok=True)

    rejilla = np.linspace(0, 2_000_000, 200)

    curvas = {}
    finales = {}
    for presupuesto in PRESUPUESTOS:
        for algo_name in ALGORITMOS:
            semillas = []
            for carpeta_semilla in sorted(glob.glob(f"logs/{presupuesto}/{algo_name}/semilla*")):
                pasos, recompensas = leer_curva(carpeta_semilla)
                if len(pasos) > 0:
                    semillas.append(np.interp(rejilla, pasos, recompensas))
            if len(semillas) > 0:
                curvas[(presupuesto, algo_name)] = (np.mean(semillas, axis=0), np.std(semillas, axis=0))
                finales[(presupuesto, algo_name)] = [s[-1] for s in semillas]
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
    plt.close(fig)

    print("\nGráfica guardada en faseB/resultados/curvas_aprendizaje.png")


    for presupuesto in PRESUPUESTOS:
        fig, panel = plt.subplots(figsize=(8, 5))
        for algo_name in ALGORITMOS:
            if (presupuesto, algo_name) in curvas:
                media, desviacion = curvas[(presupuesto, algo_name)]
                panel.plot(rejilla, media, label=algo_name)
                panel.fill_between(rejilla, media - desviacion, media + desviacion, alpha=0.2)
        panel.set_title(f"Curvas de aprendizaje – {presupuesto} ({PRESUPUESTOS[presupuesto]} bins)")
        panel.set_xlabel("Pasos de entrenamiento")
        panel.set_ylabel("Recompensa media por episodio")
        panel.grid(alpha=0.3)
        panel.legend()
        fig.tight_layout()
        fig.savefig(f"resultados/curvas_{presupuesto}.png", dpi=150)
        plt.close(fig)
        
        print(f"Gráfica guardada en faseB/resultados/curvas_{presupuesto}.png")

 
    fig, paneles = plt.subplots(1, len(ALGORITMOS), figsize=(15, 4), sharey=True)
    for panel, algo_name in zip(paneles, ALGORITMOS):
        for presupuesto in PRESUPUESTOS:
            if (presupuesto, algo_name) in curvas:
                media, desviacion = curvas[(presupuesto, algo_name)]
                panel.plot(rejilla, media, label=f"{presupuesto} ({PRESUPUESTOS[presupuesto]} bins)")
                panel.fill_between(rejilla, media - desviacion, media + desviacion, alpha=0.2)
        panel.set_title(algo_name)
        panel.set_xlabel("Pasos de entrenamiento")
        panel.grid(alpha=0.3)
    paneles[0].set_ylabel("Recompensa media por episodio")
    paneles[0].legend()
    fig.suptitle("Efecto de la discretización en cada algoritmo (media ± desviación de las semillas)")
    fig.tight_layout()
    fig.savefig("resultados/curvas_por_algoritmo.png", dpi=150)
    plt.close(fig)
    
    print("Gráfica guardada en faseB/resultados/curvas_por_algoritmo.png")
 

    csv_file = open("resultados/resumen_curvas.csv", "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["algoritmo", "presupuesto", "final_semillas", "final_media", "final_desviacion", "pasos_hasta_-400"])
    
    print("\nResumen (recompensa al final del entrenamiento = media de los últimos 100 episodios):")
    
    for (presupuesto, algo_name), media_desv in curvas.items():
        media = media_desv[0]
        valores = finales[(presupuesto, algo_name)]
        por_encima = np.where(media >= -400)[0]
        pasos_400 = int(rejilla[por_encima[0]]) if len(por_encima) > 0 else "no alcanza"
        writer.writerow([algo_name, presupuesto, " / ".join(f"{v:.1f}" for v in valores),
                         round(np.mean(valores), 1), round(np.std(valores), 1), pasos_400])
       
        print(f"  {algo_name:9s} {presupuesto:5s}  final = {np.mean(valores):7.1f} ± {np.std(valores):5.1f}   pasos hasta -400 = {pasos_400}")
   
    csv_file.close()

    print("Tabla guardada en faseB/resultados/resumen_curvas.csv")

        