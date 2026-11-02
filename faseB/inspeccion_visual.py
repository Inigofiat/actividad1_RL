"""Fase B Tarea 4: Reproduce en pantalla las tres políticas finales de la Fase B."""
import argparse
import os
import sys

import numpy as np

ACTIVIDAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[0:0] = [ACTIVIDAD, os.path.dirname(ACTIVIDAD)]

from config import crear_entorno
from faseB.tabla_q import cargar_tabla_q


def entero_positivo(valor):
    try:
        numero = int(valor)
    except ValueError as error:
        raise argparse.ArgumentTypeError("debe ser un entero") from error
    if numero <= 0:
        raise argparse.ArgumentTypeError("debe ser mayor que cero")
    return numero


def inspeccionar_politica(presupuesto, ruta, episodios, semilla):
    q_table, metadatos = cargar_tabla_q(ruta)
    presupuesto_guardado = metadatos.get("presupuesto")
    if presupuesto_guardado != presupuesto:
        raise ValueError(
            f"{ruta}: se esperaba presupuesto {presupuesto!r}, "
            f"pero los metadatos indican {presupuesto_guardado!r}"
        )

    env = crear_entorno(presupuesto, render_mode="human")
    try:
        forma_esperada = (env.observation_space.n, env.action_space.n)
        if q_table.table.shape != forma_esperada:
            raise ValueError(
                f"{ruta}: la tabla tiene forma {q_table.table.shape}; "
                f"se esperaba {forma_esperada}"
            )
        if not np.isfinite(q_table.table).all():
            raise ValueError(f"{ruta}: la tabla contiene valores no finitos")

        algoritmo = metadatos.get("algoritmo", "desconocido")
        print(
            f"\nPolítica {presupuesto} | algoritmo: {algoritmo} | "
            f"tabla: {q_table.table.shape} | archivo: {ruta}"
        )
        resultados = []

        for numero_episodio in range(episodios):
            observacion, _ = env.reset(seed=semilla + numero_episodio)
            env.render()
            recompensa_total = 0.0
            terminacion = "límite de pasos"
            pasos = 0

            while True:
                accion = int(np.argmax(q_table[observacion]))
                observacion, recompensa, terminado, truncado, _ = env.step(accion)
                env.render()
                recompensa_total += float(recompensa)
                pasos += 1

                if terminado or truncado:
                    terminacion = "objetivo alcanzado" if terminado else "truncado"
                    break

            resultados.append((recompensa_total, pasos, terminado))
            print(
                f"  Episodio {numero_episodio + 1}/{episodios}: "
                f"recompensa={recompensa_total:.1f}, pasos={pasos}, "
                f"resultado={terminacion}"
            )

        recompensas = [resultado[0] for resultado in resultados]
        pasos_medios = np.mean([resultado[1] for resultado in resultados])
        tasa_exito = np.mean([resultado[2] for resultado in resultados])
        print(
            f"  Media: recompensa={np.mean(recompensas):.2f} "
            f"(desviación={np.std(recompensas):.2f}), "
            f"pasos={pasos_medios:.1f}, éxito={tasa_exito:.0%}"
        )
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspecciona visualmente las políticas finales bajo, medio y alto. "
            "Cada archivo debe incluir metadatos con su presupuesto."
        )
    )
    for presupuesto in ("bajo", "medio", "alto"):
        parser.add_argument(
            f"--{presupuesto}",
            required=True,
            metavar="TABLA.npz",
            help=f"tabla Q final del presupuesto {presupuesto}",
        )
    parser.add_argument(
        "--episodios",
        type=entero_positivo,
        default=3,
        help="episodios renderizados por política (por defecto: 3)",
    )
    parser.add_argument(
        "--semilla",
        type=int,
        default=47,
        help="semilla inicial para hacer reproducibles los episodios (por defecto: 47)",
    )
    args = parser.parse_args()
    if args.semilla < 0:
        parser.error("--semilla debe ser mayor o igual que cero")

    for presupuesto in ("bajo", "medio", "alto"):
        inspeccionar_politica(
            presupuesto,
            getattr(args, presupuesto),
            args.episodios,
            args.semilla,
        )


if __name__ == "__main__":
    main()