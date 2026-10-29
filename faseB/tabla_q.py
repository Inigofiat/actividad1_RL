"""Funciones para guardar y cargar las tablas Q de deustorl."""

from pathlib import Path

import numpy as np

from deustorl.common import QTable


def guardar_tabla_q(tabla_q: QTable, ruta: str | Path) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    with ruta.open("wb") as archivo:
        np.save(archivo, tabla_q.table, allow_pickle=False)


def cargar_tabla_q(ruta: str | Path) -> QTable:
    with Path(ruta).open("rb") as archivo:
        valores = np.load(archivo, allow_pickle=False)

    if valores.ndim != 2:
        raise ValueError("El archivo debe contener una tabla Q de dos dimensiones.")

    n_estados, n_acciones = valores.shape
    tabla_q = QTable(n_estados, n_acciones)
    tabla_q.table = valores
    return tabla_q
