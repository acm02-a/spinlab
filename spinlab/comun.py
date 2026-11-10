"""Utilidades compartidas: rutas, ejecucion de MuMax3 y helpers numericos."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "datos"
FIGURAS = RAIZ / "figuras"

# Ejecutable de MuMax3. Se puede fijar con la variable de entorno MUMAX3.
MUMAX = os.environ.get("MUMAX3", "mumax3")


def tabla(experimento: str, nombre: str, script: str, simular: bool = False) -> np.ndarray:
    """Devuelve la tabla de salida de MuMax3 para un script dado.

    Si `datos/<experimento>/<nombre>.txt` ya existe se reutiliza, de modo que
    las figuras se pueden regenerar sin GPU. Con `simular=True` se vuelve a
    correr MuMax3 y se reemplaza la tabla.
    """
    destino = DATOS / experimento / f"{nombre}.txt"
    mx3 = destino.with_suffix(".mx3")
    destino.parent.mkdir(parents=True, exist_ok=True)
    mx3.write_text(script, encoding="utf-8")

    if simular or not destino.exists():
        try:
            subprocess.run([MUMAX, str(mx3)], check=True, stdout=subprocess.DEVNULL)
        except FileNotFoundError:
            sys.exit(
                f"No se encontro MuMax3 ('{MUMAX}'). Instalalo o define la "
                "variable de entorno MUMAX3 con la ruta al ejecutable."
            )
        salida = mx3.with_suffix(".out")
        shutil.copy(salida / "table.txt", destino)
        shutil.rmtree(salida)
    return np.loadtxt(destino)


def cruce_por_cero(t: np.ndarray, y: np.ndarray) -> float:
    """Instante del primer cambio de signo de y, por interpolacion lineal."""
    i = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0][0]
    return float(t[i] - y[i] * (t[i + 1] - t[i]) / (y[i + 1] - y[i]))


def espectro(y: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Frecuencias (GHz) y densidad espectral de potencia |FFT(y)|^2."""
    f = np.fft.rfftfreq(len(y), dt) / 1e9
    return f, np.abs(np.fft.rfft(y)) ** 2


def picos(f: np.ndarray, s: np.ndarray, n: int = 5, fmin: float = 1.0) -> list[float]:
    """Los n maximos locales mas altos del espectro por encima de fmin GHz."""
    i = np.where((s[1:-1] > s[:-2]) & (s[1:-1] > s[2:]) & (f[1:-1] > fmin))[0] + 1
    i = i[np.argsort(s[i])[::-1][:n]]
    return sorted(float(x) for x in f[i])


def ajustar_tau(t: np.ndarray, y: np.ndarray, t_min: float = 1e-9, y_min: float = 0.05) -> float:
    """Tiempo de decaimiento de y ~ exp(-t/tau), por ajuste lineal de ln(y).

    Se descarta el transitorio inicial (t < t_min) y todo lo que venga despues
    de que la senal caiga por debajo de y_min, donde manda el ruido.
    """
    ok = t >= t_min
    bajos = ok & ~(y > y_min)
    if bajos.any():
        ok &= t < t[bajos][0]
    pendiente = np.polyfit(t[ok], np.log(y[ok]), 1)[0]
    return float(-1.0 / pendiente)


def guardar(fig, nombre: str) -> Path:
    FIGURAS.mkdir(exist_ok=True)
    ruta = FIGURAS / nombre
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    return ruta
