"""Experimento 1: precesion de Larmor de un solo espin.

Un momento magnetico perpendicular a un campo de 0.1 T, sin amortiguamiento.
Prediccion: gira alrededor del campo con periodo 2 pi / (gamma B) y nunca se alinea.

Uso:  python -m spinlab.larmor [--simular]
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .comun import guardar, tabla

GAMMA = 1.7595e11  # rad/(T s), valor por defecto de MuMax3
B = 0.1  # T

SCRIPT = """\
SetGridsize(1, 1, 1)
SetCellsize(5e-9, 5e-9, 5e-9)

Msat = 8e5
Aex  = 1.3e-11
EnableDemag = false
alpha = 0

B_ext = vector(0, 0, 0.1)
m = uniform(1, 0, 0)

// El problema es tan suave que el solver adaptativo da pasos de ~28 ps,
// mas largos que el intervalo de muestreo. MaxDt lo obliga a pasos cortos.
MaxDt = 1e-12

tablesave()
for i := 0; i < 200; i++ {
	run(5e-12)
	tablesave()
}
"""


def periodo_medido(t: np.ndarray, mx: np.ndarray) -> float:
    """Tiempo medio entre cruces por cero ascendentes de mx."""
    i = np.where((mx[:-1] < 0) & (mx[1:] >= 0))[0]
    cruces = t[i] - mx[i] * (t[i + 1] - t[i]) / (mx[i + 1] - mx[i])
    return float(np.diff(cruces).mean())


def periodo_teorico() -> float:
    return 2 * np.pi / (GAMMA * B)


def main() -> None:
    d = tabla("larmor", "larmor", SCRIPT, simular="--simular" in sys.argv)
    t, mx, my, mz = d[:, 0], d[:, 1], d[:, 2], d[:, 3]

    print(f"periodo medido  = {periodo_medido(t, mx) * 1e12:.2f} ps")
    print(f"periodo teorico = {periodo_teorico() * 1e12:.2f} ps")

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.7, 1]})
    a.plot(t * 1e12, mx, color="#c44e52", label=r"$m_x$")
    a.plot(t * 1e12, my, color="#4c72b0", label=r"$m_y$")
    a.plot(t * 1e12, mz, color="#222222", label=r"$m_z$ (direccion del campo)")
    a.set_xlabel("t (ps)")
    a.set_ylabel("componente de la magnetizacion")
    a.set_title("Cada componente en el tiempo")
    a.legend(frameon=False, loc="lower left")
    b.plot(mx, my, color="#55a868")
    b.set_aspect("equal")
    b.set_xlabel(r"$m_x$")
    b.set_ylabel(r"$m_y$")
    b.set_title("Visto desde arriba")
    print("Figura:", guardar(fig, "larmor.png"))


if __name__ == "__main__":
    main()
