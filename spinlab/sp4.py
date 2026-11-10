"""Experimento 2: muMAG Standard Problem #4 (campo 1) y convergencia de malla.

Lamina de permalloy de 500 x 125 x 3 nm. Se relaja a un estado S y se aplica
un campo que invierte la magnetizacion. Se repite con tres tamanos de celda.

Referencia: https://www.ctcms.nist.gov/~rdm/std4/spec4.html

Uso:  python -m spinlab.sp4 [--simular]
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .comun import cruce_por_cero, guardar, tabla

MALLAS = [(64, 16), (128, 32), (256, 64)]

# Valor de <m> a t = 1 ns en el test oficial de MuMax3 (test/standardproblem4.mx3),
# malla 128 x 32.
REFERENCIA_MUMAX = (-0.9846124053001404, 0.12604089081287384, 0.04327124357223511)

PLANTILLA = """\
SetGridsize({nx}, {ny}, 1)
SetCellsize(500e-9/{nx}, 125e-9/{ny}, 3e-9)
Msat  = 800e3
Aex   = 13e-12
alpha = 0.02
m = uniform(1, .1, 0)
relax()
tableautosave(5e-12)
B_ext = vector(-24.6E-3, 4.3E-3, 0)
run(1e-9)
"""


def correr(simular: bool = False) -> list[dict]:
    resultados = []
    for nx, ny in MALLAS:
        d = tabla("sp4", f"sp4_{nx}x{ny}", PLANTILLA.format(nx=nx, ny=ny), simular)
        t, mx, my = d[:, 0], d[:, 1], d[:, 2]
        resultados.append(
            {"nx": nx, "ny": ny, "celda": 500 / nx, "t": t, "mx": mx, "my": my,
             "mz": d[:, 3], "t0": cruce_por_cero(t, mx)}
        )
    return resultados


def main() -> None:
    resultados = correr("--simular" in sys.argv)

    print(f"{'malla':>9} {'celda nm':>9} {'cruce mx ps':>12} {'mx(1ns)':>9} {'my(1ns)':>9}")
    for r in resultados:
        print(f"{r['nx']:>5}x{r['ny']:<3} {r['celda']:>9.2f} {r['t0'] * 1e12:>12.2f} "
              f"{r['mx'][-1]:>9.5f} {r['my'][-1]:>9.5f}")

    fig, ejes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    for r, c, ls in zip(resultados, ["#c44e52", "#4c72b0", "#222222"], [":", "--", "-"]):
        etiqueta = f"{r['nx']}x{r['ny']}  (celda {r['celda']:.1f} nm)"
        ejes[0].plot(r["t"] * 1e9, r["mx"], ls, color=c, label=etiqueta)
        ejes[1].plot(r["t"] * 1e9, r["my"], ls, color=c)
    for eje, nombre in zip(ejes, [r"$\langle m_x \rangle$", r"$\langle m_y \rangle$"]):
        eje.axhline(0, color="gray", lw=0.5)
        eje.set_ylabel(nombre)
    ejes[1].set_xlabel("t (ns)")
    ejes[0].legend(frameon=False)
    ejes[0].set_title("muMAG Standard Problem #4, campo 1: convergencia de malla")
    print("Figura:", guardar(fig, "sp4_convergencia.png"))


if __name__ == "__main__":
    main()
