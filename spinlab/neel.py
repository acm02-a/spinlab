"""Experimento 4: cuanto dura un bit magnetico contra el calor (Neel-Brown).

Sistema: 512 x 512 = 262 144 particulas identicas e independientes (sin
intercambio ni campo dipolar), cada una un cubo de 5 nm con anisotropia
uniaxial en z. Todas empiezan en +z; el calor las voltea al azar y la
magnetizacion media decae como exp(-t/tau).

Se mide tau a ocho temperaturas y se compara con la solucion exacta de la
ecuacion de Fokker-Planck de Brown (ver spinlab/teoria.py).

Uso:  python -m spinlab.neel [--simular]     (--simular: ~15 min de GPU)
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import teoria
from .comun import ajustar_tau, guardar, tabla

GAMMA = 1.7595e11  # rad/(T s), valor por defecto de MuMax3
MSAT = 8e5  # A/m
KU = 1e5  # J/m^3
LADO = 5e-9  # m
VOL = LADO**3
ALPHA = 0.1
N = 512  # particulas por lado
DT = 2e-13  # paso fijo del solver, s
SIGMAS = [3, 4, 5, 6, 7, 8, 9, 10]

PLANTILLA = """\
SetGridsize({n}, {n}, 1)
SetCellsize({lado}, {lado}, {lado})
EnableDemag = false
Msat  = {msat}
Aex   = 0
Ku1   = {ku}
AnisU = vector(0, 0, 1)
alpha = {alpha}
m = uniform(0, 0, 1)

// Cuatro cuadrantes = cuatro muestras independientes, para la barra de error.
DefRegion(1, xrange(-inf, 0).intersect(yrange(0, inf)))
DefRegion(2, xrange(0, inf).intersect(yrange(-inf, 0)))
DefRegion(3, xrange(0, inf).intersect(yrange(0, inf)))
TableAdd(m.Region(0))
TableAdd(m.Region(1))
TableAdd(m.Region(2))
TableAdd(m.Region(3))

Temp = {temp}
ThermSeed({semilla})
FixDt = {dt}
tableautosave(1e-11)
run({duracion})
"""


def temperatura(sigma: float) -> float:
    return KU * VOL / (teoria.KB * sigma)


def duracion(sigma: float) -> float:
    """Tiempo simulado: mas largo cuanto mas lento es el decaimiento."""
    return 6e-9 if sigma <= 5 else 10e-9 if sigma <= 7 else 12e-9


def correr(nombre, temp, dur, semilla, simular, dt=DT, ku=KU):
    """Devuelve (t, <mz> global, <mz> de cada cuadrante)."""
    script = PLANTILLA.format(n=N, lado=LADO, msat=MSAT, ku=ku, alpha=ALPHA,
                              temp=temp, semilla=semilla, dt=dt, duracion=dur)
    d = tabla("neel", nombre, script, simular)
    # columnas: t, mx, my, mz, y luego (mx, my, mz) por cada una de 4 regiones
    return d[:, 0], d[:, 3], d[:, [6, 9, 12, 15]]


def tau_con_error(t, cuadrantes, t_min=1e-9):
    taus = np.array([ajustar_tau(t, cuadrantes[:, k], t_min=t_min) for k in range(4)])
    return taus.mean(), taus.std(ddof=1) / 2  # error estandar de 4 muestras


def barrido(simular: bool = False):
    """Lista de (sigma, T, tau_sim, error, tau_exacto, tau_asintotico) y curvas."""
    filas, curvas = [], []
    for s in SIGMAS:
        t, mz, cuad = correr(f"sigma_{s:02d}", temperatura(s), duracion(s), 1000 + s, simular)
        tau, err = tau_con_error(t, cuad)
        filas.append((s, temperatura(s), tau, err,
                      teoria.tau_exacto(s, KU, MSAT, VOL, ALPHA, GAMMA),
                      teoria.tau_asintotico(s, KU, MSAT, ALPHA, GAMMA)))
        curvas.append((s, t, mz))
    return filas, curvas


def control_paso(simular: bool = False):
    """sigma = 5 con la mitad del paso de tiempo."""
    t, _, cuad = correr("control_dt_mitad", temperatura(5), duracion(5), 1005, simular, dt=DT / 2)
    return tau_con_error(t, cuad)


def control_sin_barrera(simular: bool = False):
    """K = 0 a 300 K: la teoria es una sola formula, tau_N."""
    t, _, cuad = correr("control_sin_barrera", 300, 3e-9, 1000, simular, ku=0)
    tau, err = tau_con_error(t, cuad, t_min=0)
    return tau, err, teoria.tau_libre(MSAT, VOL, ALPHA, GAMMA, 300)


def main() -> None:
    simular = "--simular" in sys.argv
    filas, curvas = barrido(simular)

    print(f"{'sigma':>5} {'T (K)':>7} {'tau sim (ns)':>13} {'+/-':>8} {'tau teoria':>11} {'sim/teoria':>10}")
    for s, temp, tau, err, exacto, _ in filas:
        print(f"{s:>5} {temp:>7.1f} {tau * 1e9:>13.3f} {err * 1e9:>8.3f} {exacto * 1e9:>11.3f} {tau / exacto:>10.3f}")

    tau, err = control_paso(simular)
    print(f"\nControl de paso (sigma=5): dt={DT:.0e} -> {filas[2][2] * 1e9:.3f} ns; "
          f"dt={DT / 2:.0e} -> {tau * 1e9:.3f} +/- {err * 1e9:.3f} ns")
    tau, err, libre = control_sin_barrera(simular)
    print(f"Control sin barrera (300 K): sim {tau * 1e12:.1f} +/- {err * 1e12:.1f} ps; teoria {libre * 1e12:.1f} ps")

    print("\nExtrapolacion con la teoria (no simulada):")
    for s in (40, 45, 50, 60):
        anios = teoria.tau_asintotico(s, KU, MSAT, ALPHA, GAMMA) / 3.156e7
        lado = (s * teoria.KB * 300 / KU) ** (1 / 3) * 1e9
        print(f"  sigma={s}: tau = {anios:.2e} anios; a 300 K exige un cubo de {lado:.1f} nm")

    f = np.array(filas)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))
    for (s, t, mz), c in zip(curvas, plt.cm.viridis(np.linspace(0, 0.9, len(curvas)))):
        a.semilogy(t * 1e9, np.clip(mz, 1e-3, None), color=c, label=rf"$\sigma$ = {s}")
    a.set_ylim(0.01, 1.2)
    a.set_xlabel("t (ns)")
    a.set_ylabel(r"$\langle m_z \rangle$  (fraccion neta que sigue en +z)")
    a.set_title("El calor borra la magnetizacion")
    a.legend(frameon=False, ncol=2, fontsize=8)

    ss = np.linspace(2.5, 10.5, 60)
    b.semilogy(ss, [teoria.tau_exacto(s, KU, MSAT, VOL, ALPHA, GAMMA) * 1e9 for s in ss],
               color="#222222", label="Teoria de Brown (Fokker-Planck, exacta)")
    b.semilogy(ss, teoria.tau_asintotico(ss, KU, MSAT, ALPHA, GAMMA) * 1e9, "--",
               color="#999999", label="Formula de barrera alta")
    b.errorbar(f[:, 0], f[:, 2] * 1e9, yerr=f[:, 3] * 1e9, fmt="o", color="#c44e52",
               capsize=3, label="Simulacion MuMax3")
    b.set_xlabel(r"$\sigma = KV / k_B T$   (barrera / energia termica)")
    b.set_ylabel(r"$\tau$ (ns)")
    b.set_title("Vida media del bit")
    b.legend(frameon=False)
    print("\nFigura:", guardar(fig, "neel_brown.png"))


if __name__ == "__main__":
    main()
