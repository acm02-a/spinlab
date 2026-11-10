"""Experimento 3: problema estandar de resonancia ferromagnetica (FMR).

Replica de Baker et al., J. Magn. Magn. Mater. 421, 428 (2017), figuras 2 y 3.
Lamina de permalloy de 120 x 120 x 10 nm. Se relaja con el campo en una
direccion, se gira el campo ligeramente y se registra como "suena" la
magnetizacion. El espectro de potencia de <my>(t) da los modos de resonancia.

La comparacion es contra los datos de referencia de OOMMF publicados por los
autores, que se descargan la primera vez desde su repositorio:
github.com/fangohr/micromagnetic-standard-problem-ferromagnetic-resonance

Uso:  python -m spinlab.fmr [--simular]
"""
from __future__ import annotations

import sys
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .comun import DATOS, espectro, guardar, picos, tabla

N = 4000  # muestras
DT = 5e-12  # s

URL_REFERENCIA = (
    "https://raw.githubusercontent.com/fangohr/"
    "micromagnetic-standard-problem-ferromagnetic-resonance/HEAD/"
    "micromagnetic_simulation_data/reference_data/oommf/dynamic_txyz.txt"
)

SCRIPT = """\
// Parametros tomados de los scripts OOMMF del repositorio de los autores.
SetGridsize(24, 24, 2)
SetCellsize(5e-9, 5e-9, 5e-9)

Msat    = 8e5
Aex     = 1.3e-11
GammaLL = 2.210173e5 / mu0    // el paper da gamma en m/(A s)

H := 8e4                      // A/m

// Etapa 1: relajacion con el campo a lo largo de (1, 0.715, 0), alpha = 1.
alpha = 1.0
m     = uniform(0, 0, 1)
B_ext = vector(mu0*H*0.81345856316858023, mu0*H*0.58162287266553481, 0)
run(5e-9)

// Etapa 2: el campo gira a (1, 0.7, 0). Ese cambio brusco es la excitacion.
t     = 0
alpha = 0.008
B_ext = vector(mu0*H*0.81923192051904048, mu0*H*0.57346234436332832, 0)

// tableautosave() no guarda en instantes exactos; la FFT necesita muestreo
// uniforme, asi que se avanza de a 5 ps y se guarda a mano.
tablesave()
for i := 0; i < 4000; i++ {
	run(5e-12)
	tablesave()
}
"""


def referencia() -> np.ndarray:
    """<my>(t) de OOMMF publicado por los autores (se descarga si falta)."""
    ruta = DATOS / "fmr" / "referencia_oommf.txt"
    if not ruta.exists():
        ruta.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL_REFERENCIA, ruta)
    return np.loadtxt(ruta)[:, 2]


def simulacion(simular: bool = False) -> tuple[np.ndarray, np.ndarray]:
    d = tabla("fmr", "fmr", SCRIPT, simular)
    # MuMax3 guarda tambien t = 0; el paper empieza en t = 5 ps.
    return d[1 : N + 1, 0], d[1 : N + 1, 2]


def main() -> None:
    t, my = simulacion("--simular" in sys.argv)
    my_ref = referencia()
    f, s = espectro(my, DT)
    _, s_ref = espectro(my_ref, DT)

    print(f"max |my - my_ref| = {np.abs(my - my_ref).max():.2e}")
    print("picos MuMax3 (GHz):", [round(x, 2) for x in picos(f, s)])
    print("picos OOMMF  (GHz):", [round(x, 2) for x in picos(f, s_ref)])

    fig, ejes = plt.subplots(2, 1, figsize=(8, 6.8))
    ejes[0].plot(t * 1e9, my_ref, color="#c44e52", lw=2.5, alpha=0.6, label="OOMMF (referencia del paper)")
    ejes[0].plot(t * 1e9, my, color="#222222", lw=0.8, label="MuMax3 (esta replica)")
    ejes[0].set_xlim(0, 2.5)
    ejes[0].set_xlabel("t (ns)")
    ejes[0].set_ylabel(r"$\langle m_y \rangle$")
    ejes[0].set_title("Baker et al. (2017), problema estandar FMR")
    ejes[0].legend(frameon=False)
    ejes[1].semilogy(f, s_ref, color="#c44e52", lw=2.5, alpha=0.6, label="OOMMF")
    ejes[1].semilogy(f, s, color="#222222", lw=0.8, label="MuMax3")
    ejes[1].set_xlim(0, 20)
    ejes[1].set_xlabel("f (GHz)")
    ejes[1].set_ylabel(r"$S_y(f)$ (u. arb.)")
    ejes[1].legend(frameon=False)
    print("Figura:", guardar(fig, "fmr_baker2017.png"))


if __name__ == "__main__":
    main()
