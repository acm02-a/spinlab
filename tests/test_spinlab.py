"""Tests: la teoria contra sus limites conocidos, los helpers contra senales
sinteticas, y los resultados publicados contra los datos guardados en datos/.

Ninguno necesita GPU ni MuMax3: las tablas de salida estan versionadas.
"""
import numpy as np
import pytest

from spinlab import larmor, neel, sp4, teoria
from spinlab.comun import ajustar_tau, cruce_por_cero, espectro, picos


# ------------------------------------------------------------------ teoria
def test_autovalor_sin_barrera_es_2():
    assert teoria.autovalor_brown(1e-9) == pytest.approx(2.0, rel=1e-6)


def test_autovalor_tiende_a_la_formula_de_barrera_alta():
    cocientes = []
    for sigma in (10, 15, 20):
        asintotico = 4 / np.sqrt(np.pi) * sigma**1.5 * np.exp(-sigma)
        cocientes.append(teoria.autovalor_brown(sigma) / asintotico)
    # La correccion conocida es ~ (1 - 1/sigma): se acerca a 1 desde abajo.
    assert cocientes[0] < cocientes[1] < cocientes[2] < 1
    assert cocientes[2] == pytest.approx(1 - 1 / 20, abs=0.01)


def test_autovalor_no_depende_del_tamano_de_la_base():
    assert teoria.autovalor_brown(8, 60, 600) == pytest.approx(teoria.autovalor_brown(8, 80, 800), rel=1e-6)


# ----------------------------------------------------------------- helpers
def test_cruce_por_cero():
    t = np.linspace(0, 1, 1001)
    assert cruce_por_cero(t, np.cos(2 * np.pi * t)) == pytest.approx(0.25, abs=1e-5)


def test_ajustar_tau_recupera_una_exponencial():
    t = np.linspace(0, 10e-9, 500)
    assert ajustar_tau(t, 0.9 * np.exp(-t / 3e-9)) == pytest.approx(3e-9, rel=1e-6)


def test_espectro_encuentra_el_pico():
    dt = 5e-12
    t = np.arange(4000) * dt
    f, s = espectro(np.sin(2 * np.pi * 8.25e9 * t), dt)
    assert picos(f, s, n=1)[0] == pytest.approx(8.25, abs=0.05)


# ------------------------------------------------- resultados (datos guardados)
def test_larmor_periodo_coincide_con_la_teoria():
    d = larmor.tabla("larmor", "larmor", larmor.SCRIPT)
    assert larmor.periodo_medido(d[:, 0], d[:, 1]) == pytest.approx(larmor.periodo_teorico(), rel=1e-3)


def test_sp4_coincide_con_la_referencia_de_mumax():
    r = sp4.correr()[1]  # malla 128 x 32
    final = (r["mx"][-1], r["my"][-1], r["mz"][-1])
    assert final == pytest.approx(sp4.REFERENCIA_MUMAX, abs=1e-5)


def test_sp4_converge_con_orden_2():
    t0 = [r["t0"] for r in sp4.correr()]
    orden = np.log2((t0[0] - t0[1]) / (t0[1] - t0[2]))
    assert orden == pytest.approx(2, abs=0.3)


def test_neel_coincide_con_brown_en_todo_el_barrido():
    filas, _ = neel.barrido()
    for sigma, _, tau, _, exacto, _ in filas:
        assert tau == pytest.approx(exacto, rel=0.03), f"sigma = {sigma}"


def test_neel_control_sin_barrera():
    tau, err, libre = neel.control_sin_barrera()
    assert abs(tau - libre) < 3 * err


def test_neel_no_depende_del_paso_de_tiempo():
    filas, _ = neel.barrido()
    tau_mitad, _ = neel.control_paso()
    assert tau_mitad == pytest.approx(filas[2][2], rel=0.02)
