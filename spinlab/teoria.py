"""Teoria de Neel-Brown para una particula uniaxial de dominio unico.

Notacion:
    sigma = K V / (kB T)        barrera de energia / energia termica
    tau_N = Ms V (1 + alpha^2) / (2 alpha gamma kB T)   tiempo de difusion libre
    B_K   = 2 K / Ms            campo de anisotropia

La densidad de probabilidad W(x, t), con x = cos(theta), obedece la ecuacion
de Fokker-Planck de Brown (1963):

    2 tau_N dW/dt = d/dx [ (1 - x^2) (dW/dx - 2 sigma x W) ]

El tiempo de relajacion de <mz> es tau = 2 tau_N / Lambda_1, donde Lambda_1 es
el menor autovalor no nulo del operador de la derecha.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import legendre as L

KB = 1.380649e-23  # J/K


def autovalor_brown(sigma: float, nbase: int = 60, ncuad: int = 600) -> float:
    """Menor autovalor no nulo Lambda_1(sigma) del operador de Fokker-Planck.

    Con W = exp(sigma x^2) g(x) el problema queda simetrico:
        A g = Lambda B g
        A_mn = int (1 - x^2) e^{sigma x^2} g_m' g_n' dx
        B_mn = int e^{sigma x^2} g_m g_n dx
    y se resuelve en base de polinomios de Legendre con cuadratura de Gauss.

    Limites conocidos: Lambda_1(0) = 2, y para sigma grande
    Lambda_1 -> (4 / sqrt(pi)) sigma^(3/2) exp(-sigma).
    """
    x, w = L.leggauss(ncuad)
    peso = np.exp(sigma * (x**2 - 1.0))  # el factor e^{-sigma} se cancela
    identidad = np.eye(nbase)
    P = np.array([L.legval(x, identidad[k]) for k in range(nbase)])
    dP = np.array([L.legval(x, L.legder(identidad[k])) for k in range(nbase)])
    A = (dP * (w * peso * (1 - x**2))) @ dP.T
    B = (P * (w * peso)) @ P.T
    vals = np.sort(np.linalg.eigvals(np.linalg.solve(B, A)).real)
    return float(vals[1])  # vals[0] = 0 es el estado de equilibrio


def tau_libre(msat: float, vol: float, alpha: float, gamma: float, temp: float) -> float:
    """tau_N: relajacion de <mz> sin barrera (K = 0)."""
    return msat * vol * (1 + alpha**2) / (2 * alpha * gamma * KB * temp)


def tau_exacto(sigma: float, ku: float, msat: float, vol: float, alpha: float, gamma: float) -> float:
    """Tiempo de relajacion de <mz> segun la solucion exacta de Fokker-Planck."""
    temp = ku * vol / (KB * sigma)
    return 2 * tau_libre(msat, vol, alpha, gamma, temp) / autovalor_brown(sigma)


def tau_asintotico(sigma, ku: float, msat: float, alpha: float, gamma: float):
    """Formula de Brown para barrera alta (sigma >> 1).

    tasa = 2 [alpha gamma B_K / (1 + alpha^2)] sqrt(sigma / pi) exp(-sigma)

    El factor 2 aparece porque <mz> decae al doble de la tasa de salto de un pozo.
    """
    b_k = 2 * ku / msat
    escala = (1 + alpha**2) / (alpha * gamma * b_k)
    return escala / (2 * np.sqrt(sigma / np.pi) * np.exp(-sigma))
