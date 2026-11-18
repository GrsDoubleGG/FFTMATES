"""
benchmark.py - compara la DFT directa con la FFT recursiva y numpy.fft.
Uso: python scripts/benchmark.py
"""
import time

import numpy as np


def dft(x):
    """DFT por definicion: O(N^2)."""
    N = len(x)
    n = np.arange(N)
    W = np.exp(-2j * np.pi * np.outer(n, n) / N)   # matriz de Fourier
    return W @ x


def fft(x):
    """FFT radix-2 recursiva (Cooley-Tukey). N debe ser potencia de 2."""
    N = len(x)
    if N == 1:
        return x
    E = fft(x[0::2])                                  # pares
    O = fft(x[1::2])                                  # impares
    W = np.exp(-2j * np.pi * np.arange(N // 2) / N)   # factores de giro
    return np.concatenate([E + W * O, E - W * O])


def cronometrar(f, x, rep=3):
    mejor = np.inf
    for _ in range(rep):
        t0 = time.perf_counter()
        f(x)
        mejor = min(mejor, time.perf_counter() - t0)
    return mejor


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for N in (256, 1024, 4096):
        x = rng.standard_normal(N) + 0j
        assert np.allclose(fft(x), np.fft.fft(x))
        assert np.allclose(dft(x), np.fft.fft(x))
        t1 = cronometrar(dft, x)
        t2 = cronometrar(fft, x)
        t3 = cronometrar(np.fft.fft, x, rep=50)
        print(f"N = {N:5d}   DFT matriz: {t1*1e3:9.2f} ms   FFT recursiva: {t2*1e3:7.2f} ms"
              f"   numpy.fft: {t3*1e6:7.1f} us")
