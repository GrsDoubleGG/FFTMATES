"""
wavetable.py - genera una wavetable (64 frames de 2048 muestras) disenando
cada frame en el dominio de la frecuencia y regresando al tiempo con una IFFT.

Uso: python scripts/wavetable.py      -> crea sierra.wav
Cargalo en Serum, Vital o Pigments como wavetable (frames de 2048 muestras).
"""
import numpy as np
from scipy.io import wavfile

N, F = 2048, 64            # muestras por frame, frames
tabla = []
for j in range(F):
    X = np.zeros(N // 2 + 1, complex)  # bins 0..N/2
    K = 1 + 8 * j                      # armonicos del frame j
    k = np.arange(1, K + 1)
    X[1:K + 1] = -1j / k               # sierra: amplitud 1/k
    tabla.append(np.fft.irfft(X, n=N))   # IFFT -> 1 ciclo
tabla = np.concatenate(tabla)
tabla /= np.abs(tabla).max()
wavfile.write("sierra.wav", 48000, tabla.astype(np.float32))
print("sierra.wav:", F, "frames de", N, "muestras")
