import numpy as np
import matplotlib.pyplot as plt
from math import pi
from numpy.fft import fft, fftshift

plt.close('all')
f1 = 1/18
f2 = 5/128
fc = 50/128
sr = 128

t = np.linspace(0,255,255)
xn = np.cos(2*pi*(1/18)*t) + np.cos(2*pi*f2*t)
xc = np.cos(2*pi*fc*t)
xam = xn*xc

def DFT(x):
    """
    Function to calculate the
    discrete Fourier Transform
    of a 1D real-valued signal x
    """

    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)

    X = np.dot(e, x)
    return X

def hanning(x):
    han = np.hanning(len(xam))
    A = fft(han)
    mag = np.abs(fftshift(A))
    freq1 = np.linspace(-0.5, 0.5, len(A))
    response = 20 * np.log10(mag)
    response = np.clip(response, -100, 100)
    return freq1, response

freq1, response = hanning(xam)

dft=DFT(xam)
N = len(dft)
n = np.arange(N)
T = N/sr
freq = n/T

# plt.plot(xn)
# plt.show()
# plt.plot(xc)
# plt.show()
# plt.plot(xam)
# plt.show()
# plt.stem(freq, abs(dft), 'b',
#          markerfmt=" ", basefmt="-b")
# plt.show()
plt.plot(freq1, response)
plt.show()


