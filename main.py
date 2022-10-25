import numpy as np
import matplotlib.pyplot as plt
from math import pi
from numpy.fft import fft, fftshift

plt.close('all')
f1 = 1/18
f2 = 5/128
fc = 50/128
sr = 128

#0<n<255
t = np.linspace(0,255,255)
xn = np.cos(2*pi*(1/18)*t) + np.cos(2*pi*f2*t)
xc = np.cos(2*pi*fc*t)
xam = xn*xc

#0<n<99
t1 = np.linspace(0,99,99)
xn1 = np.cos(2*pi*(1/18)*t1) + np.cos(2*pi*f2*t1)
xc1 = np.cos(2*pi*fc*t1)
xam1 = xn1*xc1

#0<n<179
t2 = np.linspace(0,179,179)
xn2 = np.cos(2*pi*(1/18)*t2) + np.cos(2*pi*f2*t2)
xc2 = np.cos(2*pi*fc*t2)
xam2 = xn2*xc2

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

def DFT1(x, point):
    A=fft(x, point)
    mag = np.abs(fftshift(A))
    freq1 = np.linspace(-0.5, 0.5, len(A))
    response = 20 * np.log10(mag)
    response = np.clip(response, -100, 100)
    return freq1, response


def hanning(x, point):
    han = np.hanning(len(xam))
    han = (han*xam)
    A = fft(han, point)
    mag = np.abs(fftshift(A))
    freq1 = np.linspace(-0.5, 0.5, len(A))
    response = 20 * np.log10(mag)
    response = np.clip(response, -200, 200)
    return freq1, response

freq1, response = hanning(xam1, 128)
freq2, response1 = hanning(xam2, 256)
freq3, response2 = DFT1(xam1, 128)
freq4, response3 = DFT1(xam1, 256)


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
# plt.plot(freq2, response1)
# plt.show()
plt.plot(freq3, response2)
plt.show()
plt.plot(freq4, response3)
plt.show()



