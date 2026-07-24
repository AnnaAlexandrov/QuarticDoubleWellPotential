import numpy as np
import matplotlib.pyplot as plt

def potential(x_start=-1.5, x_stop=1.5, num=100, V0=5.0, alpha=0):
    x = np.linspace(x_start, x_stop, num)
    V = V0 * (x**2 - 1)**2 + alpha * x
    return x, V

x, V = potential()

plt.plot(x, V)
plt.show()

def harmonic_oscillator(x_start=-1.5, x_stop=1.5, num=100, omega=1, m=1):
    x = np.linspace(x_start, x_stop, num)
    V = m * omega**2 * x**2 / 2
    return x, V

x, V = harmonic_oscillator()

plt.plot(x, V)
plt.show()

def potential(x, V0=5.0, alpha=0):
    V = V0 * (x**2 - 1)**2 + alpha * x
    return V
