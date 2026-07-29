import numpy as np
import matplotlib.pyplot as plt

def potential(x, V0=5.0, alpha=0):
    V = V0 * (x**2 - 1)**2 + alpha * x
    return V
