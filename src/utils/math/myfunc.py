import numpy as np

def normal(x, mu, sigma) -> float:
    p = 1 / np.sqrt(2 * np.pi * sigma**2)
    return p * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
