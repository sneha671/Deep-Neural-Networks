import numpy as np

def sgd_step(theta, gradient, learning_rate=0.01):
    """One vanilla SGD update."""
    theta = np.asarray(theta, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    return theta - learning_rate * gradient

if __name__ == "__main__":
    theta = np.array([5.0, 2.0])
    gradient = np.array([2.0, 0.5])
    print("Updated parameters:", sgd_step(theta, gradient))
