import numpy as np

def adagrad_step(theta, gradient, accumulator, learning_rate=0.01, epsilon=1e-8):
    """One AdaGrad update."""
    theta = np.asarray(theta, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    accumulator = np.asarray(accumulator, dtype=float)

    accumulator = accumulator + gradient ** 2
    theta = theta - learning_rate * gradient / (np.sqrt(accumulator) + epsilon)
    return theta, accumulator

if __name__ == "__main__":
    theta = np.array([5.0, 2.0])
    accumulator = np.zeros_like(theta)
    gradient = np.array([2.0, 0.5])
    theta, accumulator = adagrad_step(theta, gradient, accumulator)
    print("Updated parameters:", theta)
    print("Accumulator:", accumulator)
