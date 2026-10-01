import numpy as np

def momentum_step(theta, gradient, velocity, learning_rate=0.01, beta=0.9):
    """One SGD-with-Momentum update."""
    theta = np.asarray(theta, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    velocity = np.asarray(velocity, dtype=float)

    velocity = beta * velocity + (1 - beta) * gradient
    theta = theta - learning_rate * velocity
    return theta, velocity

if __name__ == "__main__":
    theta = np.array([5.0, 2.0])
    velocity = np.zeros_like(theta)
    gradient = np.array([2.0, 0.5])
    theta, velocity = momentum_step(theta, gradient, velocity)
    print("Updated parameters:", theta)
    print("Velocity:", velocity)
