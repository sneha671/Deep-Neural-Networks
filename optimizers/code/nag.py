import numpy as np

def nag_step(theta, gradient_fn, velocity, learning_rate=0.01, beta=0.9):
    """One Nesterov Accelerated Gradient update."""
    theta = np.asarray(theta, dtype=float)
    velocity = np.asarray(velocity, dtype=float)

    # Look ahead using current momentum.
    lookahead = theta - learning_rate * beta * velocity

    # Evaluate gradient at the look-ahead point.
    gradient = np.asarray(gradient_fn(lookahead), dtype=float)

    velocity = beta * velocity + (1 - beta) * gradient
    theta = theta - learning_rate * velocity
    return theta, velocity, lookahead

if __name__ == "__main__":
    def grad(theta):
        return theta

    theta = np.array([5.0, 2.0])
    velocity = np.zeros_like(theta)
    theta, velocity, lookahead = nag_step(theta, grad, velocity)
    print("Look-ahead:", lookahead)
    print("Updated parameters:", theta)
