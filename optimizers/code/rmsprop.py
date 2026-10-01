import numpy as np

def rmsprop_step(theta, gradient, squared_avg,
                 learning_rate=0.001, rho=0.9, epsilon=1e-8):
    """One RMSProp update."""
    theta = np.asarray(theta, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    squared_avg = np.asarray(squared_avg, dtype=float)

    # EWMA of squared gradients.
    squared_avg = rho * squared_avg + (1 - rho) * gradient ** 2

    theta = theta - learning_rate * gradient / (
        np.sqrt(squared_avg) + epsilon
    )
    return theta, squared_avg

if __name__ == "__main__":
    theta = np.array([5.0, 2.0])
    squared_avg = np.zeros_like(theta)
    gradient = np.array([2.0, 0.5])
    theta, squared_avg = rmsprop_step(theta, gradient, squared_avg)
    print("Updated parameters:", theta)
    print("Squared-gradient average:", squared_avg)
