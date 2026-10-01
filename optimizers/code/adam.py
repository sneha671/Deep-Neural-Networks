import numpy as np

def adam_step(theta, gradient, m, v, t,
              learning_rate=0.001, beta1=0.9,
              beta2=0.999, epsilon=1e-8):
    """One Adam update with bias correction."""
    theta = np.asarray(theta, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)

    # First moment: EWMA of gradients.
    m = beta1 * m + (1 - beta1) * gradient

    # Second moment: EWMA of squared gradients.
    v = beta2 * v + (1 - beta2) * gradient ** 2

    # Bias correction.
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # Parameter update.
    theta = theta - learning_rate * m_hat / (
        np.sqrt(v_hat) + epsilon
    )

    return theta, m, v

if __name__ == "__main__":
    theta = np.array([5.0, 2.0])
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    gradient = np.array([2.0, 0.5])
    theta, m, v = adam_step(theta, gradient, m, v, t=1)
    print("Updated parameters:", theta)
