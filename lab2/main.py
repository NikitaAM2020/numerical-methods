import numpy as np

def f(t, y):
    y1_prime = (y[1] - y[0]) * t
    y2_prime = (y[1] + y[0]) * t
    return np.array([y1_prime, y2_prime])

def solve_runge_kutta(t, y, tau):
    k1 = tau * f(t, y)
    k2 = tau * f(t + tau / 2, y + k1 / 2)
    k3 = tau * f(t + tau, y - k1 + 2 * k2)
    y_next = y + (k1 + 4 * k2 + k3) / 6
    return y_next

def runge_kutta_adaptive(t0, T, y0, epsilon, epsilon_M):
    t = t0
    y = np.array(y0)
    tau = 0.1  # Initial step size
    iterations = 0  # Initialize iteration count

    # Print initial values
    print("Iteration:", iterations, "t =", "{:.5f}".format(t), "y =", ["{:.5f}".format(val) for val in y])

    while abs(T - t) >= epsilon_M:
        if t + tau > T:
            tau = T - t

        y_next = solve_runge_kutta(t, y, tau)

        max_diff = max(abs(y_next - y)) / ((2 ** 3 - 1) * max(1, max(abs(y_next))))
        tau_H = tau * min(5, max(0.1, 0.9 * (epsilon / max_diff) ** (1 / 4)))

        if max_diff <= epsilon:
            t = t + tau
            y = y_next
            tau = min(tau_H, T - t)
            iterations += 1
            print("Iteration:", iterations, "t =", "{:.5f}".format(t), "y =", ["{:.5f}".format(val) for val in y])
        else:
            tau = tau_H
            y_next = y
            iterations += 1

if __name__ == "__main__":
    t0 = 0.0
    T = 1.0
    y0 = [1.0, 0.0]  # Initial values for your system of ODEs
    epsilon = 1e-4
    epsilon_M = 1e-6

    runge_kutta_adaptive(t0, T, y0, epsilon, epsilon_M)
