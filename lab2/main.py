import numpy as np


def f(t, y):
    # Define your system of ODEs here
    # Example: return np.array([y[1], -y[0]])
    pass


def runge_kutta_adaptive(t0, T, y0, N, epsilon, tau0, epsilon_M, p):
    t = t0
    tau = tau0
    y = np.array(y0)
    e_max = 0.0

    while abs(T - t) >= epsilon_M:
        if t + tau > T:
            tau = T - t

        v = np.copy(y)
        t1 = t
        kf = 0

        k1 = tau * f(t, y)

        while kf == 0:
            k = [k1]
            c = [0]
            a = [[0]]
            b = [1]

            for s in range(2, p + 1):
                c.append(sum(a[s - 2]))
                a_s = [0] * s
                for j in range(1, s - 1):
                    a_s[j] = sum(a[j - 1])
                a.append(a_s)
                b.append(0)

            for s in range(2, p + 1):
                k_s = tau * f(t + c[s - 2] * tau, v + np.dot(np.array(a[s - 2]), np.array(k)))
                k.append(k_s)
                b[s - 1] = 1 / (s * (s - 1)) * sum(b[i] * a[s - 2][i - 1] for i in range(1, s))

            for i in range(N):
                y[i] = y[i] + tau * sum(b[s - 1] * k[s - 1][i] for s in range(2, p + 1))

            if kf == 0:
                w = np.copy(y)
                y = np.copy(v)
                tau = tau / 2.0
                kf = 1

        if kf == 1:
            t = t + tau
            kf = 2
            continue

        max_diff = max(abs(y - w) / ((2 ** p - 1) * max(1, max(abs(y)))))
        tau_H = 2 * tau * min(5, max(0.1, 0.9 * (epsilon / max_diff) ** (1 / (p + 1))))

        if max_diff <= epsilon:
            t = t + tau
            y = y + (y - w) / (2 ** p - 1)
            print("t =", t, "y =", y)
            tau = tau_H
            continue
        else:
            y = np.copy(v)
            t = t1
            tau = tau_H


if __name__ == "__main__":
    t0 = 0.0
    T = 1.0
    y0 = [1.0, 0.0]  # Initial values for your system of ODEs
    N = len(y0)
    epsilon = 1e-6
    tau0 = 0.1
    epsilon_M = 1e-4
    p = 4

    runge_kutta_adaptive(t0, T, y0, N, epsilon, tau0, epsilon_M, p)
