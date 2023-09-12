import numpy as np


# Означення функції f(t, u)
def f(t, u):
    return np.exp(-t ** 2) * np.sin(t) / ((2 * np.pi) ** (1 / 2)) - 2 * (t - 2) * u


# Означення методу Рунге-Кутта 3-го порядку
def runge_kutta_3(t, u, h):
    k1 = h * f(t, u)
    k2 = h * f(t + h / 2, u + k1 / 2)
    k3 = h * f(t + h, u - k1 + 2 * k2)
    return u + (k1 + 4 * k2 + k3) / 6


# Означення функції для розв'язання задачі Коші з вибором кроку
def solve_rk3_adaptive(f, t0, u0, t_end, tol):
    t_values = [t0]
    u_values = [u0]
    h = 0.1  # Початковий крок

    while t_values[-1] < t_end:
        t = t_values[-1]
        u = u_values[-1]

        # Обчислення двох значень з різними кроками
        u1 = runge_kutta_3(t, u, h)
        u2 = runge_kutta_3(t, u, h / 2)
        u2 = runge_kutta_3(t + h / 2, u2, h / 2)

        # Обчислення оцінки похибки
        error = abs(u2 - u1)

        # Вибір нового кроку на основі оцінки похибки та заданої точності (tol)
        if error < tol:
            t_values.append(t + h)
            u_values.append(u2)
        h = h * min(max(0.1, 0.8 * (tol / error) ** (1 / 3)), 2.0)  # Вибір нового кроку

    return t_values, u_values


# Початкові значення
t0 = 1.0
t_end = 6.0
u0 = 10.0
tolerance = 1e-5

# Розв'язання задачі Коші
t_values, u_values = solve_rk3_adaptive(f, t0, u0, t_end, tolerance)

# Виведення результатів
for t, u in zip(t_values, u_values):
    print(f"t = {t:.5f}, u = {u:.10f}")

# Виведення першого та останнього результатів
print(f"\nРезультат для t = {t_values[0]:.2f}, u = {u_values[0]:.2f}")
print(f"Результат для t = {t_values[-1]:.2f}, u = {u_values[-1]:.5e}")
# Виведення максимального u
max_u = max(u_values)
print(f"Максимальне значення u = {max_u:.5f}")
# Знаходження індексу максимального u
max_u_index = u_values.index(max_u)
# Виведення відповідного значення t
t_max_u = t_values[max_u_index]
print(f"Значення t для максимального u: t = {t_max_u:.5f}")
