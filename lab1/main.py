import numpy as np


# Означення функції f(t, u)
def f(t, u):
    return np.exp(-t ** 2) * np.sin(t) * ((2 * np.pi) ** (1 / 2)) - 2 * (t - 2) * u


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
            t_new = t + h
            if t_new > t_end:
                t_new = t_end  # Зупинити обчислення, якщо нове значення t виходить за межі t_end
            t_values.append(t_new)
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


# Визначення точного розв'язку
def exact_solution(t):
    return np.exp(-t ** 2) * np.sin(t)


# Виведення результатів та похибки
print(
    "{:<10} {:<27} {:<28} {:<25}".format("Час (t)", "Чисельний результат (u)", "Точний результат (u_exact)", "Похибка"))
for t, u in zip(t_values, u_values):
    u_exact = exact_solution(t)
    error = abs(u - u_exact)
    print("{:<10.3f} {:<27.10f} {:<28.15f} {:<25.10f}".format(t, u, u_exact, error))

# Максимальна похибка
max_error = max([abs(u - exact_solution(t)) for t, u in zip(t_values, u_values)])
print(f"\nМаксимальна похибка: {max_error:.5f}")

# Виведення результату у точці t0 та T
t0_result = u_values[t_values.index(t0)]
T_result = u_values[-1]
print(f"\nРезультат в точці t0 (t = {t0:.1f}): u = {t0_result:.2f}")
print(f"Результат в точці T (t = {t_end:.1f}): u = {T_result:.5e}")
