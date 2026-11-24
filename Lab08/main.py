import numpy as np
import matplotlib.pyplot as plt


def F(x):
    return np.sin(x) - 0.15 * x + 0.3


def dF(x):
    return np.cos(x) - 0.15


def ddF(x):
    return -np.sin(x)


def simple_iteration(F, dF, x0, eps=1e-10, max_iter=10000):
    tau = -0.5 / dF(x0)
    x = x0
    for it in range(1, max_iter + 1):
        x_next = x + tau * F(x)
        if abs(F(x_next)) < eps and abs(x_next - x) < eps:
            return x_next, it
        x = x_next
    return x, max_iter


def newton(F, dF, x0, eps=1e-10, max_iter=1000):
    x = x0
    for it in range(1, max_iter + 1):
        x_next = x - F(x) / dF(x)
        if abs(F(x_next)) < eps and abs(x_next - x) < eps:
            return x_next, it
        x = x_next
    return x, max_iter


def chebyshev(F, dF, ddF, x0, eps=1e-10, max_iter=1000):
    x = x0
    for it in range(1, max_iter + 1):
        fx = F(x)
        dfx = dF(x)
        ddfx = ddF(x)
        x_next = x - fx / dfx - 0.5 * (fx**2 * ddfx) / (dfx**3)
        if abs(F(x_next)) < eps and abs(x_next - x) < eps:
            return x_next, it
        x = x_next
    return x, max_iter


def secant(F, x0, x1, eps=1e-10, max_iter=1000):
    for it in range(1, max_iter + 1):
        f0, f1 = F(x0), F(x1)
        x_next = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(F(x_next)) < eps and abs(x_next - x1) < eps:
            return x_next, it
        x0, x1 = x1, x_next
    return x1, max_iter


def parabola(F, x0, x1, x2, eps=1e-10, max_iter=1000):
    for it in range(1, max_iter + 1):
        f0, f1, f2 = F(x0), F(x1), F(x2)
        f21 = (f2 - f1) / (x2 - x1)
        f10 = (f1 - f0) / (x1 - x0)
        f210 = (f21 - f10) / (x2 - x0)

        w = f21 + (x2 - x1) * f210
        disc = np.sqrt(w**2 - 4.0 * f2 * f210 + 0j)
        d1 = (-w + disc) / (2.0 * f210)
        d2 = (-w - disc) / (2.0 * f210)
        delta = d1 if abs(d1) < abs(d2) else d2
        x_next = (x2 + delta).real
        if abs(F(x_next)) < eps and abs(x_next - x2) < eps:
            return x_next, it
        x0, x1, x2 = x1, x2, x_next
    return x2, max_iter


def inverse_interp(F, x0, x1, x2, eps=1e-10, max_iter=1000):
    for it in range(1, max_iter + 1):
        y0, y1, y2 = F(x0), F(x1), F(x2)
        t0 = (y1 * y2) / ((y0 - y1) * (y0 - y2)) * x0
        t1 = (y0 * y2) / ((y1 - y0) * (y1 - y2)) * x1
        t2 = (y0 * y1) / ((y2 - y0) * (y2 - y1)) * x2
        x_next = t0 + t1 + t2
        if abs(F(x_next)) < eps and abs(x_next - x2) < eps:
            return x_next, it
        x0, x1, x2 = x1, x2, x_next
    return x2, max_iter


def read_poly_coeffs(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return [float(val) for val in file.read().split()]


def eval_poly(a, x):
    res = 0.0
    for coef in reversed(a):
        res = res * x + coef
    return res


def horner_newton(a, x0, eps=1e-10, max_iter=1000):
    m = len(a) - 1
    x = x0
    for it in range(1, max_iter + 1):
        b = [0.0] * (m + 1)
        b[m] = a[m]
        for i in range(m - 1, -1, -1):
            b[i] = a[i] + x * b[i + 1]
        c = [0.0] * (m + 1)
        c[m] = b[m]
        for i in range(m - 1, 0, -1):
            c[i] = b[i] + x * c[i + 1]
        x_next = x - b[0] / c[1]
        if abs(b[0]) < eps and abs(x_next - x) < eps:
            return x_next, it
        x = x_next
    return x, max_iter


def lin_method(a, alpha0, beta0, eps=1e-10, max_iter=1000):
    m = len(a) - 1
    alpha, beta = alpha0, beta0
    for it in range(1, max_iter + 1):
        p = -2.0 * alpha
        q = alpha**2 + beta**2
        b = [0.0] * (m + 1)
        b[m] = a[m]
        b[m - 1] = a[m - 1] - p * b[m]
        for i in range(m - 2, 1, -1):
            b[i] = a[i] - p * b[i + 1] - q * b[i + 2]
        q_new = a[0] / b[2]
        p_new = (a[1] * b[2] - a[0] * b[3]) / (b[2]**2)
        alpha_new = -p_new / 2.0
        beta_new = np.sqrt(abs(q_new - alpha_new**2))
        if abs(alpha_new - alpha) <= eps and abs(beta_new - beta) <= eps:
            return alpha_new, beta_new, it
        alpha, beta = alpha_new, beta_new
    return alpha, beta, max_iter


xs = np.arange(-2.0, 5.0 + 1e-9, 0.1)
ys = F(xs)

with open("tabulation.txt", "w", encoding="utf-8") as file:
    file.write("x\t\tF(x)\n")
    for x_val, y_val in zip(xs, ys):
        file.write(f"{x_val:.2f}\t\t{y_val:.6f}\n")

roots_intervals = []
for i in range(len(xs) - 1):
    if ys[i] * ys[i + 1] <= 0:
        behavior = "increasing" if ys[i + 1] > ys[i] else "decreasing"
        roots_intervals.append((xs[i], xs[i + 1], behavior))

print("=== PART 1: TRANSCENDENTAL EQUATION F(x) = sin(x) - 0.15x + 0.3 = 0 ===")
print("Tabulation saved to 'tabulation.txt'.")

found_roots = []
for idx, (xl, xr, beh) in enumerate(roots_intervals, 1):
    x0 = (xl + xr) / 2.0
    print(f"\nRoot {idx} ({beh}), initial guess x0 = {x0:.2f}:")
    methods = [
        ("Simple iteration", simple_iteration(F, dF, x0)),
        ("Newton", newton(F, dF, x0)),
        ("Chebyshev", chebyshev(F, dF, ddF, x0)),
        ("Secant (Chords)", secant(F, xl, xr)),
        ("Parabola", parabola(F, xl, x0, xr)),
        ("Inverse interp", inverse_interp(F, xl, x0, xr)),
    ]
    for name, (root_val, iters) in methods:
        print(f"  {name:<18}: x = {root_val:.12f}, iterations = {iters}")
    found_roots.append(methods[1][1][0])

with open("poly_coeffs.txt", "w", encoding="utf-8") as file:
    file.write("6.0 -4.0 1.0 1.0\n")

a_coeffs = read_poly_coeffs("poly_coeffs.txt")
real_root, it_real = horner_newton(a_coeffs, -2.5)
alpha, beta, it_lin = lin_method(a_coeffs, 0.5, 0.5)

print("\n=== PART 2: ALGEBRAIC EQUATION P(x) = x^3 + x^2 - 4x + 6 = 0 ===")
print(f"Coefficients read from 'poly_coeffs.txt' (a0..a3): {a_coeffs}")
print(f"Check P(-3.0) = {eval_poly(a_coeffs, -3.0):.6f}")
print(f"Horner-Newton real root:    x = {real_root:.12f}, iterations = {it_real}")
print(f"Lin's method complex roots: x = {alpha:.12f} +/- {beta:.12f}i, iterations = {it_lin}")


plt.figure(figsize=(7, 5))
x_plot = np.linspace(-2.0, 5.0, 400)
plt.plot(x_plot, F(x_plot), 'b-', linewidth=2, label='$F(x) = \\sin(x) - 0.15x + 0.3$')
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.plot(found_roots, [0, 0], 'ro', markersize=8, label='Roots')
plt.title("Transcendental Function $F(x) = \\sin(x) - 0.15x + 0.3$")
plt.xlabel("$x$")
plt.ylabel("$F(x)$")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("lab8_plot1_transcendental.png", dpi=300)

plt.figure(figsize=(7, 5))
x_poly = np.linspace(-4.5, 3.0, 400)
y_poly = [eval_poly(a_coeffs, xv) for xv in x_poly]
plt.plot(x_poly, y_poly, 'g-', linewidth=2, label='$P(x) = x^3 + x^2 - 4x + 6$')
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.plot(real_root, 0, 'r*', markersize=12, label='Real root $x = -3$')
plt.title("Cubic Polynomial $P(x) = x^3 + x^2 - 4x + 6$")
plt.xlabel("$x$")
plt.ylabel("$P(x)$")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("lab8_plot2_polynomial.png", dpi=300)

plt.show()