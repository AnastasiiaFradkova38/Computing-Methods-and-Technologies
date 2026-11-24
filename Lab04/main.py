import math
import numpy as np
import matplotlib.pyplot as plt


def f(x: float) -> float:
    return 50.0 * math.exp(-0.1 * x) + 5.0 * math.sin(x)


def f_derivative(x: float) -> float:
    return -5.0 * math.exp(-0.1 * x) + 5.0 * math.cos(x)


def central_difference(x: float, h: float) -> float:
    return (f(x + h) - f(x - h)) / (2.0 * h)


def runge_romberg_method(y_h: float, y_2h: float) -> float:
    return y_h + (y_h - y_2h) / 3.0


def aitkens_method(y_h: float, y_2h: float, y_4h: float) -> tuple[float, float]:
    denominator = 2.0 * y_2h - (y_4h + y_h)
    y_e = (y_2h**2 - y_4h * y_h) / denominator
    p = (1.0 / math.log(2.0)) * math.log(abs((y_4h - y_2h) / (y_2h - y_h)))
    return y_e, p


def main() -> None:
    x0 = 1.0
    exact_val = f_derivative(x0)

    h_values = np.logspace(-20, 3, 2000)
    errors = np.array([abs(central_difference(x0, h) - exact_val) for h in h_values])

    min_idx = int(np.argmin(errors))
    h0 = h_values[min_idx]
    r0 = errors[min_idx]

    h = 1e-3
    y_h = central_difference(x0, h)
    y_2h = central_difference(x0, 2.0 * h)
    y_4h = central_difference(x0, 4.0 * h)
    r1 = abs(y_h - exact_val)

    y_r = runge_romberg_method(y_h, y_2h)
    r2 = abs(y_r - exact_val)

    y_e, p_est = aitkens_method(y_h, y_2h, y_4h)
    r3 = abs(y_e - exact_val)

    print("=== NUMERICAL DIFFERENTIATION RESULTS ===")
    print(f"Exact derivative M'({x0}):   {exact_val:.12f}\n")
    print(f"Optimal step h0:             {h0:.3e}")
    print(f"Minimum error R0:            {r0:.3e}\n")
    print(f"Step h = {h}:")
    print(f"  y'(h)  = {y_h:.12f}")
    print(f"  y'(2h) = {y_2h:.12f}")
    print(f"  y'(4h) = {y_4h:.12f}")
    print(f"  Central diff error R1:     {r1:.3e}\n")
    print(f"Runge-Romberg method:")
    print(f"  y'_R   = {y_r:.12f}")
    print(f"  Error R2:                  {r2:.3e}\n")
    print(f"Aitken's method:")
    print(f"  y'_E   = {y_e:.12f}")
    print(f"  Error R3:                  {r3:.3e}")
    print(f"  Estimated order p:         {p_est:.4f}")

    plt.figure(figsize=(8, 5))
    plt.loglog(h_values, errors, 'b-', linewidth=1.8, label='Absolute error $R(h)$')
    plt.loglog(h0, r0, 'ro', markersize=8, label=f'Optimal $h_0 = {h0:.2e}$, $R_0 = {r0:.2e}$')
    plt.title("Dependence of Numerical Differentiation Error on Step Size $h$")
    plt.xlabel("Step size $h$")
    plt.ylabel("Absolute error $R = |y_0'(h) - y'(x_0)|$")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lab4_plot_error.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()