import math
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt


def f(x: float) -> float:
    return 50.0 + 20.0 * math.sin(math.pi * x / 12.0) + 5.0 * math.exp(-0.2 * (x - 12.0) ** 2)


def integral_f(x: float) -> float:
    return (
        50.0 * x
        - (240.0 / math.pi) * math.cos(math.pi * x / 12.0)
        + 2.5 * math.sqrt(5.0 * math.pi) * math.erf(math.sqrt(0.2) * (x - 12.0))
    )


def simpson_method(n: int, func: Callable[[float], float], a: float, b: float) -> float:
    h = (b - a) / n
    s = func(a) + func(b)
    for i in range(1, n, 2):
        s += 4.0 * func(a + i * h)
    for i in range(2, n, 2):
        s += 2.0 * func(a + i * h)
    return s * h / 3.0


def runge_romberg_method(i_n0: float, i_n0_2: float) -> float:
    return i_n0 + (i_n0 - i_n0_2) / 15.0


def aitken_method(i_n0: float, i_n0_2: float, i_n0_4: float) -> float:
    return (i_n0_2**2 - i_n0 * i_n0_4) / (2.0 * i_n0_2 - (i_n0 + i_n0_4))


def aitken_order(i_n0: float, i_n0_2: float, i_n0_4: float) -> float:
    return (1.0 / math.log(2.0)) * math.log(abs((i_n0_4 - i_n0_2) / (i_n0_2 - i_n0)))


def adaptive_simpson(a: float, b: float, func: Callable[[float], float], delta: float) -> tuple[float, int]:
    calls = [0]

    def f_tracked(x: float) -> float:
        calls[0] += 1
        return func(x)

    fa = f_tracked(a)
    fb = f_tracked(b)
    mid = (a + b) / 2.0
    fmid = f_tracked(mid)

    def adapt(x0: float, x1: float, f0: float, f1: float, xm: float, fm: float, d: float) -> float:
        h = x1 - x0
        xl = (x0 + xm) / 2.0
        xr = (xm + x1) / 2.0
        fl = f_tracked(xl)
        fr = f_tracked(xr)
        i1 = (h / 6.0) * (f0 + 4.0 * fm + f1)
        i2 = (h / 12.0) * (f0 + 4.0 * fl + 2.0 * fm + 4.0 * fr + f1)
        if abs(i1 - i2) <= d:
            return i2
        return adapt(x0, xm, f0, fm, xl, fl, d / 2.0) + adapt(xm, x1, fm, f1, xr, fr, d / 2.0)

    result = adapt(a, b, fa, fb, mid, fmid, delta)
    return result, calls[0]


def main() -> None:
    a, b = 0.0, 24.0
    i0 = integral_f(b) - integral_f(a)

    n_values = list(range(10, 1002, 2))
    errors = [abs(simpson_method(n, f, a, b) - i0) for n in n_values]

    n_opt, eps_opt = next((n, e) for n, e in zip(n_values, errors) if e <= 1e-12)
    n0 = max(8, round((n_opt / 10) / 8) * 8)

    i_n0 = simpson_method(n0, f, a, b)
    i_n0_2 = simpson_method(n0 // 2, f, a, b)
    i_n0_4 = simpson_method(n0 // 4, f, a, b)
    eps0 = abs(i_n0 - i0)

    i_r = runge_romberg_method(i_n0, i_n0_2)
    eps_r = abs(i_r - i0)

    i_e = aitken_method(i_n0, i_n0_2, i_n0_4)
    eps_e = abs(i_e - i0)
    p_est = aitken_order(i_n0, i_n0_2, i_n0_4)

    print("=== NUMERICAL INTEGRATION RESULTS ===")
    print(f"Exact integral value (I0):   {i0:.14f}\n")
    print("Simpson's Method:")
    print(f"  Optimal N (N_opt):         {n_opt}")
    print(f"  Error at N_opt (eps_opt):  {eps_opt:.14e}")
    print(f"  Base partitioning (N0):    {n0}")
    print(f"  Error at N0 (eps0):        {eps0:.14e}\n")
    print("Runge-Romberg Method:")
    print(f"  Refined value (I_R):       {i_r:.14f}")
    print(f"  Error (epsR):              {eps_r:.14e}\n")
    print("Aitken's Method:")
    print(f"  Refined value (I_E):       {i_e:.14f}")
    print(f"  Error (epsE):              {eps_e:.14e}")
    print(f"  Order estimate (p):        {p_est:.4f}\n")

    deltas = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    adapt_errors = []
    adapt_calls = []

    print("Adaptive Simpson Algorithm:")
    for delta in deltas:
        i_adapt, calls = adaptive_simpson(a, b, f, delta)
        err = abs(i_adapt - i0)
        adapt_errors.append(err)
        adapt_calls.append(calls)
        print(f"  Delta: {delta:5.0e} | I: {i_adapt:.12f} | Error: {err:.3e} | f(x) calls: {calls}")

    x_plot = np.linspace(a, b, 500)
    y_plot = [f(xv) for xv in x_plot]

    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, y_plot, "b-", linewidth=2, label="$f(x) = 50 + 20\\sin(\\frac{\\pi x}{12}) + 5e^{-0.2(x-12)^2}$")
    plt.fill_between(x_plot, y_plot, alpha=0.15, color="blue")
    plt.title("Server Load Function $f(x)$ on $[0, 24]$")
    plt.xlabel("Time $x$ (hours)")
    plt.ylabel("Load $f(x)$")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lab5_plot1_function.png", dpi=300)

    plt.figure(figsize=(8, 5))
    errors_safe = [max(e, 1e-16) for e in errors]
    plt.semilogy(n_values, errors_safe, "r-", linewidth=1.6, label="Error $\\varepsilon(N) = |I(N) - I_0|$")
    plt.axhline(1e-12, color="green", linestyle="--", label="Target accuracy $10^{-12}$")
    plt.plot(n_opt, max(eps_opt, 1e-16), "ko", markersize=7, label=f"$N_{{opt}} = {n_opt}$")
    plt.title("Simpson's Rule Error Dependence on Number of Partitions $N$")
    plt.xlabel("Number of partitions $N$")
    plt.ylabel("Absolute error $\\varepsilon(N)$")
    plt.xlim(0, 1000)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lab5_plot2_simpson_error.png", dpi=300)

    plt.figure(figsize=(8, 5))
    plt.loglog(adapt_calls, adapt_errors, "mo-", linewidth=1.8, markersize=7)
    for d, c, e in zip(deltas, adapt_calls, adapt_errors):
        plt.annotate(f"$\\delta={d:.0e}$", (c, e), textcoords="offset points", xytext=(8, 5), fontsize=9)
    plt.title("Adaptive Simpson Algorithm: Accuracy vs Function Evaluations")
    plt.xlabel("Number of $f(x)$ evaluations")
    plt.ylabel("Absolute error $|I_{adapt} - I_0|$")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("lab5_plot3_adaptive.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()