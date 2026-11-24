import csv
import numpy as np
import matplotlib.pyplot as plt


def write_initial_csv(filename: str) -> None:
    rows = [
        (10000, 8),
        (20000, 20),
        (40000, 55),
        (80000, 150),
        (160000, 420),
    ]
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["n", "t"])
        writer.writerows(rows)


def read_data(filename: str) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    with open(filename, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row["n"]))
            y.append(float(row["t"]))
    return np.array(x), np.array(y)


def omega(x_nodes: np.ndarray, x: float) -> float:
    return float(np.prod(x - x_nodes))


def divided_differences(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j - 1 : n - 1]) / (x[j:n] - x[0 : n - j])
    return coef


def newton_interpolation(x_nodes: np.ndarray, coef: np.ndarray, x: float) -> float:
    n = len(coef) - 1
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (x - x_nodes[n - k]) * p
    return float(p)


def factorial_interpolation(x_nodes: np.ndarray, y_nodes: np.ndarray, x: float) -> float:
    n = len(x_nodes)
    h = x_nodes[1] - x_nodes[0]
    t = (x - x_nodes[0]) / h
    diff = np.copy(y_nodes).astype(float)
    res = diff[0]
    t_prod = 1.0
    for k in range(1, n):
        diff = np.diff(diff)
        t_prod *= (t - (k - 1)) / k
        res += diff[0] * t_prod
    return float(res)


def f_model(x: np.ndarray | float) -> np.ndarray | float:
    return 1.417542e-05 * (x ** 1.4335) + 10.0 / (1.0 + ((x - 85000.0) / 15000.0) ** 2)


def main() -> None:
    write_initial_csv("data.csv")
    x_exp, y_exp = read_data("data.csv")

    target_x = 120000.0
    coef_exp = divided_differences(x_exp, y_exp)
    pred_exp = newton_interpolation(x_exp, coef_exp, target_x)

    print("=== VARIANT 3: TRAINING TIME PREDICTION ===")
    print(f"Experimental data from 'data.csv': {len(x_exp)} nodes")
    print(f"Newton prediction at n = {target_x:.0f}: {pred_exp:.4f} sec\n")

    a, b = 10000.0, 160000.0
    x_dense = np.linspace(a, b, 500)
    y_true = f_model(x_dense)

    print("=== RESEARCH 1: FIXED INTERVAL [10000, 160000], n = 5, 10, 20 ===")
    print(f"Reference model value at n = {target_x:.0f}: {f_model(target_x):.4f} sec")
    errors_fixed_interval = {}
    curves_fixed_interval = {}

    for n_nodes in [5, 10, 20]:
        xn = np.linspace(a, b, n_nodes + 1)
        yn = f_model(xn)
        cn = divided_differences(xn, yn)
        y_interp = np.array([newton_interpolation(xn, cn, xv) for xv in x_dense])
        err = np.abs(y_true - y_interp)
        pred_n = newton_interpolation(xn, cn, target_x)
        pred_f = factorial_interpolation(xn, yn, target_x)
        errors_fixed_interval[n_nodes] = err
        curves_fixed_interval[n_nodes] = y_interp
        print(
            f"  n = {n_nodes:2d} -> Max error: {np.max(err):.4e} | "
            f"Newton(120000): {pred_n:.4f} | Factorial(120000): {pred_f:.4f}"
        )

    print("\n=== RESEARCH 2: FIXED STEP h = 7500, n = 5, 10, 20 ===")
    h_const = 7500.0
    for n_nodes in [5, 10, 20]:
        b_var = a + h_const * n_nodes
        xn = np.linspace(a, b_var, n_nodes + 1)
        yn = f_model(xn)
        cn = divided_differences(xn, yn)
        xd = np.linspace(a, b_var, 500)
        y_interp = np.array([newton_interpolation(xn, cn, xv) for xv in xd])
        err = np.abs(f_model(xd) - y_interp)
        print(f"  n = {n_nodes:2d}, interval [{a:.0f}, {b_var:.0f}] -> Max error: {np.max(err):.4e}")

    plt.figure(figsize=(8, 5))
    y_exp_curve = np.array([newton_interpolation(x_exp, coef_exp, xv) for xv in x_dense])
    plt.plot(x_dense, y_exp_curve, "b-", linewidth=2, label="Newton polynomial (5 exp. nodes)")
    plt.scatter(x_exp, y_exp, color="red", s=60, zorder=5, label="Experimental nodes")
    plt.plot(target_x, pred_exp, "g*", markersize=12, zorder=6, label=f"Prediction (120000) = {pred_exp:.1f} s")
    plt.title("Model Training Time Interpolation (Variant 3)")
    plt.xlabel("Dataset size $n$")
    plt.ylabel("Training time $t$ (sec)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lab2_plot1_interpolation.png", dpi=300)

    plt.figure(figsize=(8, 5))
    plt.plot(x_dense, y_true, "k--", linewidth=2, label="Reference function $f(n)$")
    for n_nodes, color in [(5, "blue"), (10, "green"), (20, "red")]:
        plt.plot(x_dense, curves_fixed_interval[n_nodes], color=color, linewidth=1.5, label=f"$N_{{{n_nodes}}}(n)$")
    plt.title("Runge Phenomenon Analysis for $n = 5, 10, 20$")
    plt.xlabel("Dataset size $n$")
    plt.ylabel("Training time $t$ (sec)")
    plt.ylim(-100, 600)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lab2_plot2_runge.png", dpi=300)

    plt.figure(figsize=(8, 5))
    for n_nodes, color in [(5, "blue"), (10, "green"), (20, "red")]:
        plt.semilogy(x_dense, errors_fixed_interval[n_nodes] + 1e-16, color=color, linewidth=1.6, label=f"Error $n = {n_nodes}$")
    plt.title("Interpolation Error $\\varepsilon(n) = |f(n) - N_n(n)|$ (Log Scale)")
    plt.xlabel("Dataset size $n$")
    plt.ylabel("Absolute error $\\varepsilon(n)$")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lab2_plot3_errors.png", dpi=300)

    plt.figure(figsize=(8, 5))
    x20 = np.linspace(a, b, 21)
    w_dense = [omega(x20, xv) for xv in x_dense]
    plt.plot(x_dense, w_dense, color="purple", linewidth=1.8, label="$\\omega_{20}(n)$")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Nodal Polynomial $\\omega_{20}(n)$ on $[10000, 160000]$")
    plt.xlabel("Dataset size $n$")
    plt.ylabel("$\\omega_{20}(n)$")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lab2_plot4_omega.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()