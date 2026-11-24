import random
import matplotlib.pyplot as plt


def generate_matrix(filename: str, n: int = 100) -> None:
    matrix = [[random.uniform(-10.0, 10.0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] += 1500.0
    with open(filename, "w", encoding="utf-8") as file:
        for row in matrix:
            file.write(" ".join(f"{val:.15f}" for val in row) + "\n")


def read_matrix_from_file(filename: str) -> list[list[float]]:
    matrix = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                matrix.append([float(val) for val in line.split()])
    return matrix


def save_vector_to_file(filename: str, vector: list[float]) -> None:
    with open(filename, "w", encoding="utf-8") as file:
        for val in vector:
            file.write(f"{val:.15f}\n")


def read_vector_from_file(filename: str) -> list[float]:
    vector = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                vector.append(float(line.strip()))
    return vector


def multiply_matrix_vector(a: list[list[float]], x: list[float]) -> list[float]:
    return [sum(row[i] * x[i] for i in range(len(row))) for row in a]


def lu_decomposition(a: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    n = len(a)
    l = [[0.0] * n for _ in range(n)]
    u = [[0.0] * n for _ in range(n)]
    for i in range(n):
        u[i][i] = 1.0
    for k in range(n):
        for i in range(k, n):
            sum_l = sum(l[i][j] * u[j][k] for j in range(k))
            l[i][k] = a[i][k] - sum_l
        for i in range(k + 1, n):
            sum_u = sum(l[k][j] * u[j][i] for j in range(k))
            u[k][i] = (a[k][i] - sum_u) / l[k][k]
    return l, u


def save_lu_to_file(filename: str, l: list[list[float]], u: list[list[float]]) -> None:
    with open(filename, "w", encoding="utf-8") as file:
        file.write("Matrix L:\n")
        for row in l:
            file.write(" ".join(f"{val:.15f}" for val in row) + "\n")
        file.write("\nMatrix U:\n")
        for row in u:
            file.write(" ".join(f"{val:.15f}" for val in row) + "\n")


def solve_lu(l: list[list[float]], u: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    z = [0.0] * n
    for k in range(n):
        sum_z = sum(l[k][j] * z[j] for j in range(k))
        z[k] = (b[k] - sum_z) / l[k][k]
    x = [0.0] * n
    for k in range(n - 1, -1, -1):
        sum_x = sum(u[k][j] * x[j] for j in range(k + 1, n))
        x[k] = z[k] - sum_x
    return x


def vector_norm(v: list[float]) -> float:
    return max(abs(val) for val in v)


def main() -> None:
    random.seed(42)
    n = 100
    matrix_file = "A.txt"
    vector_file = "B.txt"
    lu_file = "LU.txt"
    eps_0 = 1e-14

    generate_matrix(matrix_file, n)
    a = read_matrix_from_file(matrix_file)
    ideal_x = [2.5] * n
    b_calculated = multiply_matrix_vector(a, ideal_x)
    save_vector_to_file(vector_file, b_calculated)
    b = read_vector_from_file(vector_file)

    l, u = lu_decomposition(a)
    save_lu_to_file(lu_file, l, u)

    x_calc = solve_lu(l, u, b)
    ax_calc = multiply_matrix_vector(a, x_calc)
    initial_error = vector_norm([ax_calc[i] - b[i] for i in range(n)])

    print("=== LU DECOMPOSITION & ITERATIVE REFINEMENT (n = 100) ===")
    print(f"Initial residual norm (LU): {initial_error:.6e}")

    error_history = [initial_error]
    iterations = 0
    x_current = x_calc[:]

    while True:
        iterations += 1
        ax_current = multiply_matrix_vector(a, x_current)
        r = [b[i] - ax_current[i] for i in range(n)]
        delta_x = solve_lu(l, u, r)
        x_current = [x_current[i] + delta_x[i] for i in range(n)]

        ax_new = multiply_matrix_vector(a, x_current)
        current_error_norm = vector_norm([ax_new[i] - b[i] for i in range(n)])
        delta_norm = vector_norm(delta_x)
        error_history.append(current_error_norm)

        print(f"  Iter {iterations}: ||delta_x|| = {delta_norm:.6e}, ||AX - B|| = {current_error_norm:.6e}")

        if delta_norm <= eps_0 and current_error_norm <= eps_0:
            break
        if iterations >= 100:
            print("Iteration limit reached.")
            break

    final_deviation = vector_norm([x_current[i] - ideal_x[i] for i in range(n)])
    print(f"\nIterations to reach {eps_0}: {iterations}")
    print(f"Final deviation from exact X = 2.5: {final_deviation:.6e}")

    plot_errors = [max(e, 1e-16) for e in error_history]

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(plot_errors)), plot_errors, "o-", color="#1f77b4", linewidth=2, markersize=7, label="Residual norm $||AX - B||$")
    plt.axhline(eps_0, color="red", linestyle="--", linewidth=1.2, label=f"Target accuracy $\\varepsilon_0 = {eps_0:.0e}$")
    plt.yscale("log")
    plt.title("Iterative Refinement Convergence (LU Decomposition, $n = 100$)")
    plt.xlabel("Iteration number")
    plt.ylabel("Residual norm (log scale)")
    plt.xticks(range(len(plot_errors)))
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lab6_plot_convergence.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()