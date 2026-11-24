import random
import matplotlib.pyplot as plt


def generate_matrix(filename: str, n: int = 100) -> None:
    matrix = [[random.uniform(-10.0, 10.0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        row_sum = sum(abs(val) for val in matrix[i])
        matrix[i][i] = row_sum + random.uniform(10.0, 50.0)
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


def vector_norm(v: list[float]) -> float:
    return max(abs(val) for val in v)


def matrix_norm(a: list[list[float]]) -> float:
    return max(sum(abs(val) for val in row) for row in a)


def solve_simple_iteration(
    a: list[list[float]], b: list[float], x0: list[float], eps: float
) -> tuple[list[float], int, list[float]]:
    n = len(a)
    tau = 1.0 / matrix_norm(a)
    x_current = x0[:]
    history = []
    iterations = 0

    while True:
        iterations += 1
        ax_current = multiply_matrix_vector(a, x_current)
        x_new = [x_current[i] - tau * (ax_current[i] - b[i]) for i in range(n)]
        current_error = vector_norm([x_new[i] - x_current[i] for i in range(n)])
        history.append(current_error)
        if current_error < eps or iterations >= 10000:
            break
        x_current = x_new[:]

    return x_new, iterations, history


def solve_jacobi(
    a: list[list[float]], b: list[float], x0: list[float], eps: float
) -> tuple[list[float], int, list[float]]:
    n = len(a)
    x_current = x0[:]
    history = []
    iterations = 0

    while True:
        iterations += 1
        x_new = [0.0] * n
        for i in range(n):
            sum_val = sum(a[i][j] * x_current[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum_val) / a[i][i]
        current_error = vector_norm([x_new[i] - x_current[i] for i in range(n)])
        history.append(current_error)
        if current_error < eps or iterations >= 10000:
            break
        x_current = x_new[:]

    return x_new, iterations, history


def solve_seidel(
    a: list[list[float]], b: list[float], x0: list[float], eps: float
) -> tuple[list[float], int, list[float]]:
    n = len(a)
    x_current = x0[:]
    history = []
    iterations = 0

    while True:
        iterations += 1
        max_diff = 0.0
        for i in range(n):
            sum1 = sum(a[i][j] * x_current[j] for j in range(i))
            sum2 = sum(a[i][j] * x_current[j] for j in range(i + 1, n))
            x_new_val = (b[i] - sum1 - sum2) / a[i][i]
            diff = abs(x_new_val - x_current[i])
            if diff > max_diff:
                max_diff = diff
            x_current[i] = x_new_val
        history.append(max_diff)
        if max_diff < eps or iterations >= 10000:
            break

    return x_current, iterations, history


def plot_convergence(
    hist_simp: list[float], hist_jac: list[float], hist_seid: list[float], eps: float
) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(hist_simp) + 1), hist_simp, label="Simple Iteration", linewidth=2)
    plt.plot(range(1, len(hist_jac) + 1), hist_jac, label="Jacobi Method", linewidth=2)
    plt.plot(range(1, len(hist_seid) + 1), hist_seid, label="Seidel Method", linewidth=2)
    plt.axhline(eps, color="red", linestyle="--", linewidth=1.2, label=f"Target accuracy $\\varepsilon_0 = {eps:.0e}$")
    plt.yscale("log")
    plt.xlabel("Iteration Number")
    plt.ylabel("Step Error Norm $||X^{(k+1)} - X^{(k)}||$ (Log Scale)")
    plt.title("Convergence Rate of Iterative Methods ($n = 100$)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("lab7_plot_convergence.png", dpi=300)
    plt.show()


def main() -> None:
    random.seed(42)
    n = 100
    matrix_file = "A.txt"
    vector_file = "B.txt"
    eps_0 = 1e-14

    generate_matrix(matrix_file, n)
    a = read_matrix_from_file(matrix_file)
    ideal_x = [2.5] * n
    b_calculated = multiply_matrix_vector(a, ideal_x)
    save_vector_to_file(vector_file, b_calculated)
    b = read_vector_from_file(vector_file)

    x0 = [1.0] * n

    print(f"=== ITERATIVE METHODS FOR SLAE (n = {n}, eps = {eps_0:.0e}) ===")

    for name, solver in [
        ("Simple Iteration", solve_simple_iteration),
        ("Jacobi Method   ", solve_jacobi),
        ("Seidel Method   ", solve_seidel),
    ]:
        x_sol, iters, hist = solver(a, b, x0, eps_0)
        ax = multiply_matrix_vector(a, x_sol)
        res_norm = vector_norm([ax[i] - b[i] for i in range(n)])
        dev_norm = vector_norm([x_sol[i] - ideal_x[i] for i in range(n)])
        print(f"{name}: {iters:3d} iterations | ||AX - B|| = {res_norm:.3e} | ||X - X*|| = {dev_norm:.3e}")
        if "Simple" in name:
            hist_simp = hist
        elif "Jacobi" in name:
            hist_jac = hist
        else:
            hist_seid = hist

    plot_convergence(hist_simp, hist_jac, hist_seid, eps_0)


if __name__ == "__main__":
    main()