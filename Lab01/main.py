import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
import csv

def thomas_algorithm(alpha: list[float], beta: list[float], gamma: list[float], delta: list[float]):
    n = len(alpha)
    a: list[float] = [None] * n
    b: list[float] = [None] * n
    x: list[float] = [None] * n

    a[0] = -gamma[0] / beta[0]
    b[0] = delta[0] / beta[0]

    for i in range(1, n):
        a[i] = -gamma[i] / (alpha[i] * a[i - 1] + beta[i])
        b[i] = (delta[i] - alpha[i] * b[i - 1]) / (beta[i] + alpha[i] * a[i - 1])

    x[n - 1] = (delta[n - 1] - alpha[n - 1] * b[n - 2]) / (alpha[n - 1] * a[n - 2] + beta[n - 1])
    for i in range(n - 2, -1, -1):
        x[i] = (a[i] * x[i + 1] + b[i])

    return x

def analyze_error(f: Callable[[float], float], a: list[float], b: list[float], c: list[float], d: list[float], nodes_x: list[float], nodes_y: list[float]):
    n = len(a)
    N_tab = 20 * n
    h_fine = (nodes_x[-1] - nodes_x[0]) / N_tab

    x_fine: list[float] = []
    y_approx: list[float] = []
    y_exact: list[float] = []
    errors: list[float] = []

    for i in range(N_tab + 1):
        x_val = nodes_x[0] + i * h_fine
        x_fine.append(x_val)

        j_idx = n - 1
        for j in range(n):
            if nodes_x[j] <= x_val <= nodes_x[j + 1]:
                j_idx = j
                break
        
        dx = x_val - nodes_x[j_idx]
        s_val = a[j_idx] + b[j_idx] * dx + c[j_idx] * (dx**2) + d[j_idx] * (dx**3)
        y_approx.append(s_val)

        exact_val = f(x_val)
        y_exact.append(exact_val)

        errors.append(abs(exact_val - s_val))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    ax1.plot(x_fine, y_exact, color='green', label='Exact function')
    ax1.plot(x_fine, y_approx, '--', color='#66A3BF', label='Spline')
    ax1.plot(nodes_x, nodes_y, 'o', color='#3368A0', label='Nodes')
    ax1.set_title('Function and its cubic spline')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(x_fine, errors, color='red', label='Error |f(x) - S(x)|')
    ax2.set_title('Interpolation error')
    ax2.set_xlabel('x')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def tabulate_data(f: Callable[[float], float], x_start: float, x_end: float, n: int) -> tuple[list[float], list[float]]:
    step: float = (x_end - x_start) / n
    nodes_x: list[float] = [None] * n
    nodes_y: list[float] = [None] * n

    for i in range(n):
        x = x_start + i * step
        nodes_x[i] = x
        nodes_y[i] = f(x)

    with open('data.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y'])
        writer.writerows(zip(nodes_x, nodes_y))

    return (nodes_x, nodes_y)

def main() -> None:
    def f(x: float) -> float:
        return math.exp(-0.5 * x) * math.sin(2 * x)

    x_start: float = 0
    x_end: float = 10
    node_number: int = 5
    gap_number: int = node_number - 1

    nodes_x, nodes_y = tabulate_data(f, x_start, x_end, node_number)

    h: list[float] = [nodes_x[i] - nodes_x[i - 1] for i in range(1, node_number)]

    alpha: list[float] = [0] + [None] * (gap_number - 1)
    beta: list[float] = [1] + [None] * (gap_number - 1)
    gamma: list[float] = [0] + [None] * (gap_number - 1)
    delta: list[float] = [0] + [None] * (gap_number - 1)

    for i in range(1, gap_number - 1):
        alpha[i] = h[i - 1]
        beta[i] = 2 * (h[i - 1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3 * ((nodes_y[i + 1] - nodes_y[i]) / h[i] - (nodes_y[i] - nodes_y[i - 1]) / h[i - 1])

    n = gap_number - 1
    alpha[n] = h[n - 1]
    beta[n] = 2 * (h[n - 1] + h[n])
    gamma[n] = 0
    delta[n] = 3 * ((nodes_y[n + 1] - nodes_y[n]) / h[n] - (nodes_y[n] - nodes_y[n - 1]) / h[n - 1])

    a: list[float] = [None] * (node_number - 1)
    b: list[float] = [None] * (node_number - 1)
    c: list[float] = thomas_algorithm(alpha, beta, gamma, delta)
    d: list[float] = [None] * (node_number - 1)

    for i in range(gap_number - 1):
        a[i] = nodes_y[i]
        b[i] = (nodes_y[i + 1] - nodes_y[i]) / h[i] - h[i] / 3 * (c[i + 1] + 2 * c[i])
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    a[gap_number - 1] = nodes_y[gap_number - 1]
    b[gap_number - 1] = (nodes_y[gap_number] - nodes_y[gap_number - 1]) / h[gap_number - 1] - 2/3 * h[gap_number - 1] * c[gap_number - 1]
    d[gap_number - 1] = -c[gap_number - 1] / (3 * h[gap_number - 1])

    analyze_error(f, a, b, c, d, nodes_x, nodes_y)

if __name__ == '__main__':
    main()