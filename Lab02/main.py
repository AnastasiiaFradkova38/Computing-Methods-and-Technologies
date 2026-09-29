import csv
import math
import numpy as np
import matplotlib.pyplot as plt

def read_data(filename: str) -> tuple[list[float], list[float]]:
    n: list[float] = []
    t: list[float] = []

    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            n.append(float(row['n']))
            t.append(float(row['t']))

    return (n, t)

def omega(k: int, x: float, x_values: list[float]) -> float:
    if k > len(x_values) - 1:
        raise Exception('K cannot be bigger than N.')
    
    value: float = 1
    for i in range(k + 1):
        value *= (x - x_values[i])
    return value

def calculate_divided_difference(k: int, x_values: list[float], y_values: list[float]) -> float:
    divided_difference: float = 0
    for i in range(k + 1):
        denominator: float = 1
        for j in range(k + 1):
            if j != i:
                denominator *= (x_values[i] - x_values[j])
        divided_difference += y_values[i] / denominator
    return divided_difference

def newton_interpolation(x: float, x_nodes: list[float], y_nodes: list[float]) -> float:
    sum_val: float = 0
    for k in range(1, len(x_nodes)):
        sum_val += omega(k - 1, x, x_nodes) * calculate_divided_difference(k, x_nodes, y_nodes)
    return y_nodes[0] + sum_val

def calculate_finite_differences(y_values: list[float]) -> list[float]:
    n: int = len(y_values)
    diff_table: list[list[float]] = [y_values.copy()]
    
    for k in range(1, n):
        current_diff: list[float] = []
        for i in range(n - k):
            diff: float = diff_table[k - 1][i + 1] - diff_table[k - 1][i]
            current_diff.append(diff)
        diff_table.append(current_diff)
        
    finite_differences: list[float] = []
    for k in range(n):
        finite_differences.append(diff_table[k][0])
        
    return finite_differences

def factorial_polynomial(t: float, k: int) -> float:
    if k == 0:
        return 1
    
    value: float = 1
    for i in range(k):
        value *= (t - i)
    return value

def factorial_interpolation(x: float, x_nodes: list[float], y_nodes: list[float]) -> float:
    h: float = x_nodes[1] - x_nodes[0]
    t: float = (x - x_nodes[0]) / h
    
    finite_differences: list[float] = calculate_finite_differences(y_nodes)
    
    sum_val: float = 0
    for k in range(len(x_nodes)):
        term: float = (finite_differences[k] / math.factorial(k)) * factorial_polynomial(t, k)
        sum_val += term
        
    return sum_val

def calculate_error(x: float, y: float, x_nodes: list[float], y_nodes: list[float]) -> float:
    return abs(y - newton_interpolation(x, x_nodes, y_nodes))

def visualise_results(x_nodes: list[float], y_nodes: list[float]) -> None:
    x_dense = np.linspace(min(x_nodes), max(x_nodes), 500)
    y_dense = [newton_interpolation(x, x_nodes, y_nodes) for x in x_dense]
    
    n = len(x_nodes)
    w_dense = [omega(n - 1, x, x_nodes) for x in x_dense]

    plt.figure(1, figsize=(10, 6))
    plt.plot(x_dense, y_dense, label='Newton polynomial', color='blue', linewidth=2)
    plt.scatter(x_nodes, y_nodes, color='red', s=50, label='Nodes', zorder=5)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

    plt.figure(2, figsize=(10, 6))
    plt.plot(x_dense, w_dense, label=f'omega(x) function', color='purple')
    
    plt.xlabel('x')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.grid(True)

    plt.show()

def main() -> None:
    x_nodes, y_nodes = read_data('data.csv')
    visualise_results(x_nodes, y_nodes)

if __name__ == '__main__':
    main()