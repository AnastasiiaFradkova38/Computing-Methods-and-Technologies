import csv
import math
import sys
import matplotlib.pyplot as plt

def read_data(filename) -> tuple[list[float], list[float]]:
    x: list[float] = []
    y: list[float] = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    
    return (x, y)

def approximating_function(x: list[float], a: list[float]) -> list[float]:
    y: list[float] = [0] * len(x)
    for i in range(len(x)):
        y[i] = (sum(a[j] * (x[i]**j) for j in range(len(a))))
    return y

def form_matrix(x: list[float], n: int) -> list[list[float]]:
    b: list[list[float]] = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(n + 1):
            b[i][j] = sum(math.pow(el, (i + j)) for el in x)
    return b

def form_vector(x: list[float], y: list[float], n: int) -> list[float]:
    c: list[float] = [0] * (n + 1)
    for i in range(n + 1):
        c[i] = sum(y[k] * math.pow(x[k], i) for k in range(len(x)))
    return c

def variance(y: list[float], y_approximate: list[float]) -> float:
    return sum((y[i] - y_approximate[i]) ** 2 for i in range(len(y))) / len(y)

def gauss_method(b: list[list[float]], c: list[float]) -> list[float]:
    n: int = len(c)
    for k in range(n):
        max_row = k
        max_value = abs(b[k][k])
        for i in range(k + 1, n):
            if abs(b[i][k]) > max_value:
                max_value = abs(b[i][k])
                max_row = i

        b[k], b[max_row] = b[max_row], b[k]
        c[k], c[max_row] = c[max_row], c[k]
        
        for i in range(k + 1, n):
            if b[k][k] == 0:
                continue
            factor = b[i][k] / b[k][k]
            for j in range(k, n):
                b[i][j] -= factor * b[k][j]
            c[i] -= factor * c[k]
    
    a: list[float] = [0] * n
    for i in range(n - 1, -1, -1):
        s = sum(b[i][j] * a[j] for j in range(i + 1, n))
        a[i] = (c[i] - s) / b[i][i]
        
    return a

def build_plots(x: list[float], y: list[float], y_approximate: list[float], error: list[float], optimal_m: int, x_future: list[float], y_future: list[float], m_values: list[int], variances: list[float]) -> None:
    plt.figure(1, figsize=(10, 6))
    plt.plot(x, y, 'o', label='Real temperatures')
    plt.plot(x, y_approximate, 'r-', label=f'Approximation (m={optimal_m})')
    plt.plot(x_future, y_future, 'go-', label='Prediction')
    plt.xlabel('Month')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True)

    plt.figure(2, figsize=(10, 6))
    plt.plot(x, error, 'm-')
    plt.xlabel('Month')
    plt.ylabel('Error')
    plt.grid(True)

    plt.figure(3, figsize=(10, 6))
    plt.plot(m_values, variances, 'b-', label='Variance')
    plt.plot(optimal_m, variances[optimal_m - 1], 'ro', label=f'Minimum (m={optimal_m})')
    plt.xlabel('Polynomial Degree (m)')
    plt.ylabel('Variance')
    plt.xticks(m_values) 
    plt.legend()
    plt.grid(True)

    plt.show()

def main() -> None:
    x, y = read_data('data.csv')
    
    max_m: int = 10

    if max_m >= len(y):
        print("M cannot be less than the number of nodes.")
        return

    variances: list[float] = []
    optimal_m: int = 0
    min_variance: float = sys.float_info.max
    m_values: list[int] = list(range(1, max_m + 1))

    for m in m_values:
        b: list[list[float]] = form_matrix(x, m)
        c: list[float] = form_vector(x, y, m)
        a: list[float] = gauss_method(b, c)
        y_approximate: list[float] = approximating_function(x, a)
        var: float = variance(y, y_approximate)
        
        variances.append(var)
        if var < min_variance:
            min_variance = var
            optimal_m = m

    b: list[list[float]] = form_matrix(x, optimal_m)
    c: list[float] = form_vector(x, y, optimal_m)
    a: list[float] = gauss_method(b, c)
    y_approximate: list[float] = approximating_function(x, a)

    x_future_size: int = 3
    x_future: list[float] = [x[-1] + i for i in range(1, x_future_size + 1)]
    y_future: list[float] = approximating_function(x_future, a)
    error: list[float] = [abs(y[i] - y_approximate[i]) for i in range(len(y))]

    build_plots(x, y, y_approximate, error, optimal_m, x_future, y_future, m_values, variances)

if __name__ == '__main__':
    main()