import math
import matplotlib.pyplot as plt

def f(x: float) -> float:
    return 50 * (math.exp(-0.1 * x)) + 5 * math.sin(x)

def f_derivative(x: float) -> float:
    return -5 * (math.exp(-0.1 * x)) + 5 * math.cos(x)

def central_difference(x: float, h: float) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)

def draw_plot(h_values: list[float], errors: list[float]) -> None:
    plt.plot(h_values, errors, marker="o")
    plt.xlabel("H")
    plt.ylabel("Error")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

def calculate_errors(x0: float, f_x0: float, h_values: list[float]) -> list[float]:
    return [abs(f_x0 - central_difference(x0, h)) for h in h_values]

def find_h_optimal(h_values: list[float], errors: list[float]) -> tuple[float, float]:
    min_error, h_optimal = min(zip(errors, h_values))
    return h_optimal, min_error

def runge_romberg_method(f_x0: float, f0_h: float, f0_2h: float) -> float:
    f_r: float = f0_h + ((f0_h - f0_2h) / 3)
    return abs(f_r - f_x0)

def aitkens_method(f_x0: float, f0_h: float, f0_2h: float, f0_4h: float) -> float:
    denominator: float = 2 * f0_2h - (f0_4h + f0_h)
    if denominator == 0.0:
        return float('inf')
    f_e: float = (f0_2h ** 2 - f0_h * f0_4h) / denominator
    return abs(f_e - f_x0)

def main() -> None:
    x0: float = 1
    f_x0: float = f_derivative(x0)

    h_values = [10 ** x for x in range(-20, 4)]
    errors: list[float] = calculate_errors(x0, f_x0, h_values)

    draw_plot(h_values, errors)

    h_optimal, error = find_h_optimal(h_values, errors)

    print(f"Optimal h: {h_optimal}; Error: {error}\n")

    h: float = 1e-3
    f0_h: float = central_difference(x0, h)
    f0_2h: float = central_difference(x0, h * 2)
    f0_4h: float = central_difference(x0, h * 4)

    r1: float = abs(f0_h - f_x0)
    r2: float = runge_romberg_method(f_x0, f0_h, f0_2h)
    r3: float = aitkens_method(f_x0, f0_h, f0_2h, f0_4h)

    print(f"Error: {r1}")
    print(f"Runge Romberg method: {r2}")
    print(f"Aiken's method: {r3}")

if __name__ == "__main__":
    main()