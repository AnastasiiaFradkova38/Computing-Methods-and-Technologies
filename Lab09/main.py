import numpy as np
import matplotlib.pyplot as plt


def exploratory_search(f, x_base, delta, q, eps1, allow_shrink=True):
    x_curr = x_base.copy()
    delta_curr = delta.copy()
    n = len(x_curr)

    for i in range(n):
        while True:
            f_curr = f(x_curr)
            
            x_try = x_curr.copy()
            x_try[i] += delta_curr[i]
            if f(x_try) < f_curr:
                x_curr = x_try
                break
            
            x_try = x_curr.copy()
            x_try[i] -= delta_curr[i]
            if f(x_try) < f_curr:
                x_curr = x_try
                break
            
            if not allow_shrink:
                break
            
            delta_curr[i] /= q
            if delta_curr[i] < eps1:
                break

    return x_curr, delta_curr


def hooke_jeeves(f, x0, delta0, q=2.0, p=2.0, eps1=1e-6, eps2=1e-6, max_iter=10000):
    x0 = np.array(x0, dtype=float)
    delta = np.array(delta0, dtype=float)
    
    trajectory = [x0.copy()]
    iterations = 0

    while iterations < max_iter:
        iterations += 1
        
        x1, delta = exploratory_search(f, x0, delta, q, eps1, allow_shrink=True)
        
        if np.array_equal(x1, x0):
            break
            
        trajectory.append(x1.copy())
        
        if np.linalg.norm(delta) < eps1 and abs(f(x1) - f(x0)) < eps2:
            x0 = x1
            break

        while iterations < max_iter:
            iterations += 1
            
            xp = x1 + p * (x1 - x0)
            
            x2, _ = exploratory_search(f, xp, delta, q, eps1, allow_shrink=False)
            
            if f(x2) < f(x1):
                x0 = x1
                x1 = x2
                trajectory.append(x1.copy())
            else:
                x0 = x1
                break

    return x0, f(x0), np.array(trajectory)


def rosenbrock(x):
    return 100.0 * (x[0]**2 - x[1])**2 + (x[0] - 1.0)**2

x0_rosen = [-1.2, 0.0]
delta_rosen = [0.5, 0.5]

res_x, res_f, traj_rosen = hooke_jeeves(
    rosenbrock, x0_rosen, delta_rosen, q=2.0, p=1.0, eps1=1e-6, eps2=1e-6
)

print("=== TEST: Rosenbrock Function ===")
print(f"Initial point:    {x0_rosen}")
print(f"Found minimum:    x1 = {res_x[0]:.6f}, x2 = {res_x[1]:.6f}")
print(f"Function value:   f(X*) = {res_f:.2e}")
print(f"Trajectory steps: {len(traj_rosen) - 1}\n")


def f1(x1, x2):
    return x1**2 + x2**2 - 4.0

def f2(x1, x2):
    return x2 - x1**2 + 1.0

def target_phi(x):
    return f1(x[0], x[1])**2 + f2(x[0], x[1])**2

x0_sys = [2.0, 2.0]
delta_sys = [0.5, 0.5]

sol_x, sol_phi, traj_sys = hooke_jeeves(
    target_phi, x0_sys, delta_sys, q=2.0, p=2.0, eps1=1e-6, eps2=1e-6
)

print("=== NONLINEAR SYSTEM SOLUTION ===")
print(f"Initial approximation: {x0_sys}")
print(f"Found solution:        x1 = {sol_x[0]:.6f}, x2 = {sol_x[1]:.6f}")
print(f"Value of Phi(X*):      {sol_phi:.2e}")
print(f"Residual f1(X*):       {f1(sol_x[0], sol_x[1]):.2e}")
print(f"Residual f2(X*):       {f2(sol_x[0], sol_x[1]):.2e}")
print(f"Descent steps count:   {len(traj_sys) - 1}")

filename = "trajectory_system.txt"
with open(filename, "w", encoding="utf-8") as file:
    file.write("Step\tx1\t\tx2\t\tPhi(X)\n")
    for idx, pt in enumerate(traj_sys):
        file.write(f"{idx}\t{pt[0]:.6f}\t{pt[1]:.6f}\t{target_phi(pt):.6e}\n")
print(f"Trajectory successfully saved to '{filename}'.")


x_grid = np.linspace(-3, 3, 400)
y_grid = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_grid, y_grid)

plt.figure(figsize=(7, 6))
plt.contour(X, Y, f1(X, Y), levels=[0], colors='blue', linewidths=2)
plt.contour(X, Y, f2(X, Y), levels=[0], colors='green', linewidths=2)
plt.plot(x0_sys[0], x0_sys[1], 'ko', label='Initial guess $X^{(0)}$')
plt.plot(sol_x[0], sol_x[1], 'r*', markersize=12, label='Found root $X^*$')
plt.title("System of Nonlinear Equations\n(blue: $x_1^2+x_2^2=4$, green: $x_2=x_1^2-1$)")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.legend()
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig("lab9_plot1_system.png", dpi=300)

plt.figure(figsize=(7, 6))
Z = f1(X, Y)**2 + f2(X, Y)**2
levels = np.logspace(-2, 2.5, 25)

plt.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.8)
plt.plot(traj_sys[:, 0], traj_sys[:, 1], 'r.-', linewidth=1.5, markersize=6, label='Descent trajectory')
plt.plot(x0_sys[0], x0_sys[1], 'ko', label='Start $X^{(0)}$')
plt.plot(sol_x[0], sol_x[1], 'b*', markersize=12, label='Minimum $X^*$')
plt.title("Contour Lines of Objective Function $\\Phi(X)$\nand Hooke-Jeeves Descent Trajectory")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.xlim(0, 2.5)
plt.ylim(0, 2.5)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("lab9_plot2_trajectory.png", dpi=300)

plt.show()