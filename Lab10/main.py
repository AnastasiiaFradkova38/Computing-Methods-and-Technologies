import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return -2.0 * y + 4.0 * x


def exact_y(x):
    return 2.0 * np.exp(-2.0 * x) + 2.0 * x - 1.0


def rk4_step(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2.0, y + h * k1 / 2.0)
    k3 = f(x + h / 2.0, y + h * k2 / 2.0)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def adams2_fixed(f, a, b, y0, h, eps_iter=1e-10):
    xs = [a]
    ys = [y0]
    err_est = [0.0]

    x1 = a + h
    y1 = rk4_step(f, a, y0, h)
    xs.append(x1)
    ys.append(y1)
    err_est.append(0.0)

    n_steps = int(round((b - a) / h))
    for n in range(1, n_steps):
        xn = xs[n]
        yn = ys[n]
        fn = f(xn, yn)
        fn_1 = f(xs[n - 1], ys[n - 1])

        x_next = a + (n + 1) * h
        y_pred = yn + (h / 2.0) * (3.0 * fn - fn_1)

        y_corr = y_pred
        for _ in range(10):
            y_new = yn + (h / 2.0) * (f(x_next, y_corr) + fn)
            if abs(y_new - y_corr) <= eps_iter:
                y_corr = y_new
                break
            y_corr = y_new

        xs.append(x_next)
        ys.append(y_corr)
        err_est.append(-1.0 / 6.0 * (y_corr - y_pred))

    return np.array(xs), np.array(ys), np.array(err_est)


def adams2_adaptive(f, a, b, y0, h0, eps=1e-5):
    xs = [a]
    ys = [y0]
    hs = [h0]

    x = a
    y = y0
    h = h0
    k_factor = 8.0

    x_prev = x
    y_prev = y
    y = rk4_step(f, x, y, h)
    x += h
    xs.append(x)
    ys.append(y)
    hs.append(h)

    while x < b - 1e-12:
        if x + h > b:
            h = b - x
            y = rk4_step(f, x, y, h)
            x = b
            xs.append(x)
            ys.append(y)
            hs.append(h)
            break

        fn = f(x, y)
        fn_1 = f(x_prev, y_prev)
        x_next = x + h
        y_pred = y + (h / 2.0) * (3.0 * fn - fn_1)

        y_corr = y_pred
        for _ in range(10):
            y_new = y + (h / 2.0) * (f(x_next, y_corr) + fn)
            if abs(y_new - y_corr) <= 1e-12:
                y_corr = y_new
                break
            y_corr = y_new

        err = abs(-1.0 / 6.0 * (y_corr - y_pred))

        if err > eps:
            h /= 2.0
            x_prev = x
            y_prev = y
            y = rk4_step(f, x, y, h)
            x += h
            xs.append(x)
            ys.append(y)
            hs.append(h)
        else:
            x_prev = x
            y_prev = y
            x = x_next
            y = y_corr
            xs.append(x)
            ys.append(y)
            hs.append(h)
            if err < eps / k_factor:
                h *= 2.0
                if x + h <= b:
                    x_prev = x
                    y_prev = y
                    y = rk4_step(f, x, y, h)
                    x += h
                    xs.append(x)
                    ys.append(y)
                    hs.append(h)

    return np.array(xs), np.array(ys), np.array(hs)


def rk4_fixed(f, a, b, y0, h):
    n_steps = int(round((b - a) / h))
    xs = np.linspace(a, b, n_steps + 1)
    ys = np.zeros(n_steps + 1)
    ys[0] = y0
    for n in range(n_steps):
        ys[n + 1] = rk4_step(f, xs[n], ys[n], h)
    return xs, ys


def rk4_runge_error(f, a, b, y0, h):
    xs, ys_h = rk4_fixed(f, a, b, y0, h)
    _, ys_h2 = rk4_fixed(f, a, b, y0, h / 2.0)
    runge_err = (16.0 / 15.0) * (ys_h2[::2] - ys_h)
    return xs, runge_err


def rk4_adaptive(f, a, b, y0, h0, eps=1e-6):
    xs = [a]
    ys = [y0]
    hs = [h0]

    x = a
    y = y0
    h = h0
    k_factor = 32.0

    while x < b - 1e-12:
        if x + h > b:
            h = b - x
        y_full = rk4_step(f, x, y, h)
        y_half = rk4_step(f, x, y, h / 2.0)
        y_double = rk4_step(f, x + h / 2.0, y_half, h / 2.0)

        err = (16.0 / 15.0) * abs(y_full - y_double)

        if err > eps:
            h /= 2.0
        else:
            x += h
            y = y_double
            xs.append(x)
            ys.append(y)
            hs.append(h)
            if err < eps / k_factor:
                h *= 2.0

    return np.array(xs), np.array(ys), np.array(hs)


a, b, y0 = 0.0, 3.0, 1.0
h_fixed = 0.01

xa, ya, ea_est = adams2_fixed(f, a, b, y0, h_fixed)
ea_exact = ya - exact_y(xa)

xa_ad, ya_ad, ha_ad = adams2_adaptive(f, a, b, y0, h0=0.1, eps=1e-5)

print("=== PART 1: ADAMS 2ND ORDER PREDICTOR-CORRECTOR ===")
print(f"Fixed step h = {h_fixed}:")
print(f"  Max exact error:     {np.max(np.abs(ea_exact)):.3e}")
print(f"  Max estimated error: {np.max(np.abs(ea_est)):.3e}")
print(f"Adaptive step (eps = 1e-5):")
print(f"  Total steps:         {len(xa_ad) - 1}")
print(f"  Min/Max step h:      {np.min(ha_ad):.5f} / {np.max(ha_ad):.5f}")
print(f"  Max exact error:     {np.max(np.abs(ya_ad - exact_y(xa_ad))):.3e}\n")

xr, yr = rk4_fixed(f, a, b, y0, h_fixed)
er_exact = yr - exact_y(xr)
_, er_runge = rk4_runge_error(f, a, b, y0, h_fixed)

xr_ad, yr_ad, hr_ad = rk4_adaptive(f, a, b, y0, h0=0.1, eps=1e-6)

print("=== PART 2: RUNGE-KUTTA 4TH ORDER ===")
print(f"Fixed step h = {h_fixed}:")
print(f"  Max exact error:     {np.max(np.abs(er_exact)):.3e}")
print(f"  Max Runge error:     {np.max(np.abs(er_runge)):.3e}")
print("Step size study:")
for h_test in [0.1, 0.05, 0.01, 0.005]:
    xt, yt = rk4_fixed(f, a, b, y0, h_test)
    print(f"  h = {h_test:<6} -> Max exact error = {np.max(np.abs(yt - exact_y(xt))):.3e}")
print(f"Adaptive step (eps = 1e-6):")
print(f"  Total steps:         {len(xr_ad) - 1}")
print(f"  Min/Max step h:      {np.min(hr_ad):.5f} / {np.max(hr_ad):.5f}")
print(f"  Max exact error:     {np.max(np.abs(yr_ad - exact_y(xr_ad))):.3e}")


plt.figure(figsize=(7, 5))
plt.plot(xa, ea_exact, 'b-', linewidth=1.8)
plt.title("Adams 2nd Order: Exact Local Error ($y_n - y(x_n)$, $h = 0.01$)")
plt.xlabel("$x$")
plt.ylabel("Error $\\varphi_n$")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("lab10_plot1_adams_exact_err.png", dpi=300)

plt.figure(figsize=(7, 5))
plt.plot(xa, ea_est, 'g-', linewidth=1.8)
plt.title("Adams 2nd Order: Estimated Error $-\\frac{1}{6}(y_{n+1}^{cor} - y_{n+1}^{pr})$")
plt.xlabel("$x$")
plt.ylabel("Estimated Error")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("lab10_plot2_adams_est_err.png", dpi=300)

plt.figure(figsize=(7, 5))
plt.step(xa_ad, ha_ad, where='post', color='purple', linewidth=1.8)
plt.title("Adams 2nd Order: Adaptive Step Size $h(x)$ ($\\varepsilon = 10^{-5}$)")
plt.xlabel("$x$")
plt.ylabel("Step size $h$")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("lab10_plot3_adams_step.png", dpi=300)

plt.figure(figsize=(7, 5))
plt.plot(xr, er_exact, 'r-', linewidth=1.8)
plt.title("Runge-Kutta 4th Order: Exact Local Error ($y_n - y(x_n)$, $h = 0.01$)")
plt.xlabel("$x$")
plt.ylabel("Error $\\varphi_n$")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("lab10_plot4_rk4_exact_err.png", dpi=300)

plt.figure(figsize=(7, 5))
plt.plot(xr, er_runge, 'm-', linewidth=1.8)
plt.title("Runge-Kutta 4th Order: Runge Error Estimate ($h = 0.01$)")
plt.xlabel("$x$")
plt.ylabel("Runge Error")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("lab10_plot5_rk4_runge_err.png", dpi=300)

plt.figure(figsize=(7, 5))
plt.step(xr_ad, hr_ad, where='post', color='darkorange', linewidth=1.8)
plt.title("Runge-Kutta 4th Order: Adaptive Step Size $h(x)$ ($\\varepsilon = 10^{-6}$)")
plt.xlabel("$x$")
plt.ylabel("Step size $h$")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("lab10_plot6_rk4_step.png", dpi=300)

plt.show()