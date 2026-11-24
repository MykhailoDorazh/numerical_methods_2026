"""
Лабораторна робота №4. Точність формул для чисельного диференціювання.
Метод Рунге-Ромберга. Метод Ейткена.

Вологість ґрунту M(t) = 50 e^(-0.1 t) + 5 sin(t), точка t0 = 1:
- точне значення похідної;
- залежність похибки центральної різниці від кроку h (10^-20 ... 10^3), оптимальний крок;
- уточнення за методом Рунге-Ромберга (кроки h, 2h);
- уточнення та оцінка порядку точності за методом Ейткена (кроки h, 2h, 4h).
"""

import math
import os

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
RES_FILE = os.path.join(BASE_DIR, "results.txt")

X0 = 1.0
H = 1e-3


def f(t):
    """Вологість ґрунту M(t)."""
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)


def df_exact(t):
    """Аналітична похідна M'(t) = -5 e^(-0.1 t) + 5 cos(t)."""
    return -5 * math.exp(-0.1 * t) + 5 * math.cos(t)


def central_diff(func, x0, h):
    """y'_0(h) = (f(x0 + h) - f(x0 - h)) / (2h) — другий порядок точності."""
    return (func(x0 + h) - func(x0 - h)) / (2 * h)


def runge_romberg(d_h, d_2h, p=2, q=2):
    """y'_R = y'(h) + (y'(h) - y'(qh)) / (q^p - 1)."""
    return d_h + (d_h - d_2h) / (q ** p - 1)


def aitken(d_h, d_2h, d_4h, q=2):
    """Уточнене значення y'_E та порядок точності p за трьома кроками h, qh, q^2 h."""
    y_e = (d_2h ** 2 - d_4h * d_h) / (2 * d_2h - (d_4h + d_h))
    p = math.log(abs((d_4h - d_2h) / (d_2h - d_h))) / math.log(q)
    return y_e, p


def step_scan(func, x0, exact):
    """Похибка R(h) для h = k * 10^m, k = 1..9, m = -20..2, а також h = 10^3."""
    hs = [k * 10.0 ** m for m in range(-20, 3) for k in range(1, 10)] + [1e3]
    errs = [abs(central_diff(func, x0, h) - exact) for h in hs]
    return np.array(hs), np.array(errs)


def save_fig(fig, name):
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(os.path.join(IMG_DIR, name), dpi=120, bbox_inches="tight")


def main():
    plt.rcParams["axes.grid"] = True
    out = []

    def log(s=""):
        print(s)
        out.append(s)

    # ---- 1. Точне значення похідної ----
    exact = df_exact(X0)
    log(f"M'(t) = -5 e^(-0.1 t) + 5 cos(t)")
    log(f"Точне значення y'({X0:g}) = {exact:.15f}")

    # ---- 2. Залежність похибки від кроку ----
    hs, errs = step_scan(f, X0, exact)
    log("\n      h      |   y'_0(h)          |   R(h)")
    for m in range(-20, 4):
        h = 10.0 ** m
        d = central_diff(f, X0, h)
        log(f" {h:10.0e} | {d:18.12g} | {abs(d - exact):.3e}")
    i_opt = int(np.argmin(errs))
    h0, r0 = hs[i_opt], errs[i_opt]
    log(f"\nОптимальний крок h0 = {h0:.0e}, досягнута точність R0 = {r0:.3e}")

    # теоретична оцінка оптимального кроку: R = h^2 |f'''| / 6 + Δ / h
    d3 = abs(-0.05 * math.exp(-0.1 * X0) - 5 * math.cos(X0))     # |M'''(t0)|
    delta = abs(f(X0)) * np.finfo(float).eps
    psi = d3 / 6
    h_theor = (delta / (2 * psi)) ** (1 / 3)
    r_theor = psi * h_theor ** 2 + delta / h_theor
    log(f"Теоретична оцінка: |M'''(t0)| = {d3:.6f}, Δ = {delta:.2e}, "
        f"h_opt = (Δ / 2ψ)^(1/3) = {h_theor:.2e}, R_min ≈ {r_theor:.2e}")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.loglog(hs, np.maximum(errs, 1e-18), ".-", lw=1)
    ax.loglog([h0], [r0], "r*", ms=14, label=f"h₀ = {h0:.0e}, R₀ = {r0:.1e}")
    hh = np.logspace(-20, 3, 200)
    ax.loglog(hh, psi * hh ** 2, "--", color="gray", label="похибка методу ψh²")
    ax.loglog(hh, delta / hh, ":", color="gray", label="похибка округлення Δ/h")
    ax.set_ylim(1e-12, 1e4)
    ax.set_xlabel("h")
    ax.set_ylabel("R(h) = |y'₀(h) − y'(x₀)|")
    ax.set_title("Залежність похибки центральної різниці від кроку")
    ax.legend()
    save_fig(fig, "error_vs_h.png")

    # ---- 3–5. Два кроки h, 2h ----
    d_h = central_diff(f, X0, H)
    d_2h = central_diff(f, X0, 2 * H)
    d_4h = central_diff(f, X0, 4 * H)
    r1 = abs(d_h - exact)
    log(f"\nh = {H:g}")
    log(f"y'_0(h)  = {d_h:.15f}   похибка {abs(d_h - exact):.3e}")
    log(f"y'_0(2h) = {d_2h:.15f}   похибка {abs(d_2h - exact):.3e}")
    log(f"y'_0(4h) = {d_4h:.15f}   похибка {abs(d_4h - exact):.3e}")
    log(f"R1 = |y'_0(h) - y'(x0)| = {r1:.3e}")

    # ---- 6. Рунге-Ромберг ----
    y_r = runge_romberg(d_h, d_2h)
    r2 = abs(y_r - exact)
    log(f"\nРунге-Ромберг: y'_R = {y_r:.15f}, R2 = {r2:.3e}, R1 / R2 = {r1 / r2:.1f}")

    # ---- 7. Ейткен ----
    y_e, p = aitken(d_h, d_2h, d_4h)
    r3 = abs(y_e - exact)
    log(f"Ейткен: y'_E = {y_e:.15f}, p = {p:.6f}, R3 = {r3:.3e}, R1 / R3 = {r1 / r3:.1f}")

    # ---- Характер зміни похибки при різних h ----
    log("\n    h    |     R1      |     R2 (РР)  |    R3 (Ейткен) |   p")
    rows = []
    for h in [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]:
        a, b, c = (central_diff(f, X0, k * h) for k in (1, 2, 4))
        rr = runge_romberg(a, b)
        ye, pp = aitken(a, b, c)
        rows.append((h, abs(a - exact), abs(rr - exact), abs(ye - exact)))
        log(f" {h:7.0e} | {rows[-1][1]:.3e} | {rows[-1][2]:.3e} | {rows[-1][3]:.3e} | {pp:.4f}")
    rows = np.array(rows)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.loglog(rows[:, 0], rows[:, 1], "o-", label="R1 — центральна різниця, O(h²)")
    ax.loglog(rows[:, 0], rows[:, 2], "s-", label="R2 — Рунге-Ромберг, O(h⁴)")
    ax.loglog(rows[:, 0], np.maximum(rows[:, 3], 1e-17), "^-", label="R3 — Ейткен")
    ax.axvline(H, color="gray", ls="--")
    ax.set_xlabel("h")
    ax.set_ylabel("Похибка")
    ax.set_title("Похибка чисельного диференціювання та уточнених значень")
    ax.legend()
    save_fig(fig, "methods.png")

    with open(RES_FILE, "w", encoding="utf-8") as file:
        file.write("\n".join(out) + "\n")
    plt.show()


if __name__ == "__main__":
    main()
