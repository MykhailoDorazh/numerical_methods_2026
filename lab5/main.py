"""
Лабораторна робота №5. Складова квадратурна формула для чисельного інтегрування.
Методи підвищення точності. Адаптивний алгоритм.

Навантаження на сервер f(x) = 50 + 20 sin(pi x / 12) + 5 e^(-0.2 (x - 12)^2), x ∈ [0, 24]:
- точне значення інтегралу I0;
- складова формула Сімпсона, залежність похибки від N, N_opt для eps = 1e-12;
- уточнення при N0 за методами Рунге-Ромберга та Ейткена;
- адаптивний алгоритм на основі формули Сімпсона.
"""

import math
import os

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
RES_FILE = os.path.join(BASE_DIR, "results.txt")

A, B = 0.0, 24.0
EPS = 1e-12


def f(x):
    """Інтенсивність навантаження на сервер."""
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


def exact_integral(a, b):
    """Точне значення: I0 = 50(b - a) + (240/pi)(cos(pi a/12) - cos(pi b/12))
                            + 5/2 sqrt(pi/0.2) [erf(...) - erf(...)]."""
    k = math.sqrt(0.2)
    return (50 * (b - a)
            + 240 / math.pi * (math.cos(math.pi * a / 12) - math.cos(math.pi * b / 12))
            + 5 * math.sqrt(math.pi) / (2 * k)
            * (math.erf(k * (b - 12)) - math.erf(k * (a - 12))))


def simpson(func, a, b, n):
    """Складова формула Сімпсона (N парне):
    I(N) = h/3 (f0 + 4(f1 + f3 + ...) + 2(f2 + f4 + ...) + fN)."""
    if n % 2:
        raise ValueError("N має бути парним")
    h = (b - a) / n
    y = func(a + h * np.arange(n + 1))
    return h / 3 * (y[0] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum() + y[-1])


def runge_romberg(i_n, i_n2, p=4):
    """I_R = I(N0) + (I(N0) - I(N0/2)) / (2^p - 1)."""
    return i_n + (i_n - i_n2) / (2 ** p - 1)


def aitken(i1, i2, i3):
    """I_E та порядок p за значеннями I(N0), I(N0/2), I(N0/4)."""
    i_e = (i2 ** 2 - i1 * i3) / (2 * i2 - (i1 + i3))
    p = math.log(abs((i3 - i2) / (i2 - i1))) / math.log(2)
    return i_e, p


def adaptive_simpson(func, a, b, delta):
    """Адаптивний алгоритм: відрізок ділиться навпіл, поки |I(1) - I(2)| > delta_i,
    де delta_i = delta * (довжина відрізка) / (b - a).
    Повертає (I, кількість обчислень f, кількість відрізків)."""
    calls = [0]

    def fx(x):
        calls[0] += 1
        return float(func(x))

    def recurse(x0, x1, y0, ym, y1, whole, tol, depth):
        xm = (x0 + x1) / 2
        yl, yr = fx((x0 + xm) / 2), fx((xm + x1) / 2)
        h = x1 - x0
        left = h / 12 * (y0 + 4 * yl + ym)
        right = h / 12 * (ym + 4 * yr + y1)
        if abs(left + right - whole) <= tol or depth > 50:
            return left + right, 1
        l_val, l_n = recurse(x0, xm, y0, yl, ym, left, tol / 2, depth + 1)
        r_val, r_n = recurse(xm, x1, ym, yr, y1, right, tol / 2, depth + 1)
        return l_val + r_val, l_n + r_n

    y0, ym, y1 = fx(a), fx((a + b) / 2), fx(b)
    whole = (b - a) / 6 * (y0 + 4 * ym + y1)
    value, segments = recurse(a, b, y0, ym, y1, whole, delta, 0)
    return value, calls[0], segments


def save_fig(fig, name):
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(os.path.join(IMG_DIR, name), dpi=120, bbox_inches="tight")


def main():
    plt.rcParams["axes.grid"] = True
    out = []

    def log(s=""):
        print(s)
        out.append(s)

    # ---- 1. Графік функції ----
    xx = np.linspace(A, B, 1000)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(xx, f(xx), label=r"$f(x) = 50 + 20\sin(\pi x/12) + 5e^{-0.2(x-12)^2}$")
    plt.xlabel("Час, x (год)")
    plt.ylabel("Навантаження, f(x)")
    plt.legend()
    save_fig(fig, "function.png")

    # ---- 2. Точне значення ----
    i0 = exact_integral(A, B)
    log(f"I0 = {i0:.13f}")

    # ---- 3–4. Залежність похибки від N ----
    ns = np.arange(10, 1001, 2)
    errs = np.array([abs(simpson(f, A, B, n) - i0) for n in ns])
    n_opt = int(ns[np.argmax(errs <= EPS)])
    eps_opt = abs(simpson(f, A, B, n_opt) - i0)
    log("\n   N  |   I(N)              |  eps(N)")
    for n in list(range(10, 50, 4)) + [100, 500, 1000]:
        s = simpson(f, A, B, n)
        log(f" {n:4d} | {s:.13f} | {abs(s - i0):.3e}")
    log(f"\nN_opt = {n_opt}, epsopt = {eps_opt:.3e}")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.semilogy(ns, np.maximum(errs, 1e-16), ".-", ms=3, lw=0.8)
    ax.axhline(EPS, color="tab:red", ls="--", label="ε = 10⁻¹²")
    ax.plot([n_opt], [max(eps_opt, 1e-16)], "r*", ms=14, label=f"N_opt = {n_opt}")
    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("ε(N) = |I(N) − I₀|")
    ax.set_title("Похибка складової формули Сімпсона (значення 10⁻¹⁶ — точний збіг)")
    ax.legend()
    save_fig(fig, "error_vs_n.png")

    # ---- 5. N0 ≈ N_opt / 10, кратне 8 ----
    n0 = max(8, int(round(n_opt / 10 / 8)) * 8)
    i_n0 = simpson(f, A, B, n0)
    eps0 = abs(i_n0 - i0)
    log(f"\nN0 = {n0}: I(N0) = {i_n0:.13f}, eps0 = {eps0:.3e}")

    # ---- 6. Рунге-Ромберг ----
    i_n02 = simpson(f, A, B, n0 // 2)
    i_r = runge_romberg(i_n0, i_n02)
    eps_r = abs(i_r - i0)
    log(f"I(N0/2) = {i_n02:.13f}")
    log(f"Рунге-Ромберг: I_R = {i_r:.13f}, epsR = {eps_r:.3e}")

    # ---- 7. Ейткен ----
    i_n04 = simpson(f, A, B, n0 // 4)
    i_e, p = aitken(i_n0, i_n02, i_n04)
    eps_e = abs(i_e - i0)
    log(f"I(N0/4) = {i_n04:.13f}")
    log(f"Ейткен: I_E = {i_e:.13f}, p = {p:.4f}, epsE = {eps_e:.3e}")

    # ---- 8. Характер зміни похибки для різних N0 ----
    log("\n  N0 |   eps0    |   epsR    |   epsE    |    p")
    rows = []
    for n in (8, 16, 24, 32):
        a1, a2, a3 = (simpson(f, A, B, n // k) for k in (1, 2, 4))
        ie, pp = aitken(a1, a2, a3)
        rows.append((n, abs(a1 - i0), abs(runge_romberg(a1, a2) - i0), abs(ie - i0), pp))
        log(f" {n:3d} | {rows[-1][1]:.3e} | {rows[-1][2]:.3e} | {rows[-1][3]:.3e} | {pp:.4f}")

    # для порівняння: неперіодичний інтервал [0, 12], де формула Сімпсона має звичайний 4-й порядок
    log("\nКонтрольний інтервал [0, 12] (функція не є періодичною на ньому):")
    i0c = exact_integral(0, 12)
    log("  N0 |   eps0    |   epsR    |   epsE    |    p")
    for n in (8, 16, 32, 64):
        a1, a2, a3 = (simpson(f, 0, 12, n // k) for k in (1, 2, 4))
        ie, pp = aitken(a1, a2, a3)
        log(f" {n:3d} | {abs(a1 - i0c):.3e} | {abs(runge_romberg(a1, a2) - i0c):.3e} | "
            f"{abs(ie - i0c):.3e} | {pp:.4f}")

    # ---- 9. Адаптивний алгоритм ----
    log("\nАдаптивний алгоритм:")
    log("   delta  |   I               |   eps     | обчислень f | відрізків")
    deltas = [10.0 ** -k for k in range(1, 13)]
    ad = []
    for d in deltas:
        val, calls, segs = adaptive_simpson(f, A, B, d)
        ad.append((d, abs(val - i0), calls))
        log(f" {d:8.0e} | {val:.12f} | {abs(val - i0):.3e} | {calls:11d} | {segs}")
    ad = np.array(ad)

    # для порівняння: скільки обчислень потрібно складовій формулі для тієї ж точності
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].loglog(ad[:, 0], np.maximum(ad[:, 1], 1e-16), "o-", label="ε(δ)")
    axes[0].loglog(ad[:, 0], ad[:, 0], "--", color="gray", label="ε = δ")
    axes[0].set_xlabel("δ")
    axes[0].set_ylabel("ε = |I − I₀|")
    axes[0].set_title("Точність адаптивного алгоритму")
    axes[0].invert_xaxis()
    axes[0].legend()
    axes[1].semilogx(ad[:, 0], ad[:, 2], "o-")
    axes[1].set_xlabel("δ")
    axes[1].set_ylabel("Кількість обчислень f(x)")
    axes[1].set_title("Обчислювальні витрати")
    axes[1].invert_xaxis()
    fig.tight_layout()
    save_fig(fig, "adaptive.png")

    # розміщення вузлів адаптивного алгоритму та локальний крок для delta = 1e-3
    pts = []
    g = lambda x: (pts.append(float(x)), f(x))[1]
    adaptive_simpson(g, A, B, 1e-3)
    pts = np.unique(pts)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(xx, f(xx), lw=1.5)
    axes[0].plot(pts, f(pts), "o", color="tab:red", ms=3, label=f"вузли ({len(pts)}), δ = 10⁻³")
    axes[0].set_ylabel("f(x)")
    axes[0].set_title("Вузли адаптивного алгоритму")
    axes[0].legend()
    axes[1].step(pts[:-1], np.diff(pts), where="post")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Локальний крок")
    fig.tight_layout()
    save_fig(fig, "adaptive_nodes.png")

    with open(RES_FILE, "w", encoding="utf-8") as file:
        file.write("\n".join(out) + "\n")
    plt.show()


if __name__ == "__main__":
    main()
