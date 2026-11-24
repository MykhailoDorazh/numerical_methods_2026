"""
Лабораторна робота №2. Розділені різниці. Інтерполяційні многочлени Ньютона
та факторіальних многочленів.

Варіант 1. Інженерія продуктивності: прогнозування часу виконання алгоритму.
- зчитування експериментальних даних (n, t) з CSV-файлу;
- таблиця розділених різниць, многочлен Ньютона, прогноз t(6000);
- таблиця скінченних різниць, факторіальні многочлени (заміна t = log2(n / 1000),
  яка робить вузли 1000, 2000, 4000, 8000, 16000 рівновіддаленими);
- вплив кількості вузлів на прогноз;
- дослідницька частина: табуляція модельної функції у текстовий файл,
  вплив кроку та кількості вузлів, ефект Рунге, порівняння з методом Лагранжа.
"""

import csv
import math
import os

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.csv")
TAB_FILE = os.path.join(BASE_DIR, "tabulation.txt")
IMG_DIR = os.path.join(BASE_DIR, "images")

EXPERIMENT = [(1000, 3), (2000, 5), (4000, 11), (8000, 28), (16000, 85)]
X_TARGET = 6000
NODE_COUNTS = [5, 10, 20]


# ---------------------------------------------------------------------------
# 1. Вхідні дані
# ---------------------------------------------------------------------------

def write_data(filename):
    """Записує експериментальні дані варіанту у CSV-файл (якщо його ще немає)."""
    if os.path.exists(filename):
        return
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["n", "t"])
        writer.writerows(EXPERIMENT)


def read_data(filename):
    """Зчитування масивів x_i, f_i з CSV-файлу (колонки n, t)."""
    x = []
    y = []
    with open(filename, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row["n"]))
            y.append(float(row["t"]))
    return np.array(x), np.array(y)


def write_tabulation(file, f, a, b, n, title):
    """Табуляція f на [a, b] з кроком h = (b - a) / n у текстовий файл (дописуванням)."""
    h = (b - a) / n
    file.write(f"# {title}: [{a:g}, {b:g}], n = {n}, h = {h:g}\n")
    file.write("i x_i y_i\n")
    for i in range(n + 1):
        xi = a + i * h
        file.write(f"{i} {xi:.10g} {f(xi):.15g}\n")
    file.write("\n")


def read_tabulation(filename):
    """Зчитує з текстового файлу всі набори вузлів {x_i, y_i}: {назва: (x, y)}."""
    sets = {}
    title = None
    with open(filename, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("#"):
                title = line[1:].split(":")[0].strip()
                sets[title] = ([], [])
            elif line and line[0].isdigit():
                _, xi, yi = line.split()
                sets[title][0].append(float(xi))
                sets[title][1].append(float(yi))
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in sets.items()}


# ---------------------------------------------------------------------------
# 2. Многочлен Ньютона
# ---------------------------------------------------------------------------

def omega(x, nodes, k):
    """omega_k(x) = (x - x_0)(x - x_1)...(x - x_k)."""
    res = np.ones_like(np.asarray(x, float))
    for i in range(k + 1):
        res = res * (x - nodes[i])
    return res


def divided_difference(x, y, k):
    """Розділена різниця f(x_0, ..., x_k) = sum f(x_i) / prod_{j != i} (x_i - x_j)."""
    s = 0.0
    for i in range(k + 1):
        denom = 1.0
        for j in range(k + 1):
            if j != i:
                denom *= x[i] - x[j]
        s += y[i] / denom
    return s


def divided_differences_table(x, y):
    """Таблиця розділених різниць за рекурентною формулою: table[i][k] = f(x_i, ..., x_{i+k})."""
    n = len(x)
    table = np.full((n, n), np.nan)
    table[:, 0] = y
    for k in range(1, n):
        for i in range(n - k):
            table[i, k] = (table[i + 1, k - 1] - table[i, k - 1]) / (x[i + k] - x[i])
    return table


def newton_coeffs(x, y):
    """Коефіцієнти многочлена Ньютона f(x_0), f(x_0, x_1), ..., f(x_0, ..., x_n)."""
    return divided_differences_table(x, y)[0].copy()


def newton_eval(nodes, coeffs, x):
    """N_n(x) = f_0 + sum_{k=1}^{n} omega_{k-1}(x) f(x_0, ..., x_k) (схема Горнера)."""
    x = np.asarray(x, float)
    res = np.full_like(x, coeffs[-1])
    for k in range(len(coeffs) - 2, -1, -1):
        res = res * (x - nodes[k]) + coeffs[k]
    return res


def newton_backward(x, y, xp):
    """Многочлен Ньютона для інтерполяції назад (від вузла x_n)."""
    table = divided_differences_table(x, y)
    n = len(x) - 1
    res, w = 0.0, 1.0
    for k in range(n + 1):
        res += w * table[n - k, k]           # f(x_n, x_{n-1}, ..., x_{n-k})
        w *= xp - x[n - k]
    return res


# ---------------------------------------------------------------------------
# 3. Факторіальні многочлени
# ---------------------------------------------------------------------------

def finite_differences_table(y):
    """Таблиця скінченних різниць: table[i][k] = Δ^k f_i."""
    n = len(y)
    table = np.full((n, n), np.nan)
    table[:, 0] = y
    for k in range(1, n):
        table[:n - k, k] = table[1:n - k + 1, k - 1] - table[:n - k, k - 1]
    return table


def factorial_power(t, k):
    """Факторіальний многочлен t^(k) = t(t - 1)...(t - k + 1), t^(0) = 1."""
    res = np.ones_like(np.asarray(t, float))
    for i in range(k):
        res = res * (t - i)
    return res


def factorial_eval(y, t):
    """P_n(t) = sum_{k=0}^{n} Δ^k f(0) / k! * t^(k) для вузлів t = 0, 1, ..., n."""
    diffs = finite_differences_table(y)[0]
    t = np.asarray(t, float)
    return sum(diffs[k] / math.factorial(k) * factorial_power(t, k) for k in range(len(y)))


# ---------------------------------------------------------------------------
# 4. Метод Лагранжа (для порівняння)
# ---------------------------------------------------------------------------

def lagrange_eval(x_nodes, y_nodes, x):
    x = np.asarray(x, float)
    res = np.zeros_like(x)
    for i in range(len(x_nodes)):
        li = np.ones_like(x)
        for j in range(len(x_nodes)):
            if j != i:
                li = li * (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        res += y_nodes[i] * li
    return res


# ---------------------------------------------------------------------------
# Допоміжне
# ---------------------------------------------------------------------------

def print_table(title, x, table, name="f"):
    n = len(x)
    print(f"\n{title}:")
    header = f"{'i':>2} | {'x_i':>8} | " + " | ".join(f"{name + str(k):>13}" for k in range(n))
    print(header)
    for i in range(n):
        cells = " | ".join(f"{table[i, k]:13.6e}" if not np.isnan(table[i, k]) else " " * 13
                           for k in range(n))
        print(f"{i:2d} | {x[i]:8g} | {cells}")


def save_fig(fig, name):
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(os.path.join(IMG_DIR, name), dpi=120, bbox_inches="tight")


def main():
    plt.rcParams["axes.grid"] = True

    # ================= Експериментальні дані =================
    write_data(DATA_FILE)
    x, y = read_data(DATA_FILE)
    print("x:", x)
    print("y:", y)

    # ---- Таблиця розділених різниць ----
    dd = divided_differences_table(x, y)
    print_table("Таблиця розділених різниць f(x_i, ..., x_{i+k})", x, dd, name="k=")
    coeffs = newton_coeffs(x, y)
    direct = [divided_difference(x, y, k) for k in range(len(x))]
    print("\nКоефіцієнти Ньютона (рекурентно):", coeffs)
    print("Коефіцієнти Ньютона (явна формула):", np.array(direct))
    print("Макс. розбіжність:", np.max(np.abs(coeffs - direct)))

    # ---- Прогноз Ньютоном ----
    n_fwd = newton_eval(x, coeffs, X_TARGET)
    n_bwd = newton_backward(x, y, X_TARGET)
    print(f"\nN_4({X_TARGET}) вперед = {n_fwd:.6f} мс")
    print(f"N_4({X_TARGET}) назад  = {n_bwd:.6f} мс")
    print("Перевірка у вузлах (max |N(x_i) - y_i|):", np.max(np.abs(newton_eval(x, coeffs, x) - y)))

    # ---- Факторіальні многочлени: t = log2(n / 1000) -> 0, 1, 2, 3, 4 ----
    t_nodes = np.log2(x / x[0])
    t_target = math.log2(X_TARGET / x[0])
    print("\nЗаміна змінних t = log2(n / 1000):", t_nodes, f"t* = {t_target:.6f}")
    fd = finite_differences_table(y)
    print_table("Таблиця скінченних різниць Δ^k f_i", t_nodes, fd, name="Δ^")
    p_fact = float(factorial_eval(y, t_target))
    print(f"\nP_4(t*) (факторіальні многочлени) = {p_fact:.6f} мс")
    # контроль: многочлен Ньютона за змінною t дає той самий результат
    n_t = float(newton_eval(t_nodes, newton_coeffs(t_nodes, y), t_target))
    print(f"N_4(t*) (Ньютон за змінною t)      = {n_t:.6f} мс")

    # ---- Вплив кількості вузлів на прогноз (найближчі до 6000 вузли) ----
    print(f"\nПрогноз t({X_TARGET}) при різній кількості вузлів (вузли, найближчі до {X_TARGET}):")
    print(" Вузлів | Вузли                             | Ньютон, мс | Фактор., мс | Приріст (оцінка похибки)")
    order = np.argsort(np.abs(t_nodes - t_target), kind="stable")
    prev = None
    for m in range(2, len(x) + 1):
        idx = np.sort(order[:m])
        pn = float(newton_eval(x[idx], newton_coeffs(x[idx], y[idx]), X_TARGET))
        # найближчі вузли йдуть підряд, тож за змінною t вони рівновіддалені: t = idx[0], ..., idx[-1]
        pf = float(factorial_eval(y[idx], t_target - idx[0]))
        delta = "" if prev is None else f"{abs(pn - prev):.4f}"
        print(f"   {m:2d}   | {str(x[idx].astype(int).tolist()):33s} | {pn:10.4f} | {pf:11.4f} | {delta}")
        prev = pn

    # ---- Модельна (еталонна) функція: степенева апроксимація t = a n^p ----
    p, log_a = np.polyfit(np.log(x), np.log(y), 1)
    a_coef = math.exp(log_a)

    def f_model(v):
        return a_coef * np.power(v, p)

    print(f"\nМодельна функція (МНК у log-log): t(n) = {a_coef:.6e} * n^{p:.6f}")
    print(f"t_model({X_TARGET}) = {f_model(X_TARGET):.6f} мс")

    # ---- Графік: експериментальні точки та інтерполяційні криві ----
    xx = np.linspace(x[0], x[-1], 1000)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(xx, newton_eval(x, coeffs, xx), label="Многочлен Ньютона N₄(n)")
    plt.plot(xx, factorial_eval(y, np.log2(xx / x[0])), "--", label="Факторіальний многочлен P₄(log₂(n/1000))")
    plt.plot(xx, f_model(xx), ":", color="gray", label=f"Модель t = {a_coef:.2e}·n^{p:.3f}")
    plt.plot(x, y, "o", color="tab:red", ms=8, label="Експериментальні дані")
    plt.plot([X_TARGET], [n_fwd], "*", color="tab:blue", ms=15, label=f"N₄(6000) = {n_fwd:.2f} мс")
    plt.plot([X_TARGET], [p_fact], "*", color="tab:orange", ms=15, label=f"P₄(6000) = {p_fact:.2f} мс")
    plt.xlabel("n (розмір вхідних даних)")
    plt.ylabel("t, мс")
    plt.title("Час виконання алгоритму: інтерполяція")
    plt.legend()
    save_fig(fig, "interpolation.png")

    # ================= Дослідницька частина =================
    a, b = x[0], x[-1]

    # ---- Табуляція модельної функції у текстовий файл і зчитування ----
    h_fixed = (b - a) / max(NODE_COUNTS)
    with open(TAB_FILE, "w", encoding="utf-8") as file:
        for n in NODE_COUNTS:
            write_tabulation(file, f_model, a, b, n, f"interval n={n}")
        for n in NODE_COUNTS:
            write_tabulation(file, f_model, a, a + h_fixed * n, n, f"step n={n}")
    print(f"\nТабуляцію модельної функції записано у файл {os.path.basename(TAB_FILE)}")
    tab = read_tabulation(TAB_FILE)

    # ---- Вплив кроку: фіксований інтервал [a, b], n = 5, 10, 20 ----
    print(f"\nФіксований інтервал [{a:g}, {b:g}]:")
    print("  n |    h    | max ε(x)    | ε(6000)     | max |ω_n(x)|  | max|N - L|  | N(6000)")
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    for n in NODE_COUNTS:
        xn, yn = tab[f"interval n={n}"]
        cn = newton_coeffs(xn, yn)
        hh = (b - a) / (20 * n)
        xs = a + hh * np.arange(20 * n + 1)
        fx, nx = f_model(xs), newton_eval(xn, cn, xs)
        eps = np.abs(fx - nx)
        wn = omega(xs, xn, n)
        e_t = abs(f_model(X_TARGET) - newton_eval(xn, cn, X_TARGET))
        lag_diff = np.max(np.abs(nx - lagrange_eval(xn, yn, xs)))
        fact = factorial_eval(yn, (xs - a) / (xn[1] - xn[0]))
        print(f" {n:2d} | {xn[1] - xn[0]:7.1f} | {eps.max():.4e} | {e_t:.4e} | {np.abs(wn).max():.4e} | "
              f"{lag_diff:.2e} | {newton_eval(xn, cn, X_TARGET):.8f}  (max|N - P_fact| = {np.max(np.abs(nx - fact)):.1e})")
        axes[0].plot(xs, nx, label=f"N_{n}(x)")
        axes[1].semilogy(xs, eps + 1e-18, label=f"n = {n}")
        axes[2].plot(xs, wn / np.abs(wn).max(), label=f"n = {n}")
    axes[0].plot(xs, f_model(xs), "k:", lw=2, label="f(x)")
    axes[0].set_ylabel("t, мс")
    axes[0].set_title("f(x) та N_n(x) на [1000, 16000]")
    axes[1].set_ylabel("ε(x) = |f(x) − N_n(x)|")
    axes[1].set_title("Похибка інтерполяції")
    axes[1].set_ylim(bottom=1e-12)
    axes[2].set_ylabel("ω_n(x) / max|ω_n|")
    axes[2].set_xlabel("n (розмір вхідних даних)")
    axes[2].set_title("Функція ω_n(x) (нормована)")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    save_fig(fig, "step_study.png")

    # ---- Вплив кількості вузлів: фіксований крок h, інтервал [a, a + h n] ----
    print(f"\nФіксований крок h = {h_fixed:g}:")
    print("  n |     [a, b]        | max ε(x)    | max |ω_n(x)|")
    fig, ax = plt.subplots(figsize=(10, 5))
    for n in NODE_COUNTS:
        xn, yn = tab[f"step n={n}"]
        cn = newton_coeffs(xn, yn)
        bn = xn[-1]
        xs = np.linspace(a, bn, 20 * n + 1)
        eps = np.abs(f_model(xs) - newton_eval(xn, cn, xs))
        print(f" {n:2d} | [{a:g}, {bn:g}] | {eps.max():.4e} | {np.abs(omega(xs, xn, n)).max():.4e}")
        ax.semilogy(xs, eps + 1e-18, label=f"n = {n}, [{a:g}, {bn:g}]")
    ax.set_xlabel("n (розмір вхідних даних)")
    ax.set_ylabel("ε(x)")
    ax.set_ylim(bottom=1e-10)
    ax.set_title(f"Похибка при фіксованому кроці h = {h_fixed:g}")
    ax.legend()
    save_fig(fig, "nodes_study.png")

    # ---- Ефект Рунге: f(x) = 1 / (1 + 25 x^2) на [-1, 1] ----
    def runge(v):
        return 1 / (1 + 25 * v ** 2)

    print("\nЕфект Рунге, f(x) = 1/(1 + 25x²) на [-1, 1]:")
    print("  n | max ε (рівновіддалені) | max ε (вузли Чебишова)")
    xs = np.linspace(-1, 1, 2001)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(xs, runge(xs), "k", lw=2, label="f(x)")
    for n in NODE_COUNTS:
        xe = np.linspace(-1, 1, n + 1)
        xc = np.cos((2 * np.arange(n + 1) + 1) * np.pi / (2 * n + 2))[::-1]
        ne = newton_eval(xe, newton_coeffs(xe, runge(xe)), xs)
        nc = newton_eval(xc, newton_coeffs(xc, runge(xc)), xs)
        print(f" {n:2d} | {np.abs(runge(xs) - ne).max():22.4e} | {np.abs(runge(xs) - nc).max():.4e}")
        axes[0].plot(xs, ne, label=f"N_{n}(x)")
        axes[1].semilogy(xs, np.abs(runge(xs) - ne) + 1e-18, label=f"n = {n}")
    axes[0].set_ylim(-1.5, 2)
    axes[0].set_title("Рівновіддалені вузли: коливання N_n(x)")
    axes[0].legend()
    axes[1].set_ylim(bottom=1e-8)
    axes[1].set_title("Похибка ε(x)")
    axes[1].legend()
    fig.tight_layout()
    save_fig(fig, "runge.png")

    # ---- Порівняння з методом Лагранжа на експериментальних даних ----
    l_val = float(lagrange_eval(x, y, X_TARGET))
    print(f"\nЛагранж L_4({X_TARGET}) = {l_val:.10f}, Ньютон N_4({X_TARGET}) = {n_fwd:.10f}, "
          f"різниця = {abs(l_val - n_fwd):.2e}")

    plt.show()


if __name__ == "__main__":
    main()
