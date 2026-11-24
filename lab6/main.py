"""
Лабораторна робота №6. LU-розклад. Ітераційні методи уточнення розв'язку СЛАР.

- генерація випадкової матриці A (100 x 100), вектора B = A X для x_i = 2.5, запис у текстові файли;
- LU-розклад (L — нижня трикутна, U — верхня трикутна з одиницями на діагоналі), запис у файл;
- розв'язок AX = B через LZ = B, UX = Z, оцінка точності eps = max |AX - B|;
- ітераційне уточнення розв'язку до точності eps0 = 1e-14
  (нев'язка B - AX обчислюється точно, у раціональних числах).
"""

import os
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
A_FILE = os.path.join(BASE_DIR, "matrix_A.txt")
B_FILE = os.path.join(BASE_DIR, "vector_B.txt")
LU_FILE = os.path.join(BASE_DIR, "matrix_LU.txt")

N = 100
X_TRUE = 2.5
EPS0 = 1e-14
SEED = 2026


# ---------------------------------------------------------------------------
# 1. Генерація та запис даних
# ---------------------------------------------------------------------------

def generate(n, seed, grid=True):
    """Випадкова матриця A з елементами з [-10, 10] та вектор B = A X, x_i = 2.5.
    grid=True: елементи округлюються до кратних 2^-20 (≈ 6 десяткових знаків), тоді добутки
    a_ij * 2.5 і їх суми обчислюються без похибки округлення і B є точним."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-10, 10, (n, n))
    if grid:
        a = np.round(a * 2 ** 20) / 2 ** 20
    x = np.full(n, X_TRUE)
    return a, mat_vec(a, x)


def write_matrix(filename, m):
    np.savetxt(filename, m, fmt="%.17e")


def write_vector(filename, v):
    np.savetxt(filename, v, fmt="%.17e")


def read_matrix(filename):
    return np.loadtxt(filename, ndmin=2)


def read_vector(filename):
    return np.loadtxt(filename, ndmin=1)


# ---------------------------------------------------------------------------
# 2. LU-розклад і розв'язування
# ---------------------------------------------------------------------------

def lu_decomposition(a):
    """A = LU, u_ii = 1. Для k = 1..n: стовпець k матриці L, потім рядок k матриці U."""
    n = len(a)
    l = np.zeros((n, n))
    u = np.zeros((n, n))
    for k in range(n):
        u[k, k] = 1.0
        for i in range(k, n):                                   # l_ik, i >= k
            l[i, k] = a[i, k] - sum(l[i, j] * u[j, k] for j in range(k))
        for i in range(k + 1, n):                               # u_ki, i > k
            u[k, i] = (a[k, i] - sum(l[k, j] * u[j, i] for j in range(k))) / l[k, k]
    return l, u


def write_lu(filename, l, u):
    """Записує L і U в один файл: спочатку L, потім U."""
    with open(filename, "w", encoding="utf-8") as file:
        file.write("# L\n")
        np.savetxt(file, l, fmt="%.17e")
        file.write("# U\n")
        np.savetxt(file, u, fmt="%.17e")


def read_lu(filename):
    with open(filename, encoding="utf-8") as file:
        lines = file.read().splitlines()
    split = lines.index("# U")
    l = np.array([list(map(float, s.split())) for s in lines[1:split]])
    u = np.array([list(map(float, s.split())) for s in lines[split + 1:]])
    return l, u


def lu_solve(l, u, b):
    """LZ = B (прямий хід), UX = Z (зворотний хід)."""
    n = len(b)
    z = np.zeros(n)
    for k in range(n):
        z[k] = (b[k] - sum(l[k, j] * z[j] for j in range(k))) / l[k, k]
    x = np.zeros(n)
    for k in range(n - 1, -1, -1):
        x[k] = z[k] - sum(u[k, j] * x[j] for j in range(k + 1, n))
    return x


def mat_vec(a, x):
    """Добуток матриці на вектор."""
    return np.array([sum(a[i, j] * x[j] for j in range(len(x))) for i in range(len(a))])


def norm(v):
    """Норма вектора max |v_i|."""
    return max(abs(vi) for vi in v)


# ---------------------------------------------------------------------------
# 3. Ітераційне уточнення
# ---------------------------------------------------------------------------

def residual_exact(a, x, b):
    """Нев'язка R = B - AX, обчислена точно (Fraction) і округлена до double один раз."""
    n = len(x)
    return np.array([float(Fraction(b[i]) - sum(Fraction(a[i, j]) * Fraction(x[j]) for j in range(n)))
                     for i in range(len(b))])


def residual_double(a, x, b):
    """Нев'язка R = B - AX у звичайній арифметиці double."""
    return b - mat_vec(a, x)


def refine(a, l, u, b, x0, eps, residual=residual_exact, max_iter=20):
    """X = X0 + ΔX, де A ΔX = B - A X0 (з тим самим LU-розкладом).
    Умови закінчення: ||ΔX|| <= eps та ||AX - B|| <= eps."""
    x = x0.copy()
    history = []
    for it in range(1, max_iter + 1):
        dx = lu_solve(l, u, residual(a, x, b))
        x = x + dx
        res = norm(residual(a, x, b))
        history.append((it, norm(dx), res, norm(x - X_TRUE)))
        if norm(dx) <= eps and res <= eps:
            break
    return x, history


def save_fig(fig, name):
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(os.path.join(IMG_DIR, name), dpi=120, bbox_inches="tight")


def main():
    plt.rcParams["axes.grid"] = True

    # ---- 1. Генерація і запис ----
    a, b = generate(N, SEED)
    write_matrix(A_FILE, a)
    write_vector(B_FILE, b)
    print(f"Матрицю A ({N}x{N}) записано у {os.path.basename(A_FILE)}, вектор B — у {os.path.basename(B_FILE)}")

    # ---- 2. Зчитування, LU-розклад ----
    a = read_matrix(A_FILE)
    b = read_vector(B_FILE)
    l, u = lu_decomposition(a)
    write_lu(LU_FILE, l, u)
    l, u = read_lu(LU_FILE)
    print(f"LU-розклад записано у {os.path.basename(LU_FILE)}")
    print(f"max |LU - A| = {np.max(np.abs(l @ u - a)):.3e}")
    print(f"min |l_kk| = {np.min(np.abs(np.diag(l))):.4f}, max |l_kk| = {np.max(np.abs(np.diag(l))):.4f}")
    print(f"Число обумовленості cond(A) = {np.linalg.cond(a):.3e}")

    # ---- 3–4. Розв'язок і його точність ----
    x0 = lu_solve(l, u, b)
    eps = norm(residual_exact(a, x0, b))
    print(f"\nПерші компоненти X0: {x0[:5]}")
    print(f"eps = max |A X0 - B| = {eps:.3e}")
    print(f"max |X0 - X_точне| = {norm(x0 - X_TRUE):.3e}")

    # ---- 5. Ітераційне уточнення ----
    x, hist = refine(a, l, u, b, x0, EPS0)
    print(f"\nІтераційне уточнення (eps0 = {EPS0:g}):")
    print(" ітерація |  ||ΔX||    | ||AX - B||  | ||X - X_точне||")
    print(f"     0    |     —      | {eps:.3e} | {norm(x0 - X_TRUE):.3e}")
    for it, ndx, res, err in hist:
        print(f"    {it:2d}    | {ndx:.3e} | {res:.3e} | {err:.3e}")
    done = hist[-1][1] <= EPS0 and hist[-1][2] <= EPS0
    print(f"Умову досягнуто: {done}, кількість ітерацій: {len(hist)}")
    print(f"Точний розв'язок відновлено: {bool(np.all(x == X_TRUE))}")

    # ---- Порівняння: нев'язка у звичайній арифметиці double ----
    x_d, hist_d = refine(a, l, u, b, x0, EPS0, residual=residual_double)
    print("\nПорівняння — нев'язка у double:")
    for it, ndx, res, err in hist_d[:6]:
        print(f"    {it:2d}    | {ndx:.3e} | {res:.3e} | {err:.3e}")
    print(f"... ітерацій: {len(hist_d)}, умову досягнуто: {hist_d[-1][1] <= EPS0 and hist_d[-1][2] <= EPS0}, "
          f"min ||AX - B|| = {min(h[2] for h in hist_d):.3e}")

    # ---- Порівняння: довільні дійсні елементи (B = AX обчислюється з округленням) ----
    a_g, b_g = generate(N, SEED, grid=False)
    l_g, u_g = lu_decomposition(a_g)
    x0_g = lu_solve(l_g, u_g, b_g)
    x_g, hist_g = refine(a_g, l_g, u_g, b_g, x0_g, EPS0, max_iter=5)
    print("\nПорівняння — довільні дійсні елементи A (точна нев'язка):")
    print(f"     0    |     —      | {norm(residual_exact(a_g, x0_g, b_g)):.3e} | {norm(x0_g - X_TRUE):.3e}")
    for it, ndx, res, err in hist_g:
        print(f"    {it:2d}    | {ndx:.3e} | {res:.3e} | {err:.3e}")

    # ---- Графік збіжності ----
    its = [0] + [h[0] for h in hist]
    fig, ax = plt.subplots(figsize=(9, 5))
    def curve(h_list, k, start=None):
        vals = ([start] if start is not None else []) + [h[k] for h in h_list]
        return [max(v, 1e-17) for v in vals]

    ax.semilogy(its, curve(hist, 2, eps), "s-", label="||AX − B||, точна нев'язка")
    ax.semilogy(its[1:], curve(hist, 1), "o-", label="||ΔX||, точна нев'язка")
    its_d = list(range(len(hist_d) + 1))
    ax.semilogy(its_d, curve(hist_d, 2, eps), "s--", color="gray", label="||AX − B||, нев'язка у double")
    ax.text(0.98, 0.04, "значення 10⁻¹⁷ відповідають точному нулю", transform=ax.transAxes,
            ha="right", fontsize=9, color="gray")
    ax.axhline(EPS0, color="tab:red", ls="--", label="eps₀ = 10⁻¹⁴")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel("Номер ітерації")
    ax.set_ylabel("Норма")
    ax.set_title("Ітераційне уточнення розв'язку СЛАР")
    ax.legend()
    save_fig(fig, "refinement.png")

    plt.show()


if __name__ == "__main__":
    main()
