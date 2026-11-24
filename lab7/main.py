"""Лабораторна робота №7. Ітераційні методи розв'язування СЛАР.

Запуск: py main.py; без відкривання графіків: py main.py --no-show.
Усі дані та результати зберігаються поруч із програмою.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
N = 100
SEED = 2026
EPSILON = 1e-14
MAX_ITERATIONS = 10000
METHOD_NAMES = {
    "simple_iteration": "Проста ітерація",
    "jacobi": "Якобі",
    "gauss_seidel": "Гаусса-Зейделя",
}


def matrix_vector_product(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Власний добуток матриці на вектор із точнішим підсумовуванням."""
    return np.asarray([math.fsum(float(a) * float(x) for a, x in zip(row, vector))
                       for row in matrix], dtype=float)


def vector_norm(vector: np.ndarray) -> float:
    """Норма ||x||_inf = max_i |x_i|."""
    return max(abs(float(value)) for value in vector)


def matrix_norm(matrix: np.ndarray) -> float:
    """Узгоджена матрична норма ||A||_inf = max_i sum_j |a_ij|."""
    return max(math.fsum(abs(float(value)) for value in row) for row in matrix)


def read_matrix(path: Path) -> np.ndarray:
    """Прочитати квадратну матрицю з текстового файла."""
    with path.open(encoding="utf-8") as stream:
        matrix = np.asarray([[float(value) for value in line.split()]
                             for line in stream if line.strip()], dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Матриця A повинна бути квадратною.")
    if not np.isfinite(matrix).all():
        raise ValueError("Матриця містить нескінченні або невизначені числа.")
    return matrix


def read_vector(path: Path) -> np.ndarray:
    """Прочитати вектор із текстового файла."""
    values = [float(value) for value in path.read_text(encoding="utf-8").split()]
    vector = np.asarray(values, dtype=float)
    if not np.isfinite(vector).all():
        raise ValueError("Вектор містить нескінченні або невизначені числа.")
    return vector


def generate_system(directory: Path, seed: int = SEED) -> None:
    """Побудувати додатну строго діагонально переважну A та b=A*x*."""
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(-1.0, 1.0, size=(N, N)) / N
    np.fill_diagonal(matrix, 0.0)
    margins = rng.uniform(0.25, 0.5, size=N)
    for i in range(N):
        matrix[i, i] = math.fsum(abs(float(value)) for value in matrix[i]) + margins[i]
    exact = np.full(N, 2.5)
    vector = matrix_vector_product(matrix, exact)
    np.savetxt(directory / "matrix_A.txt", matrix, fmt="%.17g")
    np.savetxt(directory / "vector_B.txt", vector, fmt="%.17g")


def residual(matrix: np.ndarray, vector: np.ndarray, x: np.ndarray) -> np.ndarray:
    return matrix_vector_product(matrix, x) - vector


def history_entry(k: int, matrix: np.ndarray, vector: np.ndarray,
                  x: np.ndarray, exact: np.ndarray, update: float | None) -> dict:
    return {"iteration": k, "residual_inf": vector_norm(residual(matrix, vector, x)),
            "true_error_inf": vector_norm(x - exact), "update_inf": update}


def simple_iteration(matrix: np.ndarray, vector: np.ndarray, initial: np.ndarray,
                     exact: np.ndarray, epsilon: float, tau: float) -> dict:
    """x^(k+1) = x^k + tau * (b-A*x^k); зупинка за нормою нев'язки."""
    x = initial.copy()
    history = [history_entry(0, matrix, vector, x, exact, None)]
    for k in range(1, MAX_ITERATIONS + 1):
        x_next = x - tau * residual(matrix, vector, x)
        entry = history_entry(k, matrix, vector, x_next, exact, vector_norm(x_next - x))
        history.append(entry)
        x = x_next
        if entry["residual_inf"] <= epsilon and entry["update_inf"] <= epsilon:
            return solver_result("simple_iteration", x, history)
    raise RuntimeError("Проста ітерація не досягла заданої норми нев'язки.")


def jacobi(matrix: np.ndarray, vector: np.ndarray, initial: np.ndarray,
           exact: np.ndarray, epsilon: float) -> dict:
    """Кожна нова компонента використовує лише попередній вектор."""
    n = len(vector)
    x = initial.copy()
    history = [history_entry(0, matrix, vector, x, exact, None)]
    for k in range(1, MAX_ITERATIONS + 1):
        x_next = np.empty(n)
        for i in range(n):
            off_diagonal = math.fsum(float(matrix[i, j]) * float(x[j])
                                     for j in range(n) if j != i)
            x_next[i] = (vector[i] - off_diagonal) / matrix[i, i]
        entry = history_entry(k, matrix, vector, x_next, exact, vector_norm(x_next - x))
        history.append(entry)
        x = x_next
        if entry["residual_inf"] <= epsilon and entry["update_inf"] <= epsilon:
            return solver_result("jacobi", x, history)
    raise RuntimeError("Метод Якобі не досяг заданої норми нев'язки.")


def gauss_seidel(matrix: np.ndarray, vector: np.ndarray, initial: np.ndarray,
                 exact: np.ndarray, epsilon: float) -> dict:
    """Уже оновлені компоненти відразу використовуються в тому ж проході."""
    n = len(vector)
    x = initial.copy()
    history = [history_entry(0, matrix, vector, x, exact, None)]
    for k in range(1, MAX_ITERATIONS + 1):
        previous = x.copy()
        for i in range(n):
            off_diagonal = math.fsum(float(matrix[i, j]) * float(x[j])
                                     for j in range(n) if j != i)
            x[i] = (vector[i] - off_diagonal) / matrix[i, i]
        entry = history_entry(k, matrix, vector, x, exact, vector_norm(x - previous))
        history.append(entry)
        if entry["residual_inf"] <= epsilon and entry["update_inf"] <= epsilon:
            return solver_result("gauss_seidel", x, history)
    raise RuntimeError("Метод Зейделя не досяг заданої норми нев'язки.")


def solver_result(method: str, solution: np.ndarray, history: list[dict]) -> dict:
    last = history[-1]
    return {"method": method, "name": METHOD_NAMES[method],
            "iterations": last["iteration"], "solution": solution.tolist(),
            "residual_inf": last["residual_inf"], "true_error_inf": last["true_error_inf"],
            "update_inf": last["update_inf"], "history": history}


def iteration_matrices(matrix: np.ndarray, tau: float) -> dict:
    """Матриці переходу: C_R=I-tau*A, C_J=-D^-1(L+U), C_G=-(D+L)^-1 U."""
    n = len(matrix)
    simple = np.eye(n) - tau * matrix
    jac = -matrix / np.diag(matrix)[:, None]
    np.fill_diagonal(jac, 0.0)
    seidel = np.zeros((n, n))
    # Пряма підстановка для кожного стовпця: (D+L) C_G = -U.
    for column in range(n):
        for i in range(n):
            right = -matrix[i, column] if column > i else 0.0
            subtotal = math.fsum(float(matrix[i, j]) * float(seidel[j, column])
                                 for j in range(i))
            seidel[i, column] = (right - subtotal) / matrix[i, i]
    return {"simple_iteration": simple, "jacobi": jac, "gauss_seidel": seidel}


def calculate(directory: Path) -> dict:
    matrix = read_matrix(directory / "matrix_A.txt")
    vector = read_vector(directory / "vector_B.txt")
    n = len(vector)
    if matrix.shape != (n, n) or n != N:
        raise ValueError("Завдання передбачає узгоджену систему розмірності 100.")
    off_sums = [math.fsum(abs(float(matrix[i, j])) for j in range(n) if j != i)
                for i in range(n)]
    margins = [float(matrix[i, i]) - off_sums[i] for i in range(n)]
    if min(margins) <= 0.0:
        raise ValueError("Потрібна додатна діагональ і строге рядкове переважання.")
    exact = np.full(n, 2.5)
    initial = np.full(n, 1.0)
    tau = 1.0 / max(float(matrix[i, i]) for i in range(n))
    matrices = iteration_matrices(matrix, tau)
    q_norms = {key: matrix_norm(value) for key, value in matrices.items()}
    if not all(value < 1.0 for value in q_norms.values()):
        raise ValueError("Не виконана перевірена достатня умова збіжності.")
    methods = [simple_iteration(matrix, vector, initial, exact, EPSILON, tau),
               jacobi(matrix, vector, initial, exact, EPSILON),
               gauss_seidel(matrix, vector, initial, exact, EPSILON)]
    for method in methods:
        if method["true_error_inf"] > EPSILON:
            raise ArithmeticError("Фактична похибка розв'язку перевищила 1e-14.")
        method["q_inf"] = q_norms[method["method"]]
    return {"lab": 7, "n": n, "seed": SEED, "epsilon": EPSILON,
            "initial_value": 1.0, "exact_value": 2.5,
            "generation": "off-diagonal U[-1,1]/100; a_ii=sum(abs(off-diagonal))+U[0.25,0.5]",
            "stopping_criterion": "||A*x^(k)-b||_inf <= 1e-14 AND ||x^(k)-x^(k-1)||_inf <= 1e-14 (same for all methods)",
            "norm_definition": "vector: max(abs(x_i)); matrix: max(row sum(abs(a_ij)))",
            "matrix_norm_inf": matrix_norm(matrix), "vector_norm_inf": vector_norm(vector),
            "minimum_dominance_margin": min(margins),
            "minimum_diagonal": min(float(matrix[i, i]) for i in range(n)),
            "maximum_diagonal": max(float(matrix[i, i]) for i in range(n)),
            "tau": tau, "q_inf": q_norms,
            "exact_solution_residual_inf": vector_norm(residual(matrix, vector, exact)),
            "matrix_preview": matrix[:5, :5].tolist(), "vector_preview": vector[:5].tolist(),
            "methods": methods}


def save_outputs(result: dict, directory: Path) -> None:
    processed = directory / "processed"
    processed.mkdir(exist_ok=True)
    with (directory / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    def write_csv(name, headings, rows):
        with (processed / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(headings); writer.writerows(rows)
    write_csv("summary.csv", ["method", "iterations", "residual_inf", "true_error_inf", "update_inf", "q_inf"],
              [[m[key] for key in ["name", "iterations", "residual_inf", "true_error_inf", "update_inf", "q_inf"]]
               for m in result["methods"]])
    write_csv("solutions.csv", ["index", "exact"] + [m["method"] for m in result["methods"]],
              [[i + 1, 2.5] + [m["solution"][i] for m in result["methods"]] for i in range(result["n"])])
    for method in result["methods"]:
        write_csv(f"history_{method['method']}.csv", ["iteration", "residual_inf", "true_error_inf", "update_inf"],
                  [[h["iteration"], h["residual_inf"], h["true_error_inf"], h["update_inf"]]
                   for h in method["history"]])
    lines = ["Лабораторна робота №7. Ітераційні методи розв'язування СЛАР.",
             f"Розмірність: {result['n']} x {result['n']}; seed={result['seed']}; epsilon={result['epsilon']:.0e}.",
             "Відомий розв'язок: x_i*=2.5. Початкове наближення: x_i^(0)=1.0.",
             "Матрицю A та вектор b записано в текстові файли і повторно зчитано перед розв'язуванням.",
             f"||A||_inf={result['matrix_norm_inf']:.15g}; ||b||_inf={result['vector_norm_inf']:.15g}.",
             f"Мінімальне діагональне переважання: {result['minimum_dominance_margin']:.15g}.",
             f"tau=1/max(a_ii)={result['tau']:.15g}.",
             "Однакові критерії зупинки: ||Ax-b||_inf <= 1e-14 І ||x(k)-x(k-1)||_inf <= 1e-14.",
             "Фактична похибка перевіряється незалежно як ||x-x*||_inf.", "",
             "Метод | Ітерацій | ||Ax-b||_inf | ||x-x*||_inf | ||x(k)-x(k-1)||_inf | ||C||_inf"]
    for m in result["methods"]:
        lines.append(f"{m['name']} | {m['iterations']} | {m['residual_inf']:.9e} | {m['true_error_inf']:.9e} | {m['update_inf']:.9e} | {m['q_inf']:.9f}")
    lines += ["", "Компоненти отриманих розв'язків (i від 1 до 100):",
              " i | Проста ітерація | Якобі | Гаусса-Зейделя"]
    for i in range(result["n"]):
        values = " | ".join(f"{m['solution'][i]:.17g}" for m in result["methods"])
        lines.append(f"{i+1:3d} | {values}")
    lines += ["", "Нев'язка і похибка - різні величини. Обидві перевірені чисельно.",
              "Числа ітерацій характеризують саме згенеровану систему і задані параметри."]
    text = "\n".join(lines) + "\n"
    (directory / "results.txt").write_text(text, encoding="utf-8")
    print(text[:1600])


def make_plots(result: dict, directory: Path):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12,
                         "axes.grid": True, "grid.alpha": 0.25})
    images = directory / "images"; images.mkdir(exist_ok=True)
    palette = ["#1d668d", "#d47718", "#238b61"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for method, color in zip(result["methods"], palette):
        k = [h["iteration"] for h in method["history"]]
        axes[0].semilogy(k, [h["residual_inf"] for h in method["history"]], "o-", ms=3.5, color=color, label=method["name"])
        axes[1].semilogy(k, [h["true_error_inf"] for h in method["history"]], "o-", ms=3.5, color=color, label=method["name"])
    for axis in axes:
        axis.axhline(EPSILON, color="#8b3636", ls="--", lw=1.2, label="ε = 10⁻¹⁴")
        axis.set_xlabel("Номер ітерації k")
        axis.legend(fontsize=10)
    axes[0].set_ylabel("‖Ax⁽ᵏ⁾ − b‖∞"); axes[0].set_title("Норма нев'язки")
    axes[1].set_ylabel("‖x⁽ᵏ⁾ − x*‖∞"); axes[1].set_title("Фактична похибка")
    fig.tight_layout(); fig.savefig(images / "convergence.png", dpi=180, bbox_inches="tight")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar([m["name"] for m in result["methods"]], [m["iterations"] for m in result["methods"]], color=palette, width=.55)
    ax.bar_label(bars, padding=4)
    ax.set_ylabel("Кількість ітерацій"); ax.set_ylim(0, max(m["iterations"] for m in result["methods"]) * 1.18)
    ax.set_title("Нев’язка і зміна наближення ≤ 10⁻¹⁴")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(images / "iteration_counts.png", dpi=180, bbox_inches="tight")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    indices = range(1, result["n"] + 1)
    for method, color in zip(result["methods"], palette):
        errors = [(value - result["exact_value"]) / 1e-15 for value in method["solution"]]
        ax.plot(indices, errors, ".-", lw=1, ms=4, color=color, label=method["name"])
    ax.axhline(0.0, color="black", lw=.7)
    ax.set_xlabel("Номер компоненти i"); ax.set_ylabel("(xᵢ − 2.5) / 10⁻¹⁵")
    ax.set_title("Покомпонентні похибки отриманих розв'язків")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.0), fontsize=10)
    fig.tight_layout(); fig.savefig(images / "solution_errors.png", dpi=180, bbox_inches="tight")
    return plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Зберегти графіки без відкривання вікон")
    parser.add_argument("--regenerate", action="store_true", help="Повторно створити A,b із seed=2026")
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    if args.regenerate or not all((BASE_DIR / name).exists() for name in ["matrix_A.txt", "vector_B.txt"]):
        generate_system(BASE_DIR)
    result = calculate(BASE_DIR)
    save_outputs(result, BASE_DIR)
    plt = make_plots(result, BASE_DIR)
    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
