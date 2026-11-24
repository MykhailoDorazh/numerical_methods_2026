"""Лабораторна робота №3. Поліноміальна апроксимація методом МНК.

Запуск: python main.py; без вікон графіків: python main.py --no-show.
Усі шляхи визначаються відносно цього файлу.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent


def read_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Прочитати таблицю Month,Temp з CSV і перевірити коректність вузлів."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    x = np.asarray([float(row["Month"]) for row in rows])
    y = np.asarray([float(row["Temp"]) for row in rows])
    if len(x) < 12 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Потрібні щонайменше 12 скінченних пар Month,Temp.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("Місяці мають бути впорядкованими без повторень.")
    return x, y


def form_matrix(x: np.ndarray, m: int) -> np.ndarray:
    """B[i,j] = sum(x[k]**(i+j)), нормальна матриця МНК."""
    return np.asarray([[np.sum(x ** (i + j)) for j in range(m + 1)]
                       for i in range(m + 1)], dtype=float)


def form_vector(x: np.ndarray, y: np.ndarray, m: int) -> np.ndarray:
    """C[i] = sum(y[k]*x[k]**i), вектор правих частин."""
    return np.asarray([np.sum(y * x ** i) for i in range(m + 1)], dtype=float)


def gauss_solve(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Метод Гауса з вибором найбільшого |a[i,k]| у поточному стовпці."""
    a = np.array(matrix, dtype=float, copy=True)
    b = np.array(vector, dtype=float, copy=True)
    n = len(b)
    if a.shape != (n, n):
        raise ValueError("Матриця повинна бути квадратною і узгодженою з b.")
    tolerance = np.finfo(float).eps * n * np.linalg.norm(a, ord=np.inf)
    for k in range(n):
        pivot = k + int(np.argmax(np.abs(a[k:, k])))
        if abs(a[pivot, k]) <= tolerance:
            raise ValueError("Система вироджена або чисельно нестійка.")
        if pivot != k:
            a[[k, pivot]] = a[[pivot, k]]
            b[[k, pivot]] = b[[pivot, k]]
        for i in range(k + 1, n):
            factor = a[i, k] / a[k, k]
            a[i, k:] -= factor * a[k, k:]
            b[i] -= factor * b[k]
    solution = np.zeros(n)
    for i in range(n - 1, -1, -1):
        solution[i] = (b[i] - np.dot(a[i, i + 1:], solution[i + 1:])) / a[i, i]
    return solution


def polynomial(x: np.ndarray | float, coef: np.ndarray) -> np.ndarray:
    """Обчислити c[0]+c[1]*x+... за схемою Горнера."""
    values = np.asarray(x, dtype=float)
    result = np.zeros_like(values)
    for coefficient in reversed(coef):
        result = result * values + coefficient
    return result


def approximation_error(y: np.ndarray, y_approx: np.ndarray) -> np.ndarray:
    """Абсолютна похибка у відомих вузлах: |y-P(x)|."""
    return np.abs(y - y_approx)


def variance(y: np.ndarray, y_approx: np.ndarray) -> float:
    """D = MSE = (1/N)*sum((y[k]-P(x[k]))**2), одиниці °C²."""
    return float(np.mean((y - y_approx) ** 2))


def save_csv(path: Path, columns: list[str], rows) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def calculate(data_path: Path) -> dict:
    """Обчислення; готові бібліотечні МНК-розв'язувачі не використовуються."""
    x, y = read_data(data_path)
    center = float((x[0] + x[-1]) / 2)
    scale = float((x[-1] - x[0]) / 2)
    t = (x - center) / scale
    # У формулі h1=(xn-x0)/(20n) беремо n=24, як у тексті завдання.
    # Вхідні 24 значення мають реальний крок один місяць (23 проміжки).
    grid = np.linspace(x[0], x[-1], 20 * len(x) + 1)
    grid_t = (grid - center) / scale
    interpolated_reference = np.interp(grid, x, y)
    future = np.arange(x[-1] + 1, x[-1] + 4)
    models = []
    for m in range(1, 11):
        matrix = form_matrix(t, m)
        vector = form_vector(t, y, m)
        coef = gauss_solve(matrix, vector)
        approx = polynomial(t, coef)
        dense_approx = polynomial(grid_t, coef)
        # Лінійна заміна змінної не змінює простір многочленів степеня m.
        poly_x = np.polynomial.Polynomial(coef)(
            np.polynomial.Polynomial([-center / scale, 1 / scale]))
        mse = variance(y, approx)
        models.append({
            "degree": m, "coef_t": coef.tolist(), "coef_x": poly_x.coef.tolist(),
            "mse": mse, "rmse": float(np.sqrt(mse)), "sse": float(mse * len(x)),
            "max_node_error": float(approximation_error(y, approx).max()),
            "condition_normal_matrix": float(np.linalg.cond(matrix)),
            "normal_equations_residual_inf": float(np.max(np.abs(matrix @ coef - vector))),
            "node_approx": approx.tolist(), "node_signed_residual": (y - approx).tolist(),
            "node_abs_error": approximation_error(y, approx).tolist(),
            "dense_approx": dense_approx.tolist(),
            "dense_abs_error_vs_linear_interpolant": np.abs(interpolated_reference - dense_approx).tolist(),
            "forecast": polynomial((future - center) / scale, coef).tolist(),
        })
    optimal = min(models, key=lambda item: item["mse"])
    return {"lab": 3, "source": data_path.name, "months": x.tolist(), "temperatures": y.tolist(),
            "N": len(x), "center": center, "scale": scale,
            "grid": grid.tolist(), "grid_step": float(grid[1] - grid[0]),
            "grid_reference": "piecewise-linear interpolation of input CSV; not unknown true temperature",
            "grid_reference_values": interpolated_reference.tolist(),
            "future_months": future.tolist(), "models": models,
            "optimal_degree": optimal["degree"],
            "variance_definition": "MSE = SSE/N; delta = RMSE = sqrt(MSE)"}


def save_outputs(result: dict, output_dir: Path) -> None:
    processed = output_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    models = result["models"]
    save_csv(processed / "variances.csv", ["degree", "MSE_C2", "RMSE_C", "SSE_C2", "max_node_error_C"],
             [[m["degree"], m["mse"], m["rmse"], m["sse"], m["max_node_error"]] for m in models])
    save_csv(processed / "node_errors.csv", ["Month", "Temp"] + [f"P_m{m['degree']}" for m in models]
             + [f"abs_error_m{m['degree']}" for m in models],
             [[x, y] + [m["node_approx"][i] for m in models] + [m["node_abs_error"][i] for m in models]
              for i, (x, y) in enumerate(zip(result["months"], result["temperatures"]))])
    save_csv(processed / "dense_errors.csv", ["Month", "LinearReference"]
             + [f"P_m{m['degree']}" for m in models]
             + [f"abs_error_vs_linear_m{m['degree']}" for m in models],
             [[x, result["grid_reference_values"][i]] + [m["dense_approx"][i] for m in models]
              + [m["dense_abs_error_vs_linear_interpolant"][i] for m in models]
              for i, x in enumerate(result["grid"])])
    save_csv(processed / "coefficients.csv", ["degree", "power", "coefficient_t", "coefficient_x"],
             [[m["degree"], k, ct, m["coef_x"][k]] for m in models for k, ct in enumerate(m["coef_t"])])
    save_csv(processed / "forecast.csv", ["Month"] + [f"P_m{m['degree']}_C" for m in models],
             [[x] + [m["forecast"][i] for m in models] for i, x in enumerate(result["future_months"])])
    with (output_dir / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    optimal = models[result["optimal_degree"] - 1]
    lines = ["Лабораторна робота №3. Метод найменших квадратів.",
             "Джерело: 24 значення середньомісячних температур із прикладу завдання.",
             f"Кількість вузлів: {result['N']}; місяці: 1..24; крок: 1 місяць.",
             f"Масштабування: t = (x - {result['center']}) / {result['scale']}.",
             "", " m |     D=MSE, °C² |    delta=RMSE, °C | max|y-P|, °C"]
    lines += [f"{m['degree']:2d} | {m['mse']:14.9f} | {m['rmse']:17.9f} | {m['max_node_error']:12.9f}" for m in models]
    lines += ["", f"Оптимальний степінь за мінімумом D: m = {optimal['degree']}.",
              "Коефіцієнти P(t) у порядку зростання степеня:"]
    lines += [f"c[{i}] = {value:.15g}" for i, value in enumerate(optimal["coef_t"])]
    lines += ["", "Коефіцієнти P(x) у порядку зростання степеня:"]
    lines += [f"a[{i}] = {value:.15g}" for i, value in enumerate(optimal["coef_x"])]
    lines += ["", "Місяць | Фактична T | P10(x) | y-P10(x) | |y-P10(x)|"]
    lines += [f"{x:6.0f} | {y:10.3f} | {optimal['node_approx'][i]:10.6f} | {optimal['node_signed_residual'][i]:10.6f} | {optimal['node_abs_error'][i]:10.6f}"
              for i, (x, y) in enumerate(zip(result["months"], result["temperatures"]))]
    lines += ["", f"Табуляція на [1,24]: 481 точка, h1 = 23/480 = {result['grid_step']:.12f} місяця.",
              "Між вузлами невідома справжня функція: у dense_errors.csv використано відхилення",
              "від кусково-лінійної інтерполяції вхідних даних, а не істинну похибку температури.",
              "", "Формальна екстраполяція P10 на наступні три місяці:"]
    lines += [f"Місяць {x:.0f}: {y:.9f} °C" for x, y in zip(result["future_months"], optimal["forecast"])]
    lines += ["Ці значення є математичною екстраполяцією. Поліном високого степеня",
              "поза даними нестійкий і не забезпечує надійного метеорологічного прогнозу.",
              "Мінімальна навчальна MSE не гарантує найкращої якості прогнозу."]
    text = "\n".join(lines) + "\n"
    (output_dir / "results.txt").write_text(text, encoding="utf-8")
    print(text)


def make_plots(result: dict, output_dir: Path, show: bool) -> None:
    if not show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11,
                         "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    x, y, grid = result["months"], result["temperatures"], result["grid"]
    models = result["models"]
    optimal = models[result["optimal_degree"] - 1]
    colors = {2: "#737373", 6: "#b06a1c", 10: "#245d83"}

    fig, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
    ax.plot(x, y, "o-", color="#222222", lw=0.9, ms=4, label="Вхідні дані")
    for degree in (2, 6, 10):
        ax.plot(grid, models[degree - 1]["dense_approx"], lw=1.6,
                color=colors[degree], label=f"МНК, m = {degree}")
    ax.set(xlabel="Місяць", ylabel="Температура, °C", xlim=(1, 24), xticks=np.arange(1, 25))
    ax.legend(loc="lower right", ncol=2, fontsize=10)
    fig.savefig(image_dir / "approximation.png")

    fig, ax = plt.subplots(figsize=(8.0, 3.8), constrained_layout=True)
    degrees = [m["degree"] for m in models]
    ax.plot(degrees, [m["mse"] for m in models], "o-", color="#245d83", lw=1.8)
    ax.scatter([optimal["degree"]], [optimal["mse"]], s=110, marker="*", color="#9e352e", zorder=3)
    ax.annotate(f"m = {optimal['degree']}; D = {optimal['mse']:.6f}",
                (optimal["degree"], optimal["mse"]), xytext=(-145, 27),
                textcoords="offset points", arrowprops={"arrowstyle": "->", "color": "#9e352e"},
                fontsize=11, color="#9e352e")
    ax.set(xlabel="Степінь многочлена m", ylabel="D = MSE, °C²", xticks=degrees)
    fig.savefig(image_dir / "variance.png")

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.7), constrained_layout=True, sharex=True)
    for ax, group in zip(axes, (models[:5], models[5:])):
        for m in group:
            ax.plot(grid, m["dense_abs_error_vs_linear_interpolant"], lw=1.1, label=f"m = {m['degree']}")
        ax.set(ylabel="|L(x) - Pₘ(x)|, °C", xlim=(1, 24))
        ax.legend(ncol=5, loc="upper right", fontsize=9)
    axes[1].set(xlabel="Місяць", xticks=np.arange(1, 25))
    fig.savefig(image_dir / "errors_all_degrees.png")

    fig, ax = plt.subplots(figsize=(8.0, 3.8), constrained_layout=True)
    ax.bar(x, optimal["node_abs_error"], color="#245d83", width=0.7)
    ax.set(xlabel="Місяць", ylabel="|yᵢ - P₁₀(xᵢ)|, °C", xticks=np.arange(1, 25), xlim=(0.3, 24.7))
    fig.savefig(image_dir / "node_errors.png")

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.4), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.2, 1]})
    axes[0].plot(x, y, "o-", color="#222222", ms=3, lw=0.8, label="Вхідні дані")
    axes[0].plot(grid, optimal["dense_approx"], color="#245d83", label="P₁₀ на [1, 24]")
    axes[0].set(xlim=(1, 24), ylabel="Температура, °C", xticks=np.arange(1, 25))
    axes[0].legend(fontsize=10, ncol=2)
    forecast_grid = np.linspace(24, 27, 200)
    forecast_values = polynomial((forecast_grid - result["center"]) / result["scale"], np.array(optimal["coef_t"]))
    axes[1].plot(forecast_grid, forecast_values, "--", color="#9e352e", label="Екстраполяція P₁₀")
    axes[1].scatter(result["future_months"], optimal["forecast"], color="#9e352e", zorder=3)
    for month, value in zip(result["future_months"], optimal["forecast"]):
        axes[1].annotate(f"{value:.2f} °C", (month, value), xytext=(-5, 7), textcoords="offset points",
                         ha="right", fontsize=10)
    axes[1].set(xlabel="Місяць", ylabel="Значення P₁₀, °C", xlim=(23.9, 27.2), xticks=[24, 25, 26, 27])
    axes[1].legend(loc="lower left", fontsize=10)
    fig.savefig(image_dir / "forecast.png")
    if show:
        plt.show()
    else:
        plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Зберегти графіки без відкриття вікон")
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data.csv", help="CSV із Month,Temp")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR, help="Папка результатів")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = calculate(args.data)
    save_outputs(result, args.output_dir)
    make_plots(result, args.output_dir, show=not args.no_show)


if __name__ == "__main__":
    main()
