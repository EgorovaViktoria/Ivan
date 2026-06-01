import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    # Опциональная зависимость для построения графиков.
    plt = None

from algorithms import (
    HamiltonianOptions,
    SolverStats,
    random_solvable_state,
    reset_heuristic_cache,
    solve_hamiltonian_path,
    solve_puzzle_astar,
    solve_puzzle_backjumping,
    solve_puzzle_bfs,
    solve_puzzle_ida,
    solve_puzzle_manhattan_greedy,
)


@dataclass
class BenchmarkRecord:
    task: str
    case_id: str
    algorithm: str
    found: bool
    time_ms: float
    memory_bytes: int
    operations: int
    note: str


def _write_csv(path: Path, rows: Iterable[BenchmarkRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["Задача", "Сценарий", "Алгоритм", "Найден", "Время_мс", "Память_байт", "Операции", "Примечание"]
        )
        for row in rows:
            writer.writerow(
                [
                    row.task,
                    row.case_id,
                    row.algorithm,
                    "Да" if row.found else "Нет",
                    f"{row.time_ms:.3f}",
                    row.memory_bytes,
                    row.operations,
                    row.note,
                ]
            )


def _summarize(records: Iterable[BenchmarkRecord], key: str) -> List[Tuple[str, float]]:
    values: dict[str, List[float]] = {}
    for record in records:
        values.setdefault(record.algorithm, []).append(getattr(record, key))
    return [(algo, statistics.fmean(vals)) for algo, vals in values.items()]


def _plot_bars(title: str, data: List[Tuple[str, float]], output_path: Path, ylabel: str) -> None:
    if plt is None:
        print("Matplotlib не установлен — построение графиков пропущено.")
        return
    labels = [item[0] for item in data]
    values = [item[1] for item in data]
    colors = []
    cmap = plt.get_cmap("tab10")
    for idx in range(len(labels)):
        colors.append(cmap(idx % 10))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Алгоритм")
    ax.grid(axis="y", linestyle="--", color="#999999", alpha=0.6)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _print_progress(task_name: str, done: int, total: int, start_time: float) -> None:
    elapsed = time.perf_counter() - start_time
    avg = elapsed / max(1, done)
    remaining = avg * (total - done)
    print(f"{task_name}: {done}/{total} (≈{remaining:.1f} с осталось)")


def run_hamiltonian_cases(
    cases: List[Tuple[str, int, int, Tuple[int, int], Tuple[int, int]]],
    time_limit_s: float,
) -> List[BenchmarkRecord]:
    algorithms = [
        ("Базовый перебор", HamiltonianOptions(False, False, False, time_limit_s)),
        ("Варнсдорф", HamiltonianOptions(True, False, False, time_limit_s)),
        ("Связность", HamiltonianOptions(False, True, False, time_limit_s)),
        ("Бэкджампинг", HamiltonianOptions(False, False, True, time_limit_s)),
        ("Комбинация", HamiltonianOptions(True, True, True, time_limit_s)),
    ]
    records: List[BenchmarkRecord] = []
    total = len(cases) * len(algorithms)
    done = 0
    start_time = time.perf_counter()
    for case_id, rows, cols, start, finish in cases:
        for name, options in algorithms:
            stats = solve_hamiltonian_path(rows, cols, start, finish, options)
            records.append(
                BenchmarkRecord(
                    "Сетка",
                    case_id,
                    name,
                    stats.found,
                    stats.elapsed_ms,
                    stats.peak_memory_bytes,
                    stats.operations,
                    stats.note,
                )
            )
            done += 1
            _print_progress("Сетка", done, total, start_time)
    return records


def run_puzzle_cases(
    cases: List[Tuple[str, int, int]],
    time_limit_s: float,
    astar_limit: int,
    depth_limit: int,
) -> List[BenchmarkRecord]:
    algorithms: List[Tuple[str, Callable[[Tuple[int, ...]], SolverStats]]] = [
        ("A*", lambda state: solve_puzzle_astar(state, astar_limit, time_limit_s)),
        ("Манхэттенский", lambda state: solve_puzzle_manhattan_greedy(state, time_limit_s)),
        ("BFS", lambda state: solve_puzzle_bfs(state, time_limit_s)),
        ("IDA*", lambda state: solve_puzzle_ida(state, depth_limit, time_limit_s)),
        ("Бэкджампинг", lambda state: solve_puzzle_backjumping(state, depth_limit, time_limit_s)),
    ]
    records: List[BenchmarkRecord] = []
    total = len(cases) * len(algorithms)
    done = 0
    start_time = time.perf_counter()
    for case_id, seed, shuffle_steps in cases:
        state = random_solvable_state(seed, shuffle_steps)
        for name, runner in algorithms:
            reset_heuristic_cache()
            stats = runner(state)
            records.append(
                BenchmarkRecord(
                    "Пятнашки",
                    case_id,
                    name,
                    stats.found,
                    stats.elapsed_ms,
                    stats.peak_memory_bytes,
                    stats.operations,
                    stats.note,
                )
            )
            done += 1
            _print_progress("Пятнашки", done, total, start_time)
    return records


def build_default_cases() -> Tuple[
    List[Tuple[str, int, int, Tuple[int, int], Tuple[int, int]]], List[Tuple[str, int, int]]
]:
    def generate_hamiltonian_test_cases(
        rows: int, cols: int, count: int
    ) -> List[Tuple[str, int, int, Tuple[int, int], Tuple[int, int]]]:
        cells = [(r, c) for r in range(rows) for c in range(cols)]
        even_sum_cells = [cell for cell in cells if (cell[0] + cell[1]) % 2 == 0]
        if (rows * cols) % 2 == 1:
            pairs = [
                (start, finish)
                for start in even_sum_cells
                for finish in even_sum_cells
                if start != finish
            ]
        else:
            odd_sum_cells = [cell for cell in cells if (cell[0] + cell[1]) % 2 == 1]
            pairs = [(start, finish) for start in even_sum_cells for finish in odd_sum_cells]
        if count > len(pairs):
            raise ValueError(
                f"Недостаточно пар для {rows}x{cols}: запрошено {count}, доступно {len(pairs)}"
            )
        width = len(str(count))
        return [
            (f"{rows}x{cols}-{idx:0{width}d}", rows, cols, start, finish)
            for idx, (start, finish) in enumerate(pairs[:count], start=1)
        ]

    hamiltonian_case_plan = [
        (7, 7, 2),
        (6, 6, 6),
        (5, 5, 40),
    ]
    hamiltonian_cases: List[Tuple[str, int, int, Tuple[int, int], Tuple[int, int]]] = []
    for rows, cols, count in hamiltonian_case_plan:
        hamiltonian_cases += generate_hamiltonian_test_cases(rows, cols, count)
    puzzle_cases = [
        ("seed-17", 17, 60),
        ("seed-42", 42, 80),
        ("seed-99", 99, 100),
        ("seed-123", 123, 120),
        ("seed-256", 256, 140),
        ("seed-512", 512, 160),
    ]
    return hamiltonian_cases, puzzle_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Сценарий исследования алгоритмов для Algo2/Fourth.")
    parser.add_argument("--output", default="benchmarks", help="Папка для таблиц и графиков.")
    parser.add_argument("--time-limit", type=float, default=1200.0, help="Лимит времени на запуск (сек).")
    parser.add_argument("--astar-limit", type=int, default=1200000, help="Лимит состояний для A*.")
    parser.add_argument("--depth-limit", type=int, default=120, help="Лимит глубины для IDA*/бэкджампинга.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    hamiltonian_cases, puzzle_cases = build_default_cases()

    hamiltonian_records = run_hamiltonian_cases(hamiltonian_cases, args.time_limit)
    puzzle_records = run_puzzle_cases(puzzle_cases, args.time_limit, args.astar_limit, args.depth_limit)

    _write_csv(output_dir / "hamiltonian_results.csv", hamiltonian_records)
    _write_csv(output_dir / "puzzle_results.csv", puzzle_records)

    h_time = _summarize(hamiltonian_records, "time_ms")
    h_mem = _summarize(hamiltonian_records, "memory_bytes")
    p_time = _summarize(puzzle_records, "time_ms")
    p_mem = _summarize(puzzle_records, "memory_bytes")

    _plot_bars("Время работы (сетка)", h_time, output_dir / "hamiltonian_time.png", "мс")
    _plot_bars("Память (сетка)", h_mem, output_dir / "hamiltonian_memory.png", "байт")
    _plot_bars("Время работы (пятнашки)", p_time, output_dir / "puzzle_time.png", "мс")
    _plot_bars("Память (пятнашки)", p_mem, output_dir / "puzzle_memory.png", "байт")

    print("Готово. Таблицы и графики сохранены в:", output_dir.resolve())


if __name__ == "__main__":
    main()
