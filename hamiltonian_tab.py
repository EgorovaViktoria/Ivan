from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from algorithms import (
    MAX_HAMILTONIAN_EXACT_CELLS,
    MAX_TIME_LIMIT_S,
    HamiltonianOptions,
    PathCountStats,
    SolverStats,
    count_hamiltonian_paths,
    solve_hamiltonian_path,
)

HAMILTONIAN_CONTROL_COLUMNS = 9
HAMILTONIAN_NOTE_COLUMNS = 7
PROCESS_WORKERS = 4
# HamiltonianTask:
#   kind: "solve" | "count"
#   display_name: подпись для таблицы
#   rows, cols: размеры сетки
#   start, finish: координаты
#   options: параметры алгоритма
HamiltonianTask = Tuple[str, str, int, int, Tuple[int, int], Tuple[int, int], HamiltonianOptions]

_PROCESS_POOL: Optional[ProcessPoolExecutor] = None


def _get_process_pool() -> ProcessPoolExecutor:
    global _PROCESS_POOL
    if _PROCESS_POOL is None:
        _PROCESS_POOL = ProcessPoolExecutor(max_workers=PROCESS_WORKERS)
    return _PROCESS_POOL


def shutdown_hamiltonian_pool() -> None:
    global _PROCESS_POOL
    if _PROCESS_POOL is not None:
        _PROCESS_POOL.shutdown(wait=False)
        _PROCESS_POOL = None


def _run_hamiltonian_task(task: HamiltonianTask) -> Tuple[str, str, SolverStats | PathCountStats]:
    # task: (kind, display_name, rows, cols, start, finish, options)
    kind, name, rows, cols, start, finish, options = task
    if kind == "count":
        return kind, name, count_hamiltonian_paths(rows, cols, start, finish, options)
    return kind, name, solve_hamiltonian_path(rows, cols, start, finish, options)


class HamiltonianWorker(QtCore.QThread):
    finishedWithResults = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, tasks: List[HamiltonianTask]) -> None:
        super().__init__()
        self.tasks = tasks

    def run(self) -> None:
        pool_error_message = ""
        try:
            results = list(_get_process_pool().map(_run_hamiltonian_task, self.tasks))
        except Exception as exc:
            pool_error_message = f"{type(exc).__name__}: {exc}"
            try:
                results = [_run_hamiltonian_task(task) for task in self.tasks]
            except Exception as exc:
                self.failed.emit(str(exc))
                return
            for _, result in results:
                if result.note:
                    result.note = f"{result.note}; выполнено без пула процессов ({pool_error_message})"
                else:
                    result.note = f"Выполнено без пула процессов ({pool_error_message})"
        try:
            self.finishedWithResults.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))


class HamiltonianTab(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QtWidgets.QVBoxLayout(self)

        controls = QtWidgets.QGridLayout()
        self.rows_input = QtWidgets.QSpinBox()
        self.rows_input.setRange(1, 999)
        self.rows_input.setValue(5)

        self.cols_input = QtWidgets.QSpinBox()
        self.cols_input.setRange(1, 999)
        self.cols_input.setValue(5)

        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItems(["Поставить старт", "Поставить финиш"])

        self.algorithms_label = QtWidgets.QLabel(
            "Алгоритмы: базовый, Варнсдорф, связность, бэкджампинг, комбинация"
        )

        self.time_limit_input = QtWidgets.QDoubleSpinBox()
        self.time_limit_input.setRange(1.0, MAX_TIME_LIMIT_S)
        self.time_limit_input.setValue(120.0)
        self.time_limit_input.setSingleStep(10.0)

        self.build_btn = QtWidgets.QPushButton("Применить размер")
        self.run_btn = QtWidgets.QPushButton("Запустить алгоритмы")
        self.status_label = QtWidgets.QLabel("Готово к запуску.")
        self._worker: Optional[HamiltonianWorker] = None

        controls.addWidget(QtWidgets.QLabel("N:"), 0, 0)
        controls.addWidget(self.rows_input, 0, 1)
        controls.addWidget(QtWidgets.QLabel("M:"), 0, 2)
        controls.addWidget(self.cols_input, 0, 3)
        controls.addWidget(QtWidgets.QLabel("Режим клика:"), 0, 4)
        controls.addWidget(self.mode_box, 0, 5)
        controls.addWidget(self.build_btn, 0, 6)

        controls.addWidget(self.algorithms_label, 1, 0, 1, 6)
        controls.addWidget(QtWidgets.QLabel("Лимит времени (с):"), 1, 6)
        controls.addWidget(self.time_limit_input, 1, 7)
        controls.addWidget(self.run_btn, 1, 8)
        controls.addWidget(
            QtWidgets.QLabel(f"Примечание: точный перебор выполняется для сеток до {MAX_HAMILTONIAN_EXACT_CELLS} клеток."),
            2,
            0,
            1,
            HAMILTONIAN_NOTE_COLUMNS,
        )
        controls.addWidget(
            self.status_label,
            2,
            HAMILTONIAN_NOTE_COLUMNS,
            1,
            HAMILTONIAN_CONTROL_COLUMNS - HAMILTONIAN_NOTE_COLUMNS,
        )

        root.addLayout(controls)

        self.grid = QtWidgets.QTableWidget(5, 5)
        self.grid.horizontalHeader().setVisible(False)
        self.grid.verticalHeader().setVisible(False)
        self.grid.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.grid.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.grid.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.grid.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.grid.setMinimumHeight(620)
        root.addWidget(self.grid, 2)

        self.results = QtWidgets.QTableWidget(0, 7)
        self.results.setHorizontalHeaderLabels(
            ["Алгоритм", "Найден", "Длина пути", "Время (мс)", "Операции", "Память (байт)", "Примечание"]
        )
        self.results.horizontalHeader().setStretchLastSection(True)
        self.results.setMinimumHeight(250)
        root.addWidget(self.results, 1)

        self.paths_count_label = QtWidgets.QLabel("Количество путей: -")
        self.paths_count_label.setWordWrap(True)
        root.addWidget(self.paths_count_label)

        self.start = (0, 0)
        self.finish = (4, 4)

        self.build_btn.clicked.connect(self.rebuild_grid)
        self.grid.cellClicked.connect(self.handle_cell_click)
        self.run_btn.clicked.connect(self.run_algorithms)

        self.rebuild_grid()

    def rebuild_grid(self) -> None:
        rows = self.rows_input.value()
        cols = self.cols_input.value()
        self.grid.setRowCount(rows)
        self.grid.setColumnCount(cols)
        self.start = (0, 0)
        self.finish = (rows - 1, cols - 1)
        for r in range(rows):
            self.grid.setRowHeight(r, max(28, 560 // rows))
            for c in range(cols):
                if self.grid.item(r, c) is None:
                    self.grid.setItem(r, c, QtWidgets.QTableWidgetItem(""))
        self.repaint_grid([])

    def handle_cell_click(self, row: int, col: int) -> None:
        if self.mode_box.currentIndex() == 0:
            self.start = (row, col)
        else:
            self.finish = (row, col)
        if self.start == self.finish:
            if self.mode_box.currentIndex() == 0:
                self.finish = (self.grid.rowCount() - 1, self.grid.columnCount() - 1)
            else:
                self.start = (0, 0)
        self.repaint_grid([])

    def repaint_grid(self, path: List[Tuple[int, int]]) -> None:
        path_set = set(path)
        path_index = {cell: idx for idx, cell in enumerate(path)}
        path_len = len(path)
        for r in range(self.grid.rowCount()):
            for c in range(self.grid.columnCount()):
                item = self.grid.item(r, c)
                item.setText("")
                if (r, c) == self.start:
                    item.setBackground(QtGui.QColor(80, 200, 120))
                    item.setText("S")
                elif (r, c) == self.finish:
                    item.setBackground(QtGui.QColor(220, 70, 70))
                    item.setText("F")
                elif (r, c) in path_set:
                    idx = path_index[(r, c)]
                    ratio = 0.0 if path_len <= 1 else idx / (path_len - 1)
                    start_color = (110, 170, 255)
                    end_color = (176, 120, 255)
                    red = int(start_color[0] + (end_color[0] - start_color[0]) * ratio)
                    green = int(start_color[1] + (end_color[1] - start_color[1]) * ratio)
                    blue = int(start_color[2] + (end_color[2] - start_color[2]) * ratio)
                    item.setBackground(QtGui.QColor(red, green, blue))
                else:
                    item.setBackground(QtGui.QColor(245, 245, 245))

    def run_algorithms(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        rows = self.grid.rowCount()
        cols = self.grid.columnCount()
        limit = self.time_limit_input.value()

        runs = [
            ("Базовый перебор", HamiltonianOptions(False, False, False, limit)),
            ("Варнсдорф", HamiltonianOptions(True, False, False, limit)),
            ("Связность", HamiltonianOptions(False, True, False, limit)),
            ("Бэкджампинг", HamiltonianOptions(False, False, True, limit)),
            ("Комбинация", HamiltonianOptions(True, True, True, limit)),
        ]
        count_options = HamiltonianOptions(True, True, False, limit)

        self.results.setRowCount(0)
        self.run_btn.setEnabled(False)
        self.status_label.setText("Идёт вычисление...")
        self.paths_count_label.setText("Количество путей: идёт подсчёт...")
        tasks = [
            ("solve", name, rows, cols, self.start, self.finish, options) for name, options in runs
        ] + [("count", "Количество путей", rows, cols, self.start, self.finish, count_options)]
        self._worker = HamiltonianWorker(tasks)
        self._worker.finishedWithResults.connect(self._handle_results)
        self._worker.failed.connect(self._handle_error)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _handle_results(self, rows_data: List[Tuple[str, str, SolverStats | PathCountStats]]) -> None:
        solve_rows: List[Tuple[str, SolverStats]] = []
        count_result: Optional[PathCountStats] = None
        for kind, name, result in rows_data:
            if kind == "count":
                count_result = result if isinstance(result, PathCountStats) else None
            else:
                if isinstance(result, SolverStats):
                    solve_rows.append((name, result))
        self.results.setRowCount(len(solve_rows))
        best_path: List[Tuple[int, int]] = []
        for idx, (name, result) in enumerate(solve_rows):
            if result.found and not best_path:
                best_path = list(result.path)
            values = [
                name,
                "Да" if result.found else "Нет",
                str(len(result.path)),
                f"{result.elapsed_ms:.3f}",
                str(result.operations),
                str(result.peak_memory_bytes),
                result.note,
            ]
            for c_idx, val in enumerate(values):
                self.results.setItem(idx, c_idx, QtWidgets.QTableWidgetItem(val))

        if count_result is not None:
            label = (
                f"Количество путей: {count_result.count} "
                f"(время {count_result.elapsed_ms:.3f} мс, память {count_result.peak_memory_bytes} байт)"
            )
            if count_result.note:
                label = f"{label}\nПримечание: {count_result.note}"
            self.paths_count_label.setText(label)
        else:
            self.paths_count_label.setText("Количество путей: нет данных.")

        self.run_btn.setEnabled(True)
        self.status_label.setText("Вычисление завершено.")
        self.repaint_grid(best_path)
        self._worker = None

    def _handle_error(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self.status_label.setText("Ошибка вычисления.")
        self.paths_count_label.setText("Количество путей: ошибка.")
        QtWidgets.QMessageBox.critical(self, "Ошибка", message)
        self._worker = None
