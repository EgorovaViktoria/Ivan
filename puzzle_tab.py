import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from algorithms import (
    GOAL_STATE,
    MAX_TIME_LIMIT_S,
    SolverStats,
    is_solvable,
    random_solvable_state,
    solve_puzzle_astar,
    solve_puzzle_backjumping,
    solve_puzzle_bfs,
    solve_puzzle_ida,
    solve_puzzle_manhattan_greedy,
)

MAX_PREVIEW_MOVES = 80
PUZZLE_CONTROL_COLUMNS = 11

def _timestamp_seed() -> int:
    return int(time.time_ns())


def shutdown_puzzle_pool() -> None:
    return None


def _run_puzzle_task(task: Tuple[str, str, Tuple[int, ...], int, int, float]) -> Tuple[str, SolverStats]:
    name, solver_id, state, astar_limit, depth_limit, time_limit = task
    if solver_id == "astar":
        result = solve_puzzle_astar(state, astar_limit, time_limit)
    elif solver_id == "greedy":
        result = solve_puzzle_manhattan_greedy(state, time_limit_s=time_limit)
    elif solver_id == "bfs":
        result = solve_puzzle_bfs(state, time_limit_s=time_limit)
    elif solver_id == "ida":
        result = solve_puzzle_ida(state, depth_limit, time_limit)
    elif solver_id == "backjump":
        result = solve_puzzle_backjumping(state, depth_limit, time_limit)
    else:
        raise ValueError(f"Неизвестный алгоритм: {solver_id}")
    return name, result


class PuzzleWorker(QtCore.QThread):
    finishedWithResults = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, tasks: List[Tuple[str, str, Tuple[int, ...], int, int, float]]) -> None:
        super().__init__()
        self.tasks = tasks

    def run(self) -> None:
        pool_error_message = ""
        try:
            results = []
            for task in self.tasks:
                with ProcessPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(_run_puzzle_task, task).result()
                results.append(result)
        except Exception as exc:
            pool_error_message = f"{type(exc).__name__}: {exc}"
            try:
                results = [_run_puzzle_task(task) for task in self.tasks]
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


class PuzzleNumberDelegate(QtWidgets.QStyledItemDelegate):
    def createEditor(self, parent, option, index):  
        editor = QtWidgets.QSpinBox(parent)
        editor.setRange(0, 15)
        return editor

    def setEditorData(self, editor, index):  
        text = index.data() or "0"
        if text == "":
            text = "0"
        editor.setValue(int(text))

    def setModelData(self, editor, model, index):  
        value = editor.value()
        model.setData(index, "" if value == 0 else str(value))


class PuzzleTab(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QtWidgets.QVBoxLayout(self)

        controls = QtWidgets.QGridLayout()
        self.seed_input = QtWidgets.QLineEdit(str(_timestamp_seed()))
        self.seed_input.setPlaceholderText("Введите seed (целое число)")

        self.shuffle_input = QtWidgets.QSpinBox()
        self.shuffle_input.setRange(5, 400)
        self.shuffle_input.setValue(70)

        self.algorithms_label = QtWidgets.QLabel("Алгоритмы: A*, Манхэттен, BFS, IDA*, Бэкджампинг")

        self.astar_limit_input = QtWidgets.QSpinBox()
        self.astar_limit_input.setRange(100000, 20000000)
        self.astar_limit_input.setSingleStep(250000)
        self.astar_limit_input.setValue(6000000)

        self.depth_limit_input = QtWidgets.QSpinBox()
        self.depth_limit_input.setRange(5, 10000000)
        self.depth_limit_input.setValue(90)

        self.goal_btn = QtWidgets.QPushButton("Эталон")
        self.random_btn = QtWidgets.QPushButton("Случайная")
        self.seed_btn = QtWidgets.QPushButton("Новый seed")
        self.run_btn = QtWidgets.QPushButton("Решить")
        self.status_label = QtWidgets.QLabel("Готово к запуску.")
        self._worker: Optional[PuzzleWorker] = None

        self.time_limit_input = QtWidgets.QDoubleSpinBox()
        self.time_limit_input.setRange(5.0, MAX_TIME_LIMIT_S)
        self.time_limit_input.setValue(120.0)
        self.time_limit_input.setSingleStep(10.0)

        controls.addWidget(QtWidgets.QLabel("Seed:"), 0, 0)
        controls.addWidget(self.seed_input, 0, 1)
        controls.addWidget(QtWidgets.QLabel("Шагов перемешивания:"), 0, 2)
        controls.addWidget(self.shuffle_input, 0, 3)
        controls.addWidget(self.seed_btn, 0, 4)
        controls.addWidget(self.goal_btn, 0, 5)
        controls.addWidget(self.random_btn, 0, 6)

        controls.addWidget(self.algorithms_label, 1, 0, 1, 3)
        controls.addWidget(QtWidgets.QLabel("Лимит состояний:"), 1, 3)
        controls.addWidget(self.astar_limit_input, 1, 4)
        controls.addWidget(QtWidgets.QLabel("Лимит глубины IDA/Бэкджампинга:"), 1, 5)
        controls.addWidget(self.depth_limit_input, 1, 6)
        controls.addWidget(QtWidgets.QLabel("Лимит времени (с):"), 1, 7)
        controls.addWidget(self.time_limit_input, 1, 8)
        controls.addWidget(self.run_btn, 0, 10)
        controls.addWidget(self.status_label, 2, 0, 1, PUZZLE_CONTROL_COLUMNS)
        root.addLayout(controls)

        self.board = QtWidgets.QTableWidget(4, 4)
        self.board.horizontalHeader().setVisible(False)
        self.board.verticalHeader().setVisible(False)
        self.board.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.board.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.board.setItemDelegate(PuzzleNumberDelegate(self.board))
        self.board.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.SelectedClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
            | QtWidgets.QAbstractItemView.AnyKeyPressed
        )
        self.board.setMinimumHeight(520)
        for r in range(4):
            self.board.setRowHeight(r, 120)
            for c in range(4):
                self.board.setItem(r, c, QtWidgets.QTableWidgetItem(""))
        root.addWidget(self.board, 2)

        self.results = QtWidgets.QTableWidget(0, 7)
        self.results.setHorizontalHeaderLabels(
            ["Алгоритм", "Найден", "Ходы", "Время (мс)", "Операции", "Память (байт)", "Примечание"]
        )
        self.results.horizontalHeader().setStretchLastSection(True)
        self.results.setMinimumHeight(260)
        root.addWidget(self.results, 1)

        self.moves_preview = QtWidgets.QPlainTextEdit()
        self.moves_preview.setReadOnly(True)
        self.moves_preview.setMinimumHeight(180)
        root.addWidget(self.moves_preview, 1)

        self.goal_btn.clicked.connect(self.load_goal)
        self.random_btn.clicked.connect(self.load_random)
        self.seed_btn.clicked.connect(self.refresh_seed)
        self.run_btn.clicked.connect(self.run_algorithms)

        self.load_goal()

    def refresh_seed(self) -> None:
        self.seed_input.setText(str(_timestamp_seed()))

    def _seed_value(self) -> int:
        text = self.seed_input.text().strip()
        if text.isdigit():
            return int(text)
        seed = _timestamp_seed()
        self.seed_input.setText(str(seed))
        return seed

    def _state_from_table(self) -> Tuple[int, ...] | None:
        values = []
        for r in range(4):
            for c in range(4):
                text = self.board.item(r, c).text().strip()
                if text == "":
                    values.append(0)
                else:
                    if not text.isdigit():
                        return None
                    values.append(int(text))
        if sorted(values) != list(range(16)):
            return None
        return tuple(values)

    def _show_state(self, state: Tuple[int, ...]) -> None:
        for idx, value in enumerate(state):
            r, c = divmod(idx, 4)
            item = self.board.item(r, c)
            item.setText("" if value == 0 else str(value))
            if value == 0:
                item.setBackground(QtGui.QColor(220, 220, 220))
            else:
                item.setBackground(QtGui.QColor(245, 245, 245))

    def load_goal(self) -> None:
        self._show_state(GOAL_STATE)

    def load_random(self) -> None:
        self.refresh_seed()
        state = random_solvable_state(self._seed_value(), self.shuffle_input.value())
        self._show_state(state)

    def run_algorithms(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        state = self._state_from_table()
        if state is None:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Поле должно содержать числа 0..15 без повторов")
            return
        if not is_solvable(state):
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Эта конфигурация неразрешима")
            return

        time_limit = self.time_limit_input.value()
        runs = [
            ("Эталонный", "astar"),
            ("Манхэттенский", "greedy"),
            ("BFS", "bfs"),
            ("IDA*", "ida"),
            ("Бэкджампинг", "backjump"),
        ]

        self.results.setRowCount(0)
        self.moves_preview.clear()
        self.run_btn.setEnabled(False)
        self.status_label.setText("Идёт вычисление...")
        tasks = [
            (
                name,
                solver_id,
                state,
                self.astar_limit_input.value(),
                self.depth_limit_input.value(),
                time_limit,
            )
            for name, solver_id in runs
        ]
        self._worker = PuzzleWorker(tasks)
        self._worker.finishedWithResults.connect(self._handle_results)
        self._worker.failed.connect(self._handle_error)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _handle_results(self, rows_data: List[Tuple[str, SolverStats]]) -> None:
        self.results.setRowCount(len(rows_data))
        best_moves: List[str] = []
        for idx, (name, result) in enumerate(rows_data):
            if result.found and (not best_moves or len(result.path) < len(best_moves)):
                best_moves = list(result.path)
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

        if best_moves:
            preview = " ".join(best_moves[:MAX_PREVIEW_MOVES])
            if len(best_moves) > MAX_PREVIEW_MOVES:
                preview += " ..."
            self.moves_preview.setPlainText(f"Лучшее найденное решение ({len(best_moves)} ходов):\n{preview}")
        else:
            self.moves_preview.setPlainText("Решение не найдено.")

        self.run_btn.setEnabled(True)
        self.status_label.setText("Вычисление завершено.")
        self._worker = None

    def _handle_error(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self.status_label.setText("Ошибка вычисления.")
        QtWidgets.QMessageBox.critical(self, "Ошибка", message)
        self._worker = None
