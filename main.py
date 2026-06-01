import os
import sys

import PyQt5
from PyQt5 import QtWidgets

from hamiltonian_tab import HamiltonianTab, shutdown_hamiltonian_pool
from puzzle_tab import PuzzleTab, shutdown_puzzle_pool

plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
if os.path.isdir(plugin_path):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Algo2/Fourth — перебор, отсечения и эвристики")
        self.resize(1520, 1080)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(HamiltonianTab(), "Сетка: путь через все клетки")
        tabs.addTab(PuzzleTab(), "15-пазл")
        self.setCentralWidget(tabs)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #e9edff; }
            QWidget { font-size: 15px; }
            QTabWidget::pane {
                border: 1px solid #c7d2fe;
                border-radius: 12px;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #dbe4ff;
                border: 1px solid #c7d2fe;
                padding: 10px 16px;
                margin-right: 6px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #4f46e5;
                color: white;
            }
            QPushButton {
                background: #2563eb;
                color: white;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover { background: #1d4ed8; }
            QPushButton:disabled { background: #94a3b8; }
            QTableWidget, QPlainTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 4px;
            }
            QHeaderView::section {
                background: #e2e8f0;
                border: none;
                padding: 7px;
                font-weight: 600;
            }
            QLabel { color: #0f172a; }
            """
        )


def _shutdown_all_pools() -> None:
    shutdown_hamiltonian_pool()
    shutdown_puzzle_pool()


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.aboutToQuit.connect(_shutdown_all_pools)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
