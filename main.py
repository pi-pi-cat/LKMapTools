import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from lkmap.app.controller import AppController
from lkmap.app.window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("LKMapTools")

    window = MainWindow()

    try:
        controller = AppController(window)
    except Exception as exc:
        QMessageBox.critical(None, "启动失败", str(exc))
        return 1

    app.aboutToQuit.connect(controller.shutdown)
    window.show()
    QTimer.singleShot(0, controller.bootstrap)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
