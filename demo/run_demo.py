import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PyQt5 import QtWidgets
from demo.cube_widget import DeepCubeDemoWindow


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("DeepCubeA Rubik Demo")
    window = DeepCubeDemoWindow(repo_root=REPO_ROOT)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
