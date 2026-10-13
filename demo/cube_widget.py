from pathlib import Path
from typing import Dict, List
from PyQt5 import QtCore, QtWidgets

from demo.solver_adapter import DeepCubeASolver


class SolveWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, solver, state, weight, batch_size):
        super().__init__()
        self.solver = solver
        self.state = state
        self.weight = weight
        self.batch_size = batch_size

    @QtCore.pyqtSlot()
    def run(self):
        try:
            result = self.solver.solve(
                self.state,
                weight=self.weight,
                batch_size=self.batch_size
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class DeepCubeDemoWindow(QtWidgets.QMainWindow):
    COLOR_MAP: Dict[int, str] = {
        0: "#F7F7F7",
        1: "#FFD84D",
        2: "#4DFF71",
        3: "#0000FF",
        4: "#FF9F43",
        5: "#EF5350",
    }

    FACE_LABELS = {0:"U", 1:"D", 2:"L", 3:"R", 4:"B", 5:"F"}

    FACE_POSITIONS = {
        0: (0,1),
        2: (1,0),
        5: (1,1),
        3: (1,2),
        4: (1,3),
        1: (2,1),
    }

    MOVE_BUTTONS = [
        ("U",1), ("U'",0), ("D",3), ("D'",2),
        ("L",5), ("L'",4), ("R",7), ("R'",6),
        ("B",9), ("B'",8), ("F",11), ("F'",10),
    ]

    def __init__(self, repo_root: Path):
        super().__init__()
        self.repo_root = Path(repo_root)
        self.setWindowTitle("DeepCubeA - Rubik 3x3 Demo")
        self.resize(1180, 760)

        self.solver = DeepCubeASolver(
            repo_root=self.repo_root,
            weight=0.6,
            batch_size=100,
            nnet_batch_size=1000
        )

        self.state = self.solver.solved_state()
        self.scramble_moves: List[int] = []
        self.solution_moves: List[int] = []
        self.solution_step = 0

        self.solve_thread = None
        self.solve_worker = None

        self.animation_timer = QtCore.QTimer(self)
        self.animation_timer.timeout.connect(self._animate_next_solution_move)

        self._build_ui()
        self._apply_theme()
        self.render_state()

        QtCore.QTimer.singleShot(0, self._load_model)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(20,20,20,20)
        root.setSpacing(22)

        left_card = QtWidgets.QFrame()
        left_card.setObjectName("card")
        left = QtWidgets.QVBoxLayout(left_card)
        left.setContentsMargins(18,18,18,18)

        t = QtWidgets.QLabel("Rubik 3×3")
        t.setObjectName("sectionTitle")
        left.addWidget(t)

        s = QtWidgets.QLabel("Hiển thị trực tiếp từ Cube3State của DeepCubeA")
        s.setObjectName("muted")
        left.addWidget(s)

        cube_wrap = QtWidgets.QWidget()
        cube_net = QtWidgets.QGridLayout(cube_wrap)
        cube_net.setSpacing(10)

        self.sticker_buttons = [None] * 54

        for face_idx, (fr, fc) in self.FACE_POSITIONS.items():
            face_box = QtWidgets.QFrame()
            face_layout = QtWidgets.QVBoxLayout(face_box)
            face_layout.setContentsMargins(0,0,0,0)
            face_layout.setSpacing(5)

            label = QtWidgets.QLabel(self.FACE_LABELS[face_idx])
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setObjectName("faceLabel")
            face_layout.addWidget(label)

            grid_widget = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(grid_widget)
            grid.setSpacing(2)
            grid.setContentsMargins(0,0,0,0)

            start = face_idx * 9
            for local_idx in range(9):
                pos_idx = start + local_idx
                btn = QtWidgets.QPushButton()
                btn.setEnabled(False)
                btn.setFixedSize(44,44)
                grid.addWidget(btn, local_idx//3, local_idx%3)
                self.sticker_buttons[pos_idx] = btn

            face_layout.addWidget(grid_widget)
            cube_net.addWidget(face_box, fr, fc)

        left.addWidget(cube_wrap, 1)
        root.addWidget(left_card, 3)

        right_card = QtWidgets.QFrame()
        right_card.setObjectName("card")
        right = QtWidgets.QVBoxLayout(right_card)
        right.setContentsMargins(20,20,20,20)
        right.setSpacing(14)

        ct = QtWidgets.QLabel("DeepCubeA Solver")
        ct.setObjectName("sectionTitle")
        right.addWidget(ct)

        self.model_status = QtWidgets.QLabel("Model: đang tải...")
        self.model_status.setObjectName("statusLabel")
        right.addWidget(self.model_status)

        g1 = QtWidgets.QGroupBox("Scramble")
        gl1 = QtWidgets.QGridLayout(g1)
        gl1.addWidget(QtWidgets.QLabel("Depth:"),0,0)
        self.scramble_depth = QtWidgets.QSpinBox()
        self.scramble_depth.setRange(1,50)
        self.scramble_depth.setValue(5)
        gl1.addWidget(self.scramble_depth,0,1)

        self.btn_scramble = QtWidgets.QPushButton("Scramble")
        self.btn_scramble.clicked.connect(self.scramble_cube)
        gl1.addWidget(self.btn_scramble,1,0,1,2)

        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_cube)
        gl1.addWidget(self.btn_reset,2,0,1,2)
        right.addWidget(g1)

        g2 = QtWidgets.QGroupBox("A* settings")
        fl = QtWidgets.QFormLayout(g2)

        self.weight = QtWidgets.QDoubleSpinBox()
        self.weight.setRange(0.0,1.0)
        self.weight.setDecimals(2)
        self.weight.setSingleStep(0.05)
        self.weight.setValue(0.60)
        fl.addRow("Weight λ:", self.weight)

        self.batch_size = QtWidgets.QSpinBox()
        self.batch_size.setRange(1,10000)
        self.batch_size.setValue(100)
        fl.addRow("Batch size:", self.batch_size)
        right.addWidget(g2)

        self.btn_solve = QtWidgets.QPushButton("Solve with DeepCubeA")
        self.btn_solve.setObjectName("primaryButton")
        self.btn_solve.clicked.connect(self.solve_cube)
        self.btn_solve.setEnabled(False)
        right.addWidget(self.btn_solve)

        g3 = QtWidgets.QGroupBox("Manual moves")
        mg = QtWidgets.QGridLayout(g3)
        for i,(label, move_idx) in enumerate(self.MOVE_BUTTONS):
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(lambda checked=False, idx=move_idx: self.apply_manual_move(idx))
            mg.addWidget(btn, i//4, i%4)
        right.addWidget(g3)

        g4 = QtWidgets.QGroupBox("Kết quả")
        rl = QtWidgets.QVBoxLayout(g4)

        self.solution_text = QtWidgets.QPlainTextEdit()
        self.solution_text.setReadOnly(True)
        self.solution_text.setMaximumHeight(105)
        self.solution_text.setPlaceholderText("Chuỗi lời giải sẽ xuất hiện ở đây.")
        rl.addWidget(self.solution_text)

        self.stats_label = QtWidgets.QLabel("Moves: -   |   Time: -   |   Nodes: -")
        rl.addWidget(self.stats_label)

        nav = QtWidgets.QHBoxLayout()
        self.btn_next = QtWidgets.QPushButton("Next move")
        self.btn_next.clicked.connect(self.next_solution_move)
        self.btn_next.setEnabled(False)
        nav.addWidget(self.btn_next)

        self.btn_play = QtWidgets.QPushButton("Play solution")
        self.btn_play.clicked.connect(self.play_solution)
        self.btn_play.setEnabled(False)
        nav.addWidget(self.btn_play)

        rl.addLayout(nav)
        right.addWidget(g4)

        self.log = QtWidgets.QLabel("Sẵn sàng. Hãy Scramble rồi bấm Solve.")
        self.log.setWordWrap(True)
        self.log.setObjectName("muted")
        right.addWidget(self.log)
        right.addStretch(1)

        root.addWidget(right_card, 2)

    def _apply_theme(self):
        self.setStyleSheet("""
        QMainWindow { background: #edf5fb; }
        QFrame#card {
            background: white;
            border: 1px solid #cfe0ee;
            border-radius: 12px;
        }
        QLabel#sectionTitle {
            color: #2f6690;
            font-size: 22px;
            font-weight: 700;
        }
        QLabel#muted { color: #66788a; }
        QLabel#faceLabel {
            color: #2f6690;
            font-weight: 700;
            font-size: 14px;
        }
        QLabel#statusLabel {
            color: #2f6690;
            background: #eaf4fc;
            border-radius: 6px;
            padding: 8px;
        }
        QGroupBox {
            color: #315a75;
            font-weight: 600;
            border: 1px solid #d6e6f2;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 8px;
        }
        QPushButton {
            background: #f7fbfe;
            color: #315a75;
            border: 1px solid #bfd6e8;
            border-radius: 7px;
            padding: 7px 10px;
        }
        QPushButton:hover { background: #eaf4fc; }
        QPushButton:disabled {
            color: #9aabb7;
            background: #f1f4f6;
        }
        QPushButton#primaryButton {
            color: white;
            background: #4f8fc4;
            border: 1px solid #4f8fc4;
            font-weight: 700;
            padding: 10px;
        }
        QSpinBox, QDoubleSpinBox, QPlainTextEdit {
            border: 1px solid #cbdce8;
            border-radius: 6px;
            padding: 5px;
            background: white;
        }
        """)

    def _load_model(self):
        try:
            info = self.solver.load_model()
            self.model_status.setText(f"Model: loaded  |  device: {info['device']}")
            self.btn_solve.setEnabled(True)
            self.log.setText("Pretrained DeepCubeA model đã load thành công.")
        except Exception as exc:
            self.model_status.setText("Model: lỗi")
            self.log.setText("Không load được model pretrained:\n" + str(exc))

    def render_state(self):
        for pos_idx, sticker_id in enumerate(self.state.colors):
            color_idx = int(sticker_id) // 9
            self.sticker_buttons[pos_idx].setStyleSheet(
                f"background:{self.COLOR_MAP[color_idx]};"
                "border:1px solid #c8d2da;"
            )

    def reset_cube(self):
        self.animation_timer.stop()
        self.state = self.solver.solved_state()
        self.scramble_moves = []
        self.solution_moves = []
        self.solution_step = 0
        self.solution_text.clear()
        self.stats_label.setText("Moves: -   |   Time: -   |   Nodes: -")
        self.btn_next.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.render_state()
        self.log.setText("Đã đưa Rubik về trạng thái goal.")

    def scramble_cube(self):
        self.animation_timer.stop()
        depth = self.scramble_depth.value()
        self.state, self.scramble_moves = self.solver.scramble(depth)
        self.solution_moves = []
        self.solution_step = 0
        self.render_state()

        names = [self.solver.DISPLAY_MOVES[i] for i in self.scramble_moves]
        self.solution_text.setPlainText("Scramble: " + " ".join(names))
        self.stats_label.setText("Moves: -   |   Time: -   |   Nodes: -")
        self.btn_next.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.log.setText(f"Đã scramble {depth} bước.")

    def apply_manual_move(self, move_idx: int):
        if self.solve_thread is not None:
            return
        self.animation_timer.stop()
        self.state = self.solver.apply_move(self.state, move_idx)
        self.solution_moves = []
        self.solution_step = 0
        self.render_state()
        self.solution_text.setPlainText("Manual move: " + self.solver.DISPLAY_MOVES[move_idx])
        self.btn_next.setEnabled(False)
        self.btn_play.setEnabled(False)

    def solve_cube(self):
        if self.solve_thread is not None:
            return

        self.animation_timer.stop()
        self.solution_moves = []
        self.solution_step = 0
        self.log.setText("DeepCubeA đang tìm lời giải...")
        self.solution_text.setPlainText("Đang search...")
        self.btn_solve.setEnabled(False)
        self.btn_scramble.setEnabled(False)
        self.btn_reset.setEnabled(False)

        self.solve_thread = QtCore.QThread(self)
        self.solve_worker = SolveWorker(
            self.solver,
            self.state,
            self.weight.value(),
            self.batch_size.value()
        )
        self.solve_worker.moveToThread(self.solve_thread)
        self.solve_thread.started.connect(self.solve_worker.run)
        self.solve_worker.finished.connect(self._solve_finished)
        self.solve_worker.failed.connect(self._solve_failed)
        self.solve_worker.finished.connect(self.solve_thread.quit)
        self.solve_worker.failed.connect(self.solve_thread.quit)
        self.solve_thread.finished.connect(self._cleanup_worker)
        self.solve_thread.start()

    def _solve_finished(self, result: dict):
        self.solution_moves = list(result["moves"])
        self.solution_step = 0

        if result["move_names"]:
            self.solution_text.setPlainText("Solution: " + " ".join(result["move_names"]))
        else:
            self.solution_text.setPlainText("Rubik đã ở trạng thái goal.")

        self.stats_label.setText(
            f"Moves: {len(self.solution_moves)}   |   "
            f"Time: {result['solve_time']:.2f}s   |   "
            f"Nodes: {result['num_nodes_generated']:,}"
        )

        has_moves = bool(self.solution_moves)
        self.btn_next.setEnabled(has_moves)
        self.btn_play.setEnabled(has_moves)
        self.btn_solve.setEnabled(True)
        self.btn_scramble.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.log.setText("Đã tìm được lời giải bằng DeepCubeA.")

    def _solve_failed(self, message: str):
        self.solution_text.setPlainText("Solve failed.")
        self.log.setText("Lỗi khi solve:\n" + message)
        self.btn_solve.setEnabled(True)
        self.btn_scramble.setEnabled(True)
        self.btn_reset.setEnabled(True)

    def _cleanup_worker(self):
        if self.solve_worker is not None:
            self.solve_worker.deleteLater()
        if self.solve_thread is not None:
            self.solve_thread.deleteLater()
        self.solve_worker = None
        self.solve_thread = None

    def next_solution_move(self):
        if self.solution_step >= len(self.solution_moves):
            self.btn_next.setEnabled(False)
            self.btn_play.setEnabled(False)
            return

        move_idx = self.solution_moves[self.solution_step]
        self.state = self.solver.apply_move(self.state, move_idx)
        self.solution_step += 1
        self.render_state()

        self.log.setText(
            f"Step {self.solution_step}/{len(self.solution_moves)}: "
            f"{self.solver.DISPLAY_MOVES[move_idx]}"
        )

        if self.solution_step >= len(self.solution_moves):
            self.btn_next.setEnabled(False)
            self.btn_play.setEnabled(False)
            self.animation_timer.stop()
            solved = bool(self.solver.env.is_solved([self.state])[0])
            self.log.setText("Hoàn tất lời giải. Rubik solved." if solved
                             else "Đã chạy hết solution nhưng state chưa solved.")

    def play_solution(self):
        if not self.solution_moves:
            return

        if self.animation_timer.isActive():
            self.animation_timer.stop()
            self.btn_play.setText("Play solution")
        else:
            self.animation_timer.start(350)
            self.btn_play.setText("Pause")

    def _animate_next_solution_move(self):
        if self.solution_step >= len(self.solution_moves):
            self.animation_timer.stop()
            self.btn_play.setText("Play solution")
            return
        self.next_solution_move()
