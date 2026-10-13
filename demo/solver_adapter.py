from pathlib import Path
from typing import Dict, List, Optional
import time
import numpy as np

from environments.cube3 import Cube3, Cube3State
from search_methods.astar import AStar, get_path
from utils import nnet_utils, search_utils


class DeepCubeASolver:
    DISPLAY_MOVES = [
        "U'", "U", "D'", "D", "L'", "L",
        "R'", "R", "B'", "B", "F'", "F"
    ]

    def __init__(self, repo_root: Path, model_dir: Optional[Path] = None,
                 weight: float = 0.6, batch_size: int = 100,
                 nnet_batch_size: int = 1000):
        self.repo_root = Path(repo_root)
        self.model_dir = Path(model_dir) if model_dir else (
            self.repo_root / "saved_models" / "cube3" / "current"
        )
        self.weight = float(weight)
        self.batch_size = int(batch_size)
        self.nnet_batch_size = int(nnet_batch_size)

        self.env = Cube3()
        self.device = None
        self.on_gpu = False
        self.heuristic_fn = None

    def load_model(self):
        model_file = self.model_dir / "model_state_dict.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"Không tìm thấy model pretrained:\n{model_file}")

        device, devices, on_gpu = nnet_utils.get_device()
        self.device = device
        self.on_gpu = on_gpu

        self.heuristic_fn = nnet_utils.load_heuristic_fn(
            str(self.model_dir),
            device,
            on_gpu,
            self.env.get_nnet_model(),
            self.env,
            clip_zero=True,
            batch_size=self.nnet_batch_size,
        )
        return {"device": str(device), "gpu": bool(on_gpu), "devices": devices}

    def ensure_loaded(self):
        if self.heuristic_fn is None:
            return self.load_model()
        return None

    def solved_state(self) -> Cube3State:
        return self.env.generate_goal_states(1)[0]

    def apply_move(self, state: Cube3State, move_idx: int) -> Cube3State:
        return self.env.next_state([state], int(move_idx))[0][0]

    def scramble(self, depth: int, rng: Optional[np.random.Generator] = None):
        if depth < 1:
            raise ValueError("Scramble depth phải >= 1")

        rng = rng or np.random.default_rng()
        state = self.solved_state()
        moves: List[int] = []

        inverse = {0:1,1:0,2:3,3:2,4:5,5:4,6:7,7:6,8:9,9:8,10:11,11:10}

        for _ in range(depth):
            choices = list(range(self.env.get_num_moves()))
            if moves:
                choices.remove(inverse[moves[-1]])
            move_idx = int(rng.choice(choices))
            state = self.apply_move(state, move_idx)
            moves.append(move_idx)

        return state, moves

    def solve(self, state: Cube3State, weight: Optional[float] = None,
              batch_size: Optional[int] = None) -> Dict:
        self.ensure_loaded()

        weight = self.weight if weight is None else float(weight)
        batch_size = self.batch_size if batch_size is None else int(batch_size)

        if not (0.0 <= weight <= 1.0):
            raise ValueError("A* weight phải nằm trong [0, 1].")
        if batch_size < 1:
            raise ValueError("A* batch size phải >= 1.")

        if bool(self.env.is_solved([state])[0]):
            return {
                "moves": [], "move_names": [], "path": [state],
                "path_cost": 0.0, "solve_time": 0.0,
                "num_nodes_generated": 0, "num_iterations": 0
            }

        start = time.time()
        astar = AStar([state], self.env, self.heuristic_fn, [weight])

        num_iterations = 0
        while not astar.has_found_goal()[0]:
            astar.step(self.heuristic_fn, batch_size, verbose=False)
            num_iterations += 1

        goal = astar.get_goal_node_smallest_path_cost(0)
        path, moves, path_cost = get_path(goal)
        solve_time = time.time() - start
        nodes = astar.get_num_nodes_generated(0)

        if not search_utils.is_valid_soln(state, moves, self.env):
            raise RuntimeError("DeepCubeA trả về chuỗi move không hợp lệ.")

        return {
            "moves": moves,
            "move_names": [self.DISPLAY_MOVES[i] for i in moves],
            "path": path,
            "path_cost": float(path_cost),
            "solve_time": float(solve_time),
            "num_nodes_generated": int(nodes),
            "num_iterations": int(num_iterations),
        }
