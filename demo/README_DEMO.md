# DeepCubeA Rubik GUI Demo

Demo này dùng trực tiếp:
- `environments/cube3.py`
- pretrained `saved_models/cube3/current/model_state_dict.pt`
- `utils/nnet_utils.py`
- `search_methods/astar.py`

Project `drl-rubiks-cube` chỉ được dùng làm tham khảo bố cục GUI.

## Cấu trúc

Đặt thư mục `demo` vào gốc repo:

```text
DeepCubeA-master/
├── environments/
├── search_methods/
├── saved_models/
├── utils/
└── demo/
    ├── __init__.py
    ├── run_demo.py
    ├── cube_widget.py
    └── solver_adapter.py
```

## Cài PyQt5

Trong virtual environment của DeepCubeA:

```powershell
pip install PyQt5==5.15.4
```

## Chạy

Từ thư mục gốc DeepCubeA:

```powershell
python demo\run_demo.py
```

## Chức năng

- Hiển thị Rubik 3×3 dạng 2D.
- Reset.
- Scramble theo số bước.
- Xoay tay U/D/L/R/B/F và chiều ngược.
- Solve bằng pretrained DeepCubeA value network + Weighted/Batch A*.
- Hiển thị solution, số move, thời gian và số node.
- Chạy từng bước hoặc tự động chạy lời giải.

## Lưu ý

GUI giữ trực tiếp `Cube3State.colors` của DeepCubeA.
Màu hiển thị được suy ra bằng `sticker_id // 9`, nên không phải đổi sang state representation của project khác.
