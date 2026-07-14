# Sensor Characterization Offline Data Pipeline

## 1. 目前架構決策

本專案目前定位為：

> Offline sensor characterization tool

主要工作內容為：

```text
rosbag2
↓
extract raw depth frames
↓
NPZ dataset
↓
ROI selection
↓
metric analysis
↓
results
```

目前不需要建立 ROS package，也不需要使用 `ros2 pkg create` 或 `colcon build`。

雖然 extraction 會使用：

```python
rosbag2_py
rclpy
sensor_msgs
```

但這只代表執行環境需要 ROS2，不代表程式本身必須是 ROS package。

執行前仍需：

```bash
source /opt/ros/humble/setup.bash
```

---

## 2. 資料分層

專案資料分成三層：

```text
bags/
data/
results/
```

### `bags/`

Raw experiment data。

```text
bags/
└── scene01_white_d050_r01/
```

用途：

- 保存 rosbag2 原始資料
- 保存 experiment metadata
- 作為 raw source of truth

原則：

> 不修改原始 rosbag。

---

### `data/`

Derived dataset。

```text
data/
└── scene01_white_d050_r01/
    └── depth.npz
```

用途：

- 保存 rosbag extraction 結果
- 避免每次 analysis 都重新 deserialize rosbag
- 提供純 NumPy analysis input

原則：

> 可以刪除並由 rosbag 重新產生。

---

### `results/`

Analysis output。

```text
results/
└── scene01_white_d050_r01/
```

用途：

- inspection image
- metric map
- CSV
- YAML summary
- plots

原則：

> 可以刪除並重新分析產生。

---

## 3. ROI 決策

目前 ROI **不放在 extraction 階段**。

資料流為：

```text
rosbag
↓
full depth frame extraction
↓
depth.npz
↓
ROI selection
↓
metrics
```

原因：

- ROI 策略目前仍可能調整
- 不同距離下 white board pixel footprint 不同
- 若 extraction 時直接 crop ROI，修改 ROI 後必須重新讀 rosbag
- 保留 full depth frame 可以重複測試不同 ROI

因此：

> `depth.npz` 保存完整 depth frame。

---

## 4. 建議專案結構

目前先使用一般 Python `src` layout：

```text
orbbec_ws/
├── bags/
├── data/
├── results/
├── config/
│
├── scripts/
│   └── record_experiment.sh
│
├── tools/
│   ├── extract_dataset.py
│   └── inspect_dataset.py
│
└── src/
    ├── __init__.py
    │
    └── io/
        ├── __init__.py
        ├── bag_reader.py
        └── dataset.py
```

目前只先完成四個主要檔案：

```text
src/io/bag_reader.py
src/io/dataset.py
tools/extract_dataset.py
tools/inspect_dataset.py
```

ROI 與 metrics 等 extraction pipeline 驗證完成後再實作。

---

# 5. 各 Python 檔案用途

## 5.1 `src/io/bag_reader.py`

主要責任：

> rosbag2 → NumPy depth frame

它只處理 ROS2 rosbag I/O。

### Input

```text
rosbag path
depth topic
```

例如：

```python
reader = DepthBagReader(
    bag_path=Path("bags/scene01_white_d050_r01"),
    depth_topic="/camera/depth/image_raw",
)
```

### Output

逐 frame 回傳：

```python
timestamp_ns, depth_frame
```

其中：

```text
timestamp_ns
type: int

depth_frame
type: np.ndarray
shape: (H, W)
dtype: uint16
```

### 負責內容

1. 使用 `rosbag2_py.SequentialReader` 開啟 rosbag
2. 查詢 topic type
3. 找到指定 depth topic
4. deserialize ROS message
5. 將 `sensor_msgs/msg/Image` 轉成 NumPy array
6. 驗證 encoding
7. 驗證 width / height / step
8. 回傳 timestamp 與 depth frame

### 不負責

不要在這裡：

```python
depth[depth == 0] = np.nan
```

不要 crop ROI：

```python
depth[y1:y2, x1:x2]
```

不要計算：

```python
np.std(depth)
```

不要：

```python
np.savez(...)
```

`bag_reader.py` 的唯一主要責任是：

> ROS message decoding。

---

## 5.2 `src/io/dataset.py`

主要責任：

> 定義 internal depth dataset format，並提供 save / load。

建議資料模型：

```python
@dataclass
class DepthDataset:
    depth: np.ndarray
    timestamps_ns: np.ndarray
```

固定格式：

```text
depth
shape: (N, H, W)
dtype: uint16

timestamps_ns
shape: (N,)
dtype: int64
```

例如：

```text
depth.shape = (842, 480, 640)
```

代表：

```text
842 frames
480 image height
640 image width
```

### 負責內容

1. 固定 dataset shape convention
2. 驗證 frame count 與 timestamp count
3. 提供基本 properties
4. 保存 NPZ
5. 載入 NPZ

建議 properties：

```python
dataset.num_frames
dataset.height
dataset.width
```

### Validation

至少檢查：

```python
depth.ndim == 3
timestamps_ns.ndim == 1
depth.shape[0] == timestamps_ns.shape[0]
```

### 不負責

不要：

```python
dataset.compute_temporal_noise()
```

不要：

```python
dataset.crop_roi()
```

不要：

```python
dataset.plot_depth()
```

`DepthDataset` 只是一個 data container。

---

## 5.3 `tools/extract_dataset.py`

主要責任：

> 執行完整 rosbag → NPZ extraction pipeline。

它是 executable orchestration script。

資料流：

```text
CLI
↓
DepthBagReader
↓
collect frames
↓
DepthDataset
↓
save depth.npz
```

它會 import：

```python
from src.io.bag_reader import DepthBagReader
from src.io.dataset import DepthDataset
```

### 負責內容

1. 解析 CLI arguments
2. 建立 `DepthBagReader`
3. iterate depth frames
4. 收集 timestamps
5. 使用 `np.stack()` 建立 `(N, H, W)`
6. 建立 `DepthDataset`
7. 建立 output directory
8. 保存 `depth.npz`
9. print extraction summary

### 第一版 CLI

先保持簡單：

```bash
python3 tools/extract_dataset.py     bags/scene01_white_d050_r01     data/scene01_white_d050_r01/depth.npz
```

目前不要急著加入：

- 自動 parse `experiment.yaml`
- 自動 output path resolution
- 複雜 config loader
- ROI
- metric

先驗證 extraction pipeline。

---

## 5.4 `tools/inspect_dataset.py`

主要責任：

> 驗證 extracted dataset 是否可信。

這不是正式 characterization metric。

它是一支：

> Dataset QA / sanity check tool。

### 應檢查內容

#### Dataset structure

```text
path
shape
dtype
frame count
width
height
```

#### Timestamp statistics

```text
first timestamp
last timestamp
duration
mean interval
median interval
minimum interval
maximum interval
estimated FPS
```

因為目前 rosbag FPS 有 jitter，所以不能只輸出 average FPS。

至少需要觀察：

```text
mean dt
median dt
min dt
max dt
```

#### Raw depth statistics

先輸出原始 value：

```text
min
max
median
percentiles
```

目前先不要直接假設：

```text
0 = invalid
65535 = invalid
```

Invalid value 定義應在確認 Orbbec depth representation 後再固定。

### Visual inspection

輸出三張 depth image：

```text
first frame
middle frame
last frame
```

例如：

```text
results/
└── scene01_white_d050_r01/
    └── inspection/
        ├── frame_first.png
        ├── frame_middle.png
        └── frame_last.png
```

人工確認：

- white board 是否在畫面中
- image orientation 是否正確
- 是否出現 stride error
- 是否出現 endian error
- 是否有 horizontal stripe
- 是否有 image tearing / offset
- first / middle / last scene 是否一致

重要觀念：

> 程式成功執行，不代表 extracted data 一定正確。

---

# 6. 四個檔案實作順序

必須依照以下順序。

## Step 1 — `src/io/bag_reader.py`

第一個完成。

原因：

```text
dataset extraction
↓
依賴 bag reader
```

目標：

> 可以從一個 rosbag iterate depth frames。

---

## Step 2 — 測試 `bag_reader.py`

### Test A：讀取前 5 frames

```bash
cd ~/dev/orbbec_ws

source /opt/ros/humble/setup.bash

python3 - <<'PY'
from pathlib import Path

from src.io.bag_reader import DepthBagReader

reader = DepthBagReader(
    bag_path=Path("bags/scene01_white_d050_r01"),
    depth_topic="/camera/depth/image_raw",
)

for i, (timestamp_ns, depth) in enumerate(reader.read_frames()):
    print(
        f"frame={i}, "
        f"timestamp={timestamp_ns}, "
        f"shape={depth.shape}, "
        f"dtype={depth.dtype}, "
        f"min={depth.min()}, "
        f"max={depth.max()}"
    )

    if i >= 4:
        break
PY
```

確認：

- bag 可以開啟
- topic 找得到
- message 可以 deserialize
- shape 正確
- dtype 正確
- timestamp 有變化

### Test B：完整 iterate rosbag

```bash
python3 - <<'PY'
from pathlib import Path

from src.io.bag_reader import DepthBagReader

reader = DepthBagReader(
    Path("bags/scene01_white_d050_r01"),
    "/camera/depth/image_raw",
)

count = 0
last_timestamp = None

for timestamp_ns, depth in reader.read_frames():
    if last_timestamp is not None:
        assert timestamp_ns > last_timestamp

    last_timestamp = timestamp_ns
    count += 1

print(f"Total frames: {count}")
PY
```

確認：

```text
timestamp strictly increasing
frame count > 0
```

完成後再進入 `dataset.py`。

---

## Step 3 — `src/io/dataset.py`

第二個完成。

它不依賴 rosbag。

目標：

> 確認 dataset format 與 NPZ round-trip 正確。

---

## Step 4 — 測試 `dataset.py`

使用 synthetic NumPy data。

```bash
python3 - <<'PY'
from pathlib import Path

import numpy as np

from src.io.dataset import DepthDataset

depth = np.arange(
    5 * 4 * 3,
    dtype=np.uint16,
).reshape(5, 4, 3)

timestamps_ns = np.array(
    [100, 200, 300, 400, 500],
    dtype=np.int64,
)

dataset = DepthDataset(
    depth=depth,
    timestamps_ns=timestamps_ns,
)

print(dataset.num_frames)
print(dataset.height)
print(dataset.width)

output = Path("/tmp/test_depth_dataset.npz")

dataset.save(output)

loaded = DepthDataset.load(output)

assert np.array_equal(
    dataset.depth,
    loaded.depth,
)

assert np.array_equal(
    dataset.timestamps_ns,
    loaded.timestamps_ns,
)

assert loaded.depth.dtype == np.uint16
assert loaded.timestamps_ns.dtype == np.int64

print("DepthDataset test passed")
PY
```

預期：

```text
5
4
3
DepthDataset test passed
```

### Invalid input test

```bash
python3 - <<'PY'
import numpy as np

from src.io.dataset import DepthDataset

try:
    DepthDataset(
        depth=np.zeros(
            (10, 480, 640),
            dtype=np.uint16,
        ),
        timestamps_ns=np.zeros(
            9,
            dtype=np.int64,
        ),
    )
except ValueError as exc:
    print(f"Expected error: {exc}")
else:
    raise RuntimeError(
        "Expected ValueError was not raised"
    )
PY
```

確認：

> dataset shape 不一致時必須 fail。

不要 silent ignore。

---

## Step 5 — `tools/extract_dataset.py`

第三個完成。

目標：

```text
DepthBagReader
↓
DepthDataset
↓
depth.npz
```

---

## Step 6 — 測試 `extract_dataset.py`

先刪除舊測試 output：

```bash
rm -rf data/scene01_white_d050_r01
```

執行 extraction：

```bash
source /opt/ros/humble/setup.bash

python3 tools/extract_dataset.py     bags/scene01_white_d050_r01     data/scene01_white_d050_r01/depth.npz
```

建議輸出：

```text
Bag: bags/scene01_white_d050_r01
Depth topic: /camera/depth/image_raw
Frames extracted: 842
Depth shape: (842, 480, 640)
Depth dtype: uint16
Output: data/scene01_white_d050_r01/depth.npz
```

確認檔案：

```bash
ls -lh data/scene01_white_d050_r01/
```

應存在：

```text
depth.npz
```

---

## Step 7 — Bag / NPZ consistency test

這是 extraction pipeline 最重要的測試。

比較：

```text
rosbag original frame
==
NPZ extracted frame
```

測試：

```bash
python3 - <<'PY'
from pathlib import Path

import numpy as np

from src.io.bag_reader import DepthBagReader
from src.io.dataset import DepthDataset

bag_path = Path(
    "bags/scene01_white_d050_r01"
)

dataset_path = Path(
    "data/scene01_white_d050_r01/depth.npz"
)

reader = DepthBagReader(
    bag_path,
    "/camera/depth/image_raw",
)

dataset = DepthDataset.load(
    dataset_path
)

bag_frames = list(
    reader.read_frames()
)

assert len(bag_frames) == dataset.num_frames

indices = [
    0,
    len(bag_frames) // 2,
    len(bag_frames) - 1,
]

for index in indices:
    timestamp_ns, depth = bag_frames[index]

    assert (
        timestamp_ns
        == dataset.timestamps_ns[index]
    )

    assert np.array_equal(
        depth,
        dataset.depth[index],
    )

    print(
        f"frame {index}: passed"
    )

print("Bag/NPZ consistency test passed")
PY
```

預期：

```text
frame 0: passed
frame 421: passed
frame 841: passed
Bag/NPZ consistency test passed
```

要求：

> Depth frame 必須 bitwise equal。

使用：

```python
np.array_equal(
    bag_depth,
    npz_depth,
)
```

只有這個測試通過，才能合理地將 `depth.npz` 視為可信 analysis input。

---

## Step 8 — `tools/inspect_dataset.py`

最後完成。

目標：

> 對 extracted NPZ 進行數值與視覺 QA。

執行：

```bash
python3 tools/inspect_dataset.py     data/scene01_white_d050_r01/depth.npz
```

建議輸出格式：

```text
Dataset:
  path: data/scene01_white_d050_r01/depth.npz

Depth:
  shape: (842, 480, 640)
  dtype: uint16

Frames:
  count: 842

Timestamp:
  duration: 30.042 sec
  mean interval: 35.68 ms
  median interval: 33.42 ms
  min interval: 0.01 ms
  max interval: 161.23 ms
  estimated FPS: 28.03 Hz

Raw depth:
  min: 0
  max: 65535
  median: 503
```

另外輸出：

```text
frame_first.png
frame_middle.png
frame_last.png
```

完成人工 visual inspection。

---

# 7. 完整執行順序

```text
1. 寫 src/io/bag_reader.py

2. 讀前 5 frames
   - shape
   - dtype
   - timestamp
   - min/max

3. 完整 iterate rosbag
   - frame count
   - timestamp strictly increasing

4. 寫 src/io/dataset.py

5. synthetic data save/load test

6. dataset invalid input test

7. 寫 tools/extract_dataset.py

8. extract scene01_white_d050_r01

9. 比較 bag 與 NPZ
   - first frame
   - middle frame
   - last frame

10. 寫 tools/inspect_dataset.py

11. 檢查 timestamp statistics

12. 輸出 first / middle / last depth image

13. 人工確認 depth image

14. extraction pipeline 通過後才開始 ROI

15. ROI 完成後才開始 temporal noise / invalid ratio / measured depth metrics
```

---

# 8. Extraction Pipeline Passing Criteria

以下條件必須全部通過：

| Test | Requirement |
|---|---|
| Bag 可以正常開啟 | Pass |
| Depth topic 存在 | Pass |
| ROS Image 可以 deserialize | Pass |
| Depth encoding 符合預期 | Pass |
| Frame shape 固定 | Pass |
| Depth dtype 固定 | Pass |
| Timestamp strictly increasing | Pass |
| Dataset save/load 完整一致 | Pass |
| Dataset invalid shape 會 fail | Pass |
| Bag frame count = NPZ frame count | Pass |
| First bag frame = NPZ frame | Pass |
| Middle bag frame = NPZ frame | Pass |
| Last bag frame = NPZ frame | Pass |
| First/middle/last image 視覺正常 | Pass |

最重要的 passing criterion：

```text
bag frame
==
extracted NPZ frame
```

必須通過：

```python
np.array_equal(
    bag_depth,
    npz_depth,
)
```

---

# 9. 目前明確下一步

目前先不要開始：

```text
ROI
temporal noise
invalid ratio
measured depth
cross-distance comparison
```

先完成並驗證：

```text
src/io/bag_reader.py
src/io/dataset.py
tools/extract_dataset.py
tools/inspect_dataset.py
```

目前階段的目標是證明：

> rosbag → full raw depth NPZ extraction pipeline 是可信、可重現，而且 extracted frame 與 rosbag original frame 完全一致。

完成這個基礎後，再開始 ROI 與 characterization metrics。
