# Maximum-uint16 異常數值診斷計畫

## 1. 背景

目前在抽取後的 depth dataset 中觀察到：

```text
65535
```

此數值等於 `uint16` 的最大值。全畫面初步統計顯示，它只出現在部分 frames，且 affected frame 內可能同時出現數十至數百個像素。

目前只能將它描述為：

> observed maximum-uint16 special value or burst-like artifact

在取得官方感測器文件或完成資料來源追查前，不應直接宣稱：

```text
Orbbec invalid depth sentinel = 65535
```

## 2. 診斷目標

本診斷需要回答：

1. `65535` 出現在哪些 frames？
2. 它在時間上是單次事件、連續 burst，還是週期性事件？
3. 它在影像中出現在哪些位置？
4. 它是固定像素、局部區域、物體邊界，還是隨機分布？
5. 它是否進入後續使用的白板 ROI？
6. 它已存在於 rosbag 原始 depth message，還是 extraction 過程產生？
7. 正式 baseline analysis 應如何記錄並排除此數值？

## 3. 診斷原則

- 使用原始 `uint16` depth frames 進行診斷。
- 不修改 `depth.npz`。
- 不在計數前將 `65535` 轉成 `NaN`。
- `65535` 不可參與正常 depth 顯示範圍的 percentile 計算。
- 先完成全畫面定位，再檢查白板 ROI 是否受到影響。
- 將「觀察到此數值」與「確認此數值的官方語意」分開記錄。
- 診斷工具保持 read-only，不改變 dataset 或 ROI configuration。

## 4. 建議工具

新增獨立工具：

```text
tools/inspect_max_uint16.py
```

建議 CLI：

```bash
python3 tools/inspect_max_uint16.py \
    data/scene01_white_d050_r01
```

可選參數可在基本版本驗證後再決定是否加入：

```text
--output-root
--max-overlay-frames
--roi-root
--include-all-affected-frames
```

第一版不需要互動式 GUI，也不應依賴 ROI 已經存在。

## 5. 第一階段：NPZ 數值確認

載入：

```text
data/<experiment>/depth.npz
```

確認：

```text
depth dtype = uint16
depth shape = (N, H, W)
```

建立 mask：

```python
max_uint16 = np.iinfo(np.uint16).max
mask = depth == max_uint16
```

計算全資料集摘要：

```text
total sample count
65535 total count
65535 overall ratio
affected frame count
first affected frame
last affected frame
maximum pixels in one frame
frame index with maximum count
```

基本一致性檢查：

```python
total_count = np.count_nonzero(mask)
per_frame_count = np.count_nonzero(mask, axis=(1, 2))

assert total_count == per_frame_count.sum()
assert np.count_nonzero(per_frame_count) == affected_frame_count
```

## 6. 第二階段：時間分布

對每個 frame 計算：

```python
max_uint16_count = np.count_nonzero(
    mask,
    axis=(1, 2),
)
```

輸出：

```text
frame_counts.csv
```

建議欄位：

| frame_index | timestamp_ns | elapsed_ms | max_uint16_count | max_uint16_ratio | affected |
|---:|---:|---:|---:|---:|---|

需要辨識的時間型態：

```text
isolated event
consecutive burst
periodic event
long-duration affected interval
```

### 6.1 Burst 定義

第一版可將連續 affected frames 視為同一個 burst：

```text
frame 120 affected
frame 121 affected
frame 122 affected
→ one burst: 120–122
```

輸出：

```text
bursts.csv
```

建議欄位：

| burst_id | start_frame | end_frame | frame_count | start_timestamp_ns | end_timestamp_ns | total_pixels | peak_pixels | peak_frame |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|

如果之後發現單一正常 frame 會切斷同一事件，再評估是否允許短 gap；第一版不先加入此規則。

## 7. 第三階段：空間分布

計算每個像素在所有 frames 中出現 `65535` 的次數：

```python
occurrence_count_map = np.sum(
    mask,
    axis=0,
)
```

計算 occurrence ratio：

```python
occurrence_ratio_map = np.mean(
    mask,
    axis=0,
)
```

輸出原始陣列：

```text
max_uint16_count_map.npy
max_uint16_ratio_map.npy
```

輸出視覺化：

```text
max_uint16_count_heatmap.png
max_uint16_ratio_heatmap.png
```

heatmap 需要明確標示：

```text
image width and height
minimum and maximum occurrence
color scale meaning
experiment name
number of frames
```

需要觀察的空間型態：

- 固定像素反覆出現
- 固定區域反覆出現
- 影像邊緣集中
- 深度不連續邊界集中
- 大片矩形或條帶狀區域
- 每次事件位置不同
- 白板表面或白板邊界集中

## 8. 第四階段：代表 frame 視覺化

至少選擇：

```text
first affected frame
frame with maximum affected pixels
middle frame of the largest burst
last affected frame
```

若這些條件指向同一 frame，不需要重複輸出。

每個代表 frame 輸出：

```text
frames/frame_<index>_depth.png
frames/frame_<index>_mask.png
frames/frame_<index>_overlay.png
```

### 8.1 Depth 顯示影像

顯示範圍只使用正常候選值：

```python
valid_for_display = (
    (depth > 0)
    & (depth < np.iinfo(np.uint16).max)
)
```

對有效值使用 percentile normalization，例如：

```text
1st percentile → 0
99th percentile → 255
```

此轉換只用於顯示，不可修改原始 depth。

### 8.2 Mask

Mask 圖像應採二值顯示：

```text
black = normal
white = 65535
```

### 8.3 Overlay

Overlay 建議：

```text
normal depth = grayscale or colormap
65535 pixels = red
affected bounding boxes = yellow
```

圖像上記錄：

```text
experiment name
frame index
timestamp
65535 pixel count
65535 frame ratio
```

若加入 connected-component 分析，第一版只將它用於視覺化 bounding box，不將 component 數量視為正式 sensor metric。

## 9. 第五階段：ROI 影響確認

ROI YAML 建立後，使用相同的 raw mask 計算白板 ROI 內的結果：

```python
raw_roi = roi.crop(depth)
roi_mask = raw_roi == np.iinfo(np.uint16).max
```

記錄：

```text
ROI 65535 total count
ROI 65535 ratio
ROI affected frames
ROI maximum pixels per frame
```

此步驟回答的是：

> 全畫面觀察到的 artifact 是否實際影響白板 baseline measurement？

它不取代全畫面的空間與時間診斷。

## 10. 第六階段：回溯 rosbag

NPZ 診斷只能證明 extraction 後存在 `65535`。若要判斷來源，需要選擇 affected frames，依 timestamp 回查 rosbag。

優先回查：

```text
first affected frame
peak affected frame
one frame before a burst
one frame after a burst
```

比較層級：

```text
rosbag depth message payload
↓
decoded uint16 frame
↓
saved depth.npz frame
```

檢查：

```text
encoding
height and width
is_bigendian
row step
payload length
padding handling
reshape behavior
```

### 10.1 判斷準則

若 rosbag decoded frame 和 NPZ 在相同座標都為 `65535`：

> 此數值不是由 NPZ 儲存步驟產生；來源位於 sensor、firmware、driver 或 ROS publisher 的上游。

若 rosbag decoded frame 不是 `65535`，但 NPZ 是：

> 優先檢查 extraction pipeline 的 decoding、row step、padding、endianness 或 array reshape。

若 rosbag payload 本身已包含對應的 `0xFF 0xFF`：

> 此數值在 extraction 前已存在，但仍不能只靠此結果確認其官方語意。

注意：`65535` 的 byte representation 是 `0xFF 0xFF`，大小端交換後數值不變。因此 endianness 本身通常無法單獨解釋 `65535`，但整體解碼流程仍應檢查。

## 11. 建議輸出結構

```text
results/
└── <experiment>/
    └── max_uint16_diagnostic/
        ├── summary.yaml
        ├── frame_counts.csv
        ├── bursts.csv
        ├── max_uint16_count_map.npy
        ├── max_uint16_ratio_map.npy
        ├── max_uint16_count_heatmap.png
        ├── max_uint16_ratio_heatmap.png
        └── frames/
            ├── frame_<index>_depth.png
            ├── frame_<index>_mask.png
            └── frame_<index>_overlay.png
```

建議 `summary.yaml`：

```yaml
dataset:
  experiment: scene01_white_d050_r01
  num_frames: 842
  width: 640
  height: 480

max_uint16:
  value: 65535
  total_count: 7376
  overall_ratio: 0.0000203
  affected_frames: 48
  first_affected_frame: 120
  last_affected_frame: 731
  max_pixels_per_frame: 434
  peak_frame: 421
  burst_count: 7

representative_frames:
  - 120
  - 421
  - 731
```

範例數值僅用來說明格式，實際輸出必須由 dataset 計算。

## 12. 測試計畫

為診斷核心邏輯建立小型 synthetic depth array，測試：

```text
no 65535 values
one isolated 65535 pixel
multiple pixels in one frame
consecutive affected frames
multiple separated bursts
fixed pixel across frames
affected region touching image boundary
all pixels equal 65535 in one frame
```

必須驗證：

- total count 正確
- overall ratio 正確
- per-frame count 正確
- affected frame count 正確
- peak frame 正確
- occurrence count/ratio maps 正確
- burst start/end frame 正確
- 無異常值時仍能輸出有效的空結果
- display normalization 不包含 `0` 與 `65535`
- 視覺化不修改輸入 array

rosbag 回溯屬於整合診斷，不應只用 synthetic unit test 取代。

## 13. 與正式 baseline pipeline 的關係

診斷流程：

```text
raw uint16 depth
↓
locate and visualize 65535
↓
trace affected frames to rosbag
↓
document likely source layer
```

正式 baseline analysis：

```text
raw ROI uint16
├── count 65535 as depth-quality metric
└── convert 65535 to NaN
        ↓
    temporal noise
    measured depth
```

即使尚未確認 `65535` 的官方語意，也不得將它納入正常 measured-depth 或 temporal-noise 統計。

## 14. 實作順序

```text
1. 定義純 NumPy 統計函式與結果資料模型
2. 使用 synthetic arrays 完成單元測試
3. 實作 frame_counts.csv
4. 實作 occurrence maps
5. 實作代表 frame 選取
6. 實作 depth、mask、overlay 圖像輸出
7. 在 scene01_white_d050_r01 上執行
8. 人工檢查時間與空間型態
9. ROI 建立後計算 ROI 內 occurrence
10. 依 timestamp 回查代表 frames 的 rosbag message
11. 記錄來源層級判斷與仍未確認的語意
```

不要在第一版加入複雜互動式 GUI、即時動畫或自動 sensor defect 分類。

## 15. 完成條件

此診斷 milestone 完成時應具備：

- 可重現的 full-frame `65535` 數值摘要
- 每個 frame 的時間序列資料
- burst 清單
- 全資料集空間 occurrence maps
- 代表 affected frames 的 depth、mask 與 overlay
- 白板 ROI 內的 occurrence 摘要
- 至少一個 affected frame 的 rosbag-to-NPZ 比對結果
- 明確區分 observed behavior、likely source layer 與 confirmed sensor semantics

最終診斷結論應避免超出證據，例如：

```text
65535 was already present in the decoded rosbag depth frame and appeared as
spatially clustered bursts in 48 frames. Its official sensor-level semantics
remain unconfirmed.
```

