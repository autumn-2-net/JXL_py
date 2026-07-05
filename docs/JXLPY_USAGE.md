# jxlpy 使用文档

`jxlpy` 是围绕 libjxl 的 Python/CFFI 绑定。通过 native shim（`jxlpy_native`）直接调用 libjxl C API，不依赖命令行工具。

---

## 安装与构建

### 依赖

- Python >= 3.10
- `numpy >= 1.22`
- `cffi >= 1.15`
- `torch`（可选，仅传入 tensor 或 `out="torch"` 时使用）

### 构建 native 库

#### Windows

```bat
.\scripts\build_windows.cmd jxlpy_native
```

产物：`out/build/windows-clang-cl-cli/jxlpy_native.dll`

环境变量可覆盖 CMake/Ninja 路径：

```bat
set CMAKE_EXE=D:\path\to\cmake.exe
set NINJA_EXE=D:\path\to\ninja.exe
.\scripts\build_windows.cmd jxlpy_native
```

#### Linux

```bash
./scripts/build_linux.sh jxlpy_native
```

产物：`out/build/linux-clang-python/libjxlpy_native.so`

#### macOS

```bash
./scripts/build_macos.sh jxlpy_native
```

产物：`out/build/macos-clang-python/libjxlpy_native.dylib`

### Wheel 打包

```bat
.\scripts\build_wheel.cmd
```

```bash
./scripts/build_wheel.sh
```

`setup.py` 自动从 `out/build/` 查找 native 库并复制到 wheel 中。默认静态链接 libjxl。

### 环境变量

| 变量 | 作用 |
|---|---|
| `JXLPY_NATIVE_LIB` | 直接指定 native 库路径，跳过自动查找 |

---

## API 概览

```python
import jxlpy
```

| 函数 | 功能 |
|---|---|
| `encode` | numpy/torch/路径/bytes → JXL bytes |
| `decode` | JXL/PNG/JPEG → numpy/torch/bytes |
| `decode_to_png` | JXL/PNG/JPEG → PNG bytes |
| `decode_to_jpeg` | JXL/PNG/JPEG → JPEG bytes（自动尝试 JPEG reconstruction） |
| `reconstruct_jpeg` | 从 JPEG lossless transcode 的 JXL 提取原始 JPEG（bit-exact） |
| `convert` | 通用格式转换（支持 png/jpg/ppm/pgm/pam/pfm/pgx） |
| `encode_multiframe` | 多帧序列 → JXL animation |
| `decode_layer` | 读取非合成 layer/crop（残差帧） |
| `decode_extra_channel` | 读取单个 extra channel |
| `analyze_multiframe` | 分析帧序列，评估多帧压缩收益 |
| `info` | 获取文件元数据 |

---

## 编码

### 从 numpy 数组编码

```python
import numpy as np

rgba = np.zeros((256, 256, 4), dtype=np.uint8)
rgba[..., 3] = 255

# 返回 bytes
jxl_bytes = jxlpy.encode(rgba)

# 写入文件
jxlpy.encode(rgba, output="out/image.jxl")
```

### 从文件路径/bytes 编码

```python
# 路径 → JXL（JPEG 输入自动走 lossless transcode）
jxl_bytes = jxlpy.encode("input.png")
jxl_bytes = jxlpy.encode("input.jpg")

# bytes → JXL
with open("input.jpg", "rb") as f:
    jxl_bytes = jxlpy.encode(f.read())
```

### 从 torch tensor 编码

```python
import torch

tensor = torch.zeros(256, 256, 4, dtype=torch.uint8)
jxl_bytes = jxlpy.encode(tensor)

# CHW layout
tensor_chw = torch.zeros(4, 256, 256, dtype=torch.uint8)
jxl_bytes = jxlpy.encode(tensor_chw, layout="chw")
```

### 编码参数

```python
jxlpy.encode(
    src,
    output=None,          # 输出路径，None 返回 bytes
    layout="auto",        # "auto" | "hwc" | "chw"
    lossless=None,        # None → 根据 distance 自动判断
    distance=None,        # None → lossless；> 0 → lossy
    alpha_distance=0.0,   # alpha 通道距离
    effort=7,             # 编码努力（1-10）
    modular=None,         # None=auto, 0=VarDCT, 1=modular
    level=-1,             # -1=auto, 5 或 10
    threads=0,            # 0=默认线程数
    use_container=False,  # 是否使用容器格式
    bits_per_sample=0,    # 0=自动
    extra_channels=None,  # 附加通道列表
)
```

**默认行为**：未传 `distance` 时 `lossless=True`。传入 JPEG 文件时自动使用 lossless JPEG transcode。

### 支持的 dtype

| numpy dtype | 用途 |
|---|---|
| `uint8` | 8-bit 标准 |
| `uint16` | 16-bit HDR/深度 |
| `float16` | 半精度浮点 |
| `float32` | 全精度浮点 |

---

## 解码

### 解码为 numpy

```python
arr = jxlpy.decode("image.jxl")                 # numpy ndarray (H, W, C)
arr = jxlpy.decode(jxl_bytes)                    # 从 bytes 解码
arr = jxlpy.decode("image.png")                  # PNG 也支持
```

### 解码为 torch

```python
tensor = jxlpy.decode(jxl_bytes, out="torch")    # torch.Tensor
```

### 带元信息

```python
arr, meta = jxlpy.decode(jxl_bytes, return_info=True)
# meta: {"xsize", "ysize", "num_channels", "num_frames", "have_animation", ...}
```

### 解码为 PNG/JPEG 文件

```python
# JXL → PNG bytes
png_bytes = jxlpy.decode_to_png("image.jxl")

# JXL → PNG 文件
jxlpy.decode_to_png("image.jxl", output="out/image.png")

# JXL → JPEG bytes（可设置质量）
jpeg_bytes = jxlpy.decode_to_jpeg("image.jxl", quality=90)

# JXL → JPEG 文件
jxlpy.decode_to_jpeg("image.jxl", output="out/image.jpg", quality=95)
```

### JPEG 无损还原

当 JXL 文件是通过 JPEG lossless transcode 产生的，`decode_to_jpeg` 会自动尝试提取原始 JPEG（bit-exact）：

```python
# JPEG → JXL → JPEG (bit-exact roundtrip)
jxl_bytes = jxlpy.encode("photo.jpg")       # 默认 lossless transcode
original = jxlpy.decode_to_jpeg(jxl_bytes)   # 自动走 JPEG reconstruction

# 显式调用（如果 JXL 不含 reconstruction data 会抛 RuntimeError）
original = jxlpy.reconstruct_jpeg(jxl_bytes)
```

如果 JXL 不含 JPEG reconstruction data（例如从像素编码），`decode_to_jpeg` 会自动 fallback 到重新编码 JPEG。

### 通用格式转换

```python
# 支持: png, jpg/jpeg, ppm, pgm, pam, pfm, pgx
jxlpy.convert("input.jxl", output="out.png", format="png")
jxlpy.convert("input.png", output="out.jpg", format="jpg", quality=85)
```

所有格式转换支持任意方向：PNG→JXL→JPEG、JPEG→JXL→PNG 等。

---

## 多帧分析

在编码前分析帧序列，判断多帧打包是否有收益：

```python
result = jxlpy.analyze_multiframe(frames)
print(result["recommendation"])  # "highly_beneficial" / "moderately_beneficial" / "minimal_benefit"
print(result["avg_bbox_pct"])    # 平均 diff bbox 占画布百分比
```

返回每帧的变化统计：

```python
for f in result["frames"]:
    print(f"frame {f['index']}: bbox={f['bbox_pct']:.1f}%, changed={f['changed_pct']:.1f}%")
```

---

## 多帧 / 动画

### 编码

```python
frames = [frame0, frame1, frame2]  # numpy 数组列表

jxl_bytes = jxlpy.encode_multiframe(
    frames,
    durations=1,            # 统一 duration 或逐帧列表 [1, 2, 1]
    tps=(1000, 1),          # ticks per second (numerator, denominator)
    reference="auto",       # "auto" | "first" | "previous" | "none"
    min_crop_ratio=0.98,    # crop 面积 < 全图 × ratio 时启用裁剪
    lossless=True,
    effort=7,
)
```

帧可以是 numpy 数组、torch tensor、文件路径或 bytes。

### Delta 策略

| `reference` | 行为 |
|---|---|
| `"auto"` | 比较上一帧和首帧 reference，选 bbox 更小的 |
| `"first"` | 仅比较首帧 |
| `"previous"` | 仅比较上一帧 |
| `"none"` / `"full"` | 不做 delta，每帧全尺寸 |

- blend mode 固定使用 `JXL_BLEND_REPLACE`
- diff bbox 比较 **所有通道**（含 alpha=0 的 RGB 和 extra channels）
- 这保证 lossless 场景下 invisible pixels 也精确还原

### 解码指定帧

```python
# 读取合成完整帧（默认）
frame2 = jxlpy.decode(jxl_bytes, frame=2, coalesced=True)

# 读取原始 layer/crop（残差裁剪区域）
layer, meta = jxlpy.decode_layer(jxl_bytes, layer=1)
print(meta["layer_have_crop"])   # True
print(meta["crop_x0"], meta["crop_y0"])
print(layer.shape)               # 裁剪后的尺寸
```

---

## Extra Channels

用于深度图、mask、热图、训练标签等非颜色数据平面。

### 编码

```python
rgb = np.zeros((256, 256, 3), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
depth = np.zeros((256, 256), dtype=np.uint16)

jxl_bytes = jxlpy.encode(
    rgb,
    extra_channels=[
        ("mask", "selection_mask", mask),
        {"name": "depth", "type": "depth", "data": depth},
    ],
)
```

支持的 spec 格式：

```python
plane                                         # 匿名 unknown 类型
("name", plane)                               # 命名
("name", "type", plane)                       # 命名 + 类型
{"name": "...", "type": "...", "data": plane} # dict 形式
```

支持的类型：`alpha`, `depth`, `spot_color`, `selection_mask`, `black`, `cfa`, `thermal`, `unknown`, `optional`

### 解码

```python
# 单个 extra channel
plane, meta = jxlpy.decode_extra_channel(jxl_bytes, index=1)

# 主图 + 所有 extra channels
image, meta = jxlpy.decode(
    jxl_bytes,
    return_info=True,
    return_extra_channels=True,
)
for ch in meta["extra_channels"]:
    print(ch["name"], ch["type"], ch["data"].shape)
```

> **注意**：RGBA 图的 alpha 是 JXL extra channel index 0。用户附加的 extra channel 从 index 1 开始。默认 `decode(return_extra_channels=True)` 跳过 alpha extra，设 `include_alpha_extra=True` 包含。

### 多帧 extra channels

```python
masks = [mask0, mask1, mask2]  # 逐帧 list，或单个 array 广播到所有帧

jxl_bytes = jxlpy.encode_multiframe(
    frames,
    extra_channels=[("mask", "selection_mask", masks)],
)
```

---

## 元数据查询

```python
meta = jxlpy.info("image.jxl")
# {
#   "xsize": 1920,
#   "ysize": 1080,
#   "num_channels": 4,
#   "bits_per_sample": 8,
#   "dtype": dtype('uint8'),
#   "num_frames": 10,
#   "have_animation": True,
#   "num_extra_channels": 2,
#   ...
# }
```

也接受 PNG/JPEG 输入。

---

## 测试

### 基础 smoke test

```bat
.\scripts\smoke_jxlpy.cmd
```

```bash
python scripts/smoke_jxlpy.py
```

### 完整测试套件

```bash
python scripts/test_multilayer.py     # 多帧/多图层全面测试
python scripts/test_cross_format.py   # 跨格式转换 (PNG↔JXL↔JPEG)
python scripts/test_new_features.py   # JPEG reconstruction + analyze
```

测试覆盖：
- 单帧 lossless roundtrip（uint8/uint16/float32）
- 多帧 delta crop + reference modes（auto/first/previous/none）
- Extra channels 编解码
- JPEG lossless transcode roundtrip（bit-exact）
- PNG→JXL→JPEG / JPEG→JXL→PNG 跨格式
- 多线程/多进程并发安全
- Layer/crop 非合成帧读取

---

## Wheel 打包

### Windows

```bat
.\scripts\build_wheel.cmd
```

### Linux / macOS

```bash
./scripts/build_wheel.sh
```

### 打包流程

1. 构建 `jxlpy_native`（静态链接 libjxl）
2. `pip wheel . --no-deps --wheel-dir dist`
3. `setup.py` 的 `build_py` hook 从 `out/build/` 查找并复制 native 库到 wheel

### 产物

```text
dist/jxlpy-0.1.0-cp310-cp310-win_amd64.whl
```

wheel 内部结构：

```
jxlpy/
├── __init__.py
├── _ffi.py
├── api.py
└── jxlpy_native.dll    (或 .so / .dylib)
```

### 环境变量

| 变量 | 作用 |
|---|---|
| `PYTHON_EXE` | 指定 Python 解释器路径 |
| `CMAKE_EXE` | 指定 CMake 路径（Windows） |
| `NINJA_EXE` | 指定 Ninja 路径（Windows） |
| `CMAKE_C_COMPILER` | C 编译器（Linux/macOS） |
| `CMAKE_CXX_COMPILER` | C++ 编译器（Linux/macOS） |
| `JXLPY_NATIVE_LIB` | 运行时直接指定 native 库路径 |

### 安装 wheel

```bash
pip install dist/jxlpy-0.1.0-*.whl
```

安装后不需要设置 `JXLPY_NATIVE_LIB`，native 库已包含在 wheel 内。

---

## 架构

```
jxlpy/
├── __init__.py         # 公开 API 导出
├── _ffi.py             # CFFI dlopen 加载 native 库
└── api.py              # Python 高层接口 (encode/decode/convert/...)

native/
├── jxlpy_native.h      # C ABI 声明 (JXLPY_EXPORT)
└── jxlpy_native.cc     # 调用 libjxl C API + libjxl extras

scripts/
├── build_windows.cmd   # Windows native 构建
├── build_linux.sh      # Linux native 构建
├── build_macos.sh      # macOS native 构建
├── build_wheel.cmd     # Windows wheel 打包
├── build_wheel.sh      # Unix wheel 打包
├── smoke_jxlpy.py      # 基础 smoke test
├── test_multilayer.py   # 多帧完整测试
├── test_cross_format.py # 跨格式转换测试
└── test_new_features.py # JPEG reconstruction + analyze 测试
```

Python 层通过 CFFI `dlopen` 加载编译好的 `jxlpy_native`（dll/so/dylib），不依赖 CLI 工具。native 层静态链接 libjxl，无外部运行时依赖。

---

## 注意事项

- **透明图精确性**：lossless 模式下 diff 比较包含 alpha=0 区域的 RGB，使用 `REPLACE + crop` 而非 `BLEND`，保证 invisible pixels 精确还原。
- **layout 歧义**：对于 `(4, 4, 4)` 等方形小尺寸 array，auto layout 可能误判通道轴。请显式传 `layout="chw"` 或 `layout="hwc"`。
- **CLI 独立**：`cjxl`/`djxl`/`jxlinfo`/`jxltran` 与 `jxlpy_native` 共享同一 libjxl 源码树但各自独立构建，互不影响。
- **JPEG reconstruction**：仅当 JXL 通过 `encode(jpeg_bytes)` 默认 lossless 模式产生时才包含 reconstruction data。如果用 `encode(pixels)` 从像素编码，`reconstruct_jpeg` 会报错，`decode_to_jpeg` 会 fallback 到重编码。
- **线程安全**：每次 encode/decode 创建独立的 libjxl encoder/decoder 实例，可安全并发调用。

---

## 限制

- wheel CI（`cibuildwheel`）尚未配置，当前手动构建。
- Linux/macOS native wheel 脚本已就位但尚未实际验证。
- 多帧 delta 基于 bbox 面积启发式，非熵编码后最优搜索。
- 不支持 signed int16；需自行映射到 `uint16` 或 float。
- `decode(return_extra_channels=True)` 对每个 extra channel 单独解码文件，大量 extra channels 时性能不佳。
- `effort` 在某些特殊图案下（如纯色块），lossy 高 effort 反而可能产生更大文件（VarDCT 路径选择差异）。
