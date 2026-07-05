# jxlpy Python/CFFI 使用说明

本文记录当前 `jxlpy` 包的构建方式、API 形状和多帧 delta 语义。

## 当前定位

`jxlpy` 是本仓库里围绕 libjxl 的 Python/CFFI 包装层：

- native 层：`native/jxlpy_native.cc`
- CFFI loader：`jxlpy/_ffi.py`
- Python API：`jxlpy/api.py`

Python 层不调用 `cjxl.exe` / `djxl.exe`。它通过 CFFI 加载
`jxlpy_native.dll`，native 层再调用 libjxl C API 和 libjxl extras。

当前已跑通：

- numpy `uint8` RGBA 单帧 encode/decode 无损回读
- 多帧 `REPLACE + crop` delta encode
- coalesced 完整帧读取
- non-coalesced layer/crop 读取
- torch 懒加载适配

## 构建脚本

### Windows

默认使用用户当前机器上的 CLion CMake/Ninja 路径：

```bat
.\scripts\build_windows.cmd jxlpy_native
```

脚本默认值：

```bat
CMAKE_EXE=C:\Program Files\JetBrains\CLion 2025.2.4\bin\cmake\win\x64\bin\cmake.exe
NINJA_EXE=C:\Program Files\JetBrains\CLion 2025.2.4\bin\ninja\win\x64\ninja.exe
BUILD_DIR=out\build\windows-clang-cl-cli
TARGET=jxlpy_native
```

如果路径不同，可以覆盖环境变量：

```bat
set CMAKE_EXE=D:\path\to\cmake.exe
set NINJA_EXE=D:\path\to\ninja.exe
.\scripts\build_windows.cmd jxlpy_native
```

构建产物：

```text
out/build/windows-clang-cl-cli/jxlpy_native.dll
```

### Linux

```bash
./scripts/build_linux.sh jxlpy_native
```

默认使用：

```text
clang / clang++
out/build/linux-clang-python
```

可覆盖：

```bash
CMAKE_C_COMPILER=gcc CMAKE_CXX_COMPILER=g++ ./scripts/build_linux.sh jxlpy_native
```

### macOS

```bash
./scripts/build_macos.sh jxlpy_native
```

默认使用：

```text
clang / clang++
out/build/macos-clang-python
```

Linux/macOS 脚本是按跨平台 CMake target 写的，当前仓库已提供入口，但本机只验证了 Windows。

### Wheel

Windows：

```bat
.\scripts\build_wheel.cmd
```

Linux/macOS：

```bash
./scripts/build_wheel.sh
```

wheel 构建逻辑：

- 先构建 `jxlpy_native`
- `setup.py bdist_wheel` 会从 `out/build/**` 查找当前平台 native 库
- 找到后复制到 wheel 的 `jxlpy/` 包目录中

默认 native target 使用 `BUILD_SHARED_LIBS=OFF`，也就是尽量把 libjxl 静态链接进 `jxlpy_native`。如果改成 shared lib wheel，还需要额外处理依赖库拷贝、rpath/auditwheel/delocate。

## Python 环境

当前验证使用：

```text
C:\Users\autumn\.conda\envs\py10\python.exe
```

依赖：

- `numpy`
- `cffi`
- `torch` 可选；只有传入 tensor 或 `out="torch"` 时才 import

smoke test：

```bat
.\scripts\smoke_jxlpy.cmd
```

期望输出包含：

```text
single_shape (16, 16, 4) uint8 True
multi_frame2 (16, 16, 4) uint8 True
layer1 (4, 4, 4) True 4 4
```

## 基础 API

```python
import jxlpy
import numpy as np

rgba = np.zeros((256, 256, 4), dtype=np.uint8)
rgba[..., 3] = 255

jxl_bytes = jxlpy.encode(rgba)
decoded = jxlpy.decode(jxl_bytes)
```

写入文件：

```python
jxlpy.encode(rgba, output="out/image.jxl")
```

从路径或 bytes 编码：

```python
jxl_bytes = jxlpy.encode("test_img/input.png")

with open("test_img/input.jpg", "rb") as f:
    jxl_bytes = jxlpy.encode(f.read())
```

读取为 torch：

```python
tensor = jxlpy.decode(jxl_bytes, out="torch")
```

返回元信息：

```python
arr, meta = jxlpy.decode(jxl_bytes, return_info=True)
print(meta["xsize"], meta["ysize"], meta["num_frames"])
```

## 编码参数

高层 API 默认走安全无损：

- 数组/PNG：pixel-lossless
- JPEG 文件：lossless JPEG transcode 优先
- 有损必须显式传 `distance`

常用参数：

```python
jxlpy.encode(
    rgba,
    lossless=True,
    effort=7,
    modular=None,
    level=-1,
    threads=0,
)
```

有损示例：

```python
jxlpy.encode(
    rgba,
    lossless=False,
    distance=1.0,
    effort=7,
)
```

参数说明：

- `lossless`: 是否无损；默认 `distance is None` 时为 True
- `distance`: libjxl 感知距离；传入后可启用有损
- `alpha_distance`: alpha 通道距离
- `effort`: 编码努力等级
- `modular`: `None` 表示交给 libjxl；`0/1` 强制关闭/开启
- `level`: `-1` 自动；`5` 或 `10` 显式指定 codestream level
- `threads`: `0` 使用 libjxl 默认线程数

## 多帧 / 图层

多帧编码：

```python
frames = [frame0, frame1, frame2]
jxl_bytes = jxlpy.encode_multiframe(
    frames,
    durations=1,
    tps=(1000, 1),
    reference="auto",
)
```

读取完整合成帧：

```python
frame2 = jxlpy.decode(jxl_bytes, frame=2, coalesced=True)
```

读取内部 layer/crop：

```python
layer, meta = jxlpy.decode_layer(jxl_bytes, layer=1)
print(meta["layer_have_crop"], meta["crop_x0"], meta["crop_y0"])
```

当前 delta 策略：

- 第一帧保存完整图
- 后续帧比较参考帧并计算 changed bbox
- bbox 足够小时只写 crop layer
- blend mode 使用 `JXL_BLEND_REPLACE`
- 默认 `reference="auto"` 会在上一帧 reference 和首帧 reference 中选 bbox 更小的候选

这不是 libjxl 自动跨帧预测，而是 wrapper 自己做预处理。libjxl 本身不会稳定自动把 full-size 多帧变成差分层。

## 透明图和精确性

lossless 场景默认按字节精确思路处理：

- diff bbox 比较所有通道
- 包括 `alpha=0` 区域里的 RGB
- 不用 `BLEND transparent black trick` 作为主方案
- 精确还原优先使用 `REPLACE + crop`

这对训练数据、归档数据和需要保留 invisible RGB 的 RGBA 图比较重要。

## Extra Channels

主图像数组只自动识别 1/2/3/4 通道：

- `H x W`
- `H x W x C`
- `C x H x W`，当 `C <= 4` 时可自动识别

`extra_channels` 不等于“自动拆 RGB/RGBA”。它应该用于深度、mask、第二 alpha、热图、训练标签等非颜色平面。

显式 extra channel API 已实现：

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

支持的 spec 形式：

```python
extra_channels=[plane]
extra_channels=[("name", plane)]
extra_channels=[("name", "selection_mask", plane)]
extra_channels=[{"name": "depth", "type": "depth", "data": plane}]
```

支持的 `type`：

```text
alpha, depth, spot_color, selection_mask, black, cfa, thermal, unknown, optional
```

读取单个 extra channel：

```python
mask, meta = jxlpy.decode_extra_channel(jxl_bytes, 1)
print(mask.shape, meta["extra_channel_type"], meta["extra_channel_name"])
```

注意：RGBA 图的 alpha 在 JXL 里也是 extra channel，通常 index 0 是 alpha；用户传入的第一个 extra channel 会排在 alpha 后面。所以 RGBA + 一个 mask 时，mask 通常是 index 1。

读取主图时一起返回非 alpha extra channels：

```python
image, meta = jxlpy.decode(
    jxl_bytes,
    return_info=True,
    return_extra_channels=True,
)

for channel in meta["extra_channels"]:
    print(channel["name"], channel["type"], channel["data"].shape)
```

默认会跳过 alpha extra，避免和主图 RGBA alpha 重复。需要包含 alpha 时：

```python
image, meta = jxlpy.decode(
    jxl_bytes,
    return_info=True,
    return_extra_channels=True,
    include_alpha_extra=True,
)
```

多帧 extra channel 支持逐帧数组。wrapper 会用主图同一个 crop bbox 裁剪 extra plane：

```python
jxl_bytes = jxlpy.encode_multiframe(
    frames,
    reference="auto",
    extra_channels=[
        ("mask", "selection_mask", [mask0, mask1, mask2]),
    ],
)
```

仍然不把任意 `C` 维 tensor 自动全部塞进 JXL extra channels，原因是：

- 会改变 JXL level 需求
- 解码时需要保留每个通道的语义
- Level 5 只适合少量 extra channels
- Level 10 兼容性更弱

## 当前限制

- wheel 打包脚本已提供，但还没有做完整 `cibuildwheel` CI 流水线。
- Linux/macOS 构建脚本已提供，但 Python native wheel 尚未在这些平台验证。
- 多帧 `reference="auto"` 是 bbox 面积启发式，不是熵编码后真实大小搜索。
- 暂未做 signed int16 语义；需要用户自行映射到 `uint16` 或使用 float。
