# JXL 多帧/图层实验记录

本文记录当前项目里关于 libjxl 构建、多帧 layer、delta/crop、patches、JPG 多帧策略和 benchmark 的结论。后续上下文不足时，先读这个文件。

## 当前状态

- 工作区：`D:\cpp_pj\jxl_python`
- libjxl 源码目录：`libjxl`
- Windows CLI 已构建成功：
  - `out/build/windows-clang-cl-cli/libjxl/tools/cjxl.exe`
  - `out/build/windows-clang-cl-cli/libjxl/tools/djxl.exe`
  - `out/build/windows-clang-cl-cli/libjxl/tools/jxlinfo.exe`
  - `out/build/windows-clang-cl-cli/libjxl/tools/jxltran.exe`
- Python 测试环境：
  - `C:\Users\autumn\.conda\envs\py10\python.exe`
  - 已确认有 Pillow 10.0.1

根目录已有的构建辅助文件：

- `CMakeLists.txt`
- `CMakePresets.json`
- `.gitignore`
- `README.md`

已加 benchmark 脚本：

- `scripts/benchmark_jxl_layers.py`

测试数据：

- `test_img/mt_lay/t1`：PNG 差分图，4 张，4096x4096 RGBA
- `test_img/mt_lay/t2`：JPG 差分图，4 张，2000x2183 RGB
- `test_img/mt_lay/t3`：不同 JPG，3 张，尺寸不完全一致

输出报告：

- `out/mt_lay_benchmark/results.md`
- `out/mt_lay_benchmark/results.csv`
- `out/mt_lay_benchmark/diff_stats.json`

脚本默认复用已有输出，不会重复生成旧 JXL。需要强制重跑时加 `--force`。

## JXL 多帧结论

JPEG XL 支持多帧、动画、零时长 frame、crop、blend、reference。

语义上要区分：

- `duration = 0`：layer / composite still，不是动画播放帧
- `duration > 0`：动画帧
- `duration = 0xFFFFFFFF`：page，支持情况不如普通动画/图层

解码端默认会 coalesce，也就是把 layer 合成成完整图。如果要作为图层容器读取原始 layer，需要关闭 coalescing：

```c
JxlDecoderSetCoalescing(dec, JXL_FALSE);
```

Python 包装后面应该暴露两种读取方式：

```python
read_frames(path, coalesced=True)   # 返回完整重建后的每帧
read_layers(path, coalesced=False)  # 返回原始 crop/layer 信息
```

## Extra Channels / 位深

当前 libjxl 编码器实际限制：

- Level 5：最多 4 个 extra channels
- Level 10：源码检查 `num_extra_channels > 256` 会失败

所以不要按 4096 extra channels 设计 Python API。4096/4099 更像格式宣传或理论 component 上限，不是当前 libjxl encoder 可直接承诺的能力。

16-bit：

- 支持 `JXL_TYPE_UINT16`
- `bits_per_sample = 16`
- `exponent_bits_per_sample = 0`

注意它是 unsigned 16-bit。signed int16 差分图需要自己定义映射，例如 `value + 32768`，或用 float buffer。

## Patches 结论

`--patches=1` 是有效功能，但不是视频式跨帧差分。

sanity check 使用 libjxl 自带图：

```text
libjxl/testdata/jxl/grayscale_patches.png
```

结果：

```text
lossless patches=0: 14315 B
lossless patches=1:  4727 B

lossy d=1 patches=0: 42484 B
lossy d=1 patches=1: 21133 B
```

所以 patcher 没坏。

但 `mt_lay` 中 `patches=0` 和 `patches=1` 大小完全一样。原因是 patches 当前更擅长同一帧/编码上下文内的重复 text/tiles/规则块，不会稳定地自动利用“上一帧背景相同、人物变化”这种跨帧相似性。

结论：

```text
patches = 附加优化
delta/crop layer = 多帧差分主力方案
```

## Inter-frame / Delta 结论

已查代码和 issue。当前 libjxl/cjxl encoder 基本按给定 frame 编码，不会主动：

```text
比较上一帧
找变化 bbox
生成透明差分帧
写 crop layer
```

`save_as_reference` / `blend_info.source` 也不是自动差分。它只是告诉解码器：

```text
前一帧保存为 reference 1
当前帧以 reference 1 为底进行 blend/replace/add
```

当前帧编码内容仍然由我们决定。如果给 full-size opaque frame，它就编码整张图。

正式方案需要包装层自己做 delta 预处理，不是重写 JXL 编码器。

## Delta 实现路线

第一版建议只做两种简单稳妥策略。

### 1. bbox delta / crop layer

逻辑：

```text
prev_frame vs cur_frame
找 changed pixel 的最小矩形 bbox
如果 bbox 面积足够小：
    只编码 bbox 内像素
    设置 crop_x0 / crop_y0 / xsize / ysize
否则：
    存 full frame
```

JXL frame header 关键字段：

```c
frame_header.duration = 0;  // layer 模式；动画则用实际 duration
frame_header.layer_info.have_crop = JXL_TRUE;
frame_header.layer_info.crop_x0 = x0;
frame_header.layer_info.crop_y0 = y0;
frame_header.layer_info.xsize = w;
frame_header.layer_info.ysize = h;
frame_header.layer_info.blend_info.blendmode = JXL_BLEND_REPLACE;
```

如果要引用前一帧：

```c
previous_frame.layer_info.save_as_reference = 1;
current_frame.layer_info.blend_info.source = 1;
```

### 2. transparent-black delta

逻辑：

```text
完整画布
与上一帧相同的像素设成 RGBA(0,0,0,0)
变化像素保留
blendmode = JXL_BLEND_BLEND
```

适合变化像素分散但数量少的情况。半透明内容要谨慎，容易和 alpha blend 语义冲突。

### 暂不建议第一版做复杂预测

先不做：

- motion estimation
- block matching
- residual frame
- kAdd float residual
- 多参考帧搜索

这些成本高，调试复杂。第一版先用 bbox/透明 delta。

## APNG/Pillow 中转说明

当前 benchmark 的 `apng_delta_*` 是原型路径：

```text
完整图片序列
-> Pillow 保存 APNG
-> Pillow/APNG writer 自动算 bbox
-> cjxl 读取 APNG 并保留 crop/blend
-> JXL layer
```

这只是验证收益，不是正式依赖。正式 Python 包装不应该依赖 Pillow/APNG 中转，应该直接：

```text
numpy/内存图像
-> 自己算 bbox/transparent delta
-> CFFI 调 JxlEncoderSetFrameHeader / JxlEncoderAddImageFrame
```

## Benchmark 结果摘要

详细结果见 `out/mt_lay_benchmark/results.md`。

### t1：PNG / RGBA / 稀疏差分

```text
original_sources:             302.28 KiB
single_source_jxl_sum:         22.64 KiB
apng_full_patches0:            22.62 KiB
apng_full_patches1:            22.62 KiB
apng_delta_patches0:            9.28 KiB
apng_delta_patches1:            9.28 KiB
lossy_d1_apng_delta_patches0:  91.16 KiB
```

差分覆盖率：

```text
bbox area:       0.10%, 1.17%, 53.71%
changed pixels:  0.10%, 0.29%, 5.37%
```

结论：

- lossless modular + delta/crop 最优
- 有损 `distance=1` 反而更大，因为这类稀疏/透明/平面图更适合 lossless
- full 多帧和单张 JXL 求和几乎一样，说明 libjxl 没自动吃跨帧相似性

### t2：JPG 差分

```text
original_sources:             4.57 MiB
single_source_jxl_sum:        3.69 MiB
single_pixel_jxl_sum:         3.63 MiB
single_lossy_d1_jxl_sum:      1.23 MiB
apng_full_patches0:           4.63 MiB
apng_delta_patches0:          4.61 MiB
lossy_d1_apng_delta_patches0: 1.26 MiB
```

差分覆盖率：

```text
bbox area:       75.2%, 75.2%, 75.2%
changed pixels:  27.06%, 29.01%, 26.38%
```

结论：

- JPG 精确像素差分不划算
- JPEG 噪声/DCT/重保存误差会让 changed pixels 和 bbox 很大
- 有损单张 JXL 求和最小

### t3：不同 JPG / 尺寸不一致

```text
original_sources:             1.24 MiB
single_source_jxl_sum:        972.64 KiB
single_pixel_jxl_sum:         499.41 KiB
single_lossy_d1_jxl_sum:      445.30 KiB
apng_full_patches0:           499.20 KiB
apng_delta_patches0:          499.20 KiB
lossy_d1_apng_full_patches0:  445.54 KiB
```

结论：

- 图像不相关时，多帧基本没有额外收益
- 有损单张求和略优或持平

## JPG 多帧策略

JPG 要和 PNG/无损图分开处理。JPG 即使视觉上背景相同，像素层面也常常大量不同：

- 压缩噪声
- DCT ringing
- 色度子采样误差
- 重保存导致的微小变化

因此不要默认精确 delta。

候选策略：

```text
A. JPEG lossless transcode per image
   可还原原 JPG 文件，不能吃跨帧相似性

B. Decode to pixels -> lossless JXL per frame
   只保留像素，不可还原原 JPG bitstream

C. Decode to pixels -> lossy JXL
   照片类通常最小，但再次有损

D. Fuzzy delta/crop
   对近似相同背景可能有效，需要阈值和质量策略
```

注意：

`single_pixel_jxl_sum` 比原始 JPG 小是正常的。它保存的是 JPG 解码后的 RGB 像素，不保留原始 JPEG bitstream。它不能 byte-for-byte 还原原 JPG。

推荐 auto 逻辑：

```text
if preserve_jpeg:
    用 JPEG lossless transcode
elif lossy_allowed:
    试 lossy JXL
else:
    pixel lossless JXL，必要时试 fuzzy delta
```

对 JPG 的 auto 搜索可先做采样估算：

```text
exact_changed_ratio
bbox_area_ratio
fuzzy_changed_ratio(threshold=2/4/8)
fuzzy_bbox_area_ratio
```

再对少量帧试编码，选最小方案。

## 后续 Python/CFFI 包装建议

初版 API 形态：

```python
jxl.save_frames(
    path,
    frames,
    lossless=True,
    delta="auto",
    keyframe_interval=30,
    durations=None,
    preserve_jpeg=False,
)
```

建议行为：

- `durations is None`：按 layer/composite still 保存，`duration=0`
- `durations` 有值：按 animation 保存
- `delta="auto"`：
  - synthetic/PNG/RGBA 优先 bbox delta
  - 变化分散时考虑 transparent-black delta
  - JPG 默认不做精确 delta，先评估 lossy/pixel/JPEG transcode
- 每隔 `keyframe_interval` 插 full frame，便于随机访问和容错
- 对读取提供 coalesced / raw layer 两种模式

## 当前重要判断

1. libjxl 构建和 CLI 路线已经跑通。
2. JXL 格式支持多帧、layer、crop、blend、reference。
3. 当前 encoder 不自动做跨帧差分。
4. `patches` 可用，但不是多帧差分主力。
5. PNG/RGBA/synthetic 稀疏图：lossless delta/crop 很强。
6. JPG/照片：优先 lossy JXL 或 pixel JXL；是否 preserve original JPG 是关键分支。
7. 正式实现需要自己写 delta 预处理，然后直接调用 libjxl encoder API。
