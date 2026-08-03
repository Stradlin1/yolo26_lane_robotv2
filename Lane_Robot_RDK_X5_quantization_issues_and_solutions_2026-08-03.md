# Lane Robot 模型在 RDK X5 量化中遇到的问题与解决思路

> 日期：2026-08-03  
> 工程：`lane_robot`  
> 工具链：OpenExplorer / `hb_mapper 1.24.3` / `hbdk 3.49.15` / `bayes-e`  
> 当前目标：将四线 Lane Robot ONNX 模型转换为 RDK X5 可运行的 NV12 Runtime BIN，并尽可能让主要计算落在 BPU。

---

## 1. 当前模型基本信息

### 1.1 ONNX 输入输出

```text
输入名称：images
输入 shape：[1, 3, 320, 320]
输入类型：float32
输入布局：NCHW

输出名称：lane_output
输出 shape：[1, 322, 56, 4]
输出类型：float32

ONNX Opset：11
IR version：6
节点数：758
```

### 1.2 原始推理预处理

推理脚本中的默认预处理为：

```text
原始图片
→ EXIF 方向修正
→ 转为 RGB
→ 直接 resize 到 320×320，INTER_LINEAR
→ float32
→ 除以 255
→ HWC 转 CHW
→ 增加 batch
→ [1,3,320,320]
```

默认不使用 letterbox。只有运行推理脚本时显式传入 `--letterbox`，才会启用顶部补黑、底部对齐的自定义 letterbox。

因此，默认量化输入配置应使用：

```yaml
input_type_train: rgb
input_layout_train: NCHW
norm_type: data_scale
scale_value: 0.003921568627451
```

其中：

```text
0.003921568627451 = 1 / 255
```

### 1.3 输出语义

输出：

```text
lane_output [1,322,56,4]
```

实际拆分为：

```text
cls_logits = lane_output[:, 0:321, :, :]
offset     = lane_output[:, 321:322, :, :]
```

即：

```text
cls_logits [1,321,56,4]
offset     [1,1,56,4]
```

分类通道含义：

```text
0～319：320 个横向网格位置
320：no-lane 类别
```

最后一个通道是位置微调 `offset`。

模型没有单独的 confidence 输出头。置信度来自 `cls_logits` 经 Softmax 后的概率：

```text
点存在置信度 = 1 - P(no-lane)
位置置信度   = max(P(grid 0～319))
```

---


。




## 6. 问题四：分类输出层 `cls_fc2/Gemm` 无法进入 BPU

### 6.1 现象

工具链反复报告：

```text
/model/model.16/cls_fc2/Gemm
feature size in axis 3 should in range [1, 65536].
But given size 71904
```

最终节点表：

```text
/model/model.16/cls_fc2/Gemm
CPU
float
Cosine Similarity: 0.999832
```

### 6.2 原因

分类头一次性输出：

```text
321 × 56 × 4 = 71904
```

而当前 BPU 对该算子的相关维度限制为：

```text
最大 65536
```

因此：

```text
71904 > 65536
```

这个大 `Gemm` 无法编译到 BPU，只能回退到 CPU。

### 6.3 为什么日志重复很多次

量化工具链会在多个阶段重复检查算子约束，例如：

```text
模型检查
校准
量化
图优化
模型转换
运行时编译
```

所以同一个 `71904 > 65536` 会多次出现，并不代表出现了多个不同错误。

### 6.4 当前执行结构

当前模型大致为：

```text
NV12 输入
→ 主干网络：BPU INT8
→ 注意力 Softmax：指定 BPU
→ cls_fc2：CPU float
→ offset_fc：BPU INT8
→ 尾部 Reshape / Concat：CPU
→ lane_output [1,322,56,4]
```

其中 offset 分支：

```text
/model/model.16/offset_fc/Gemm
BPU
INT8
Cosine Similarity: 0.997696
```

### 6.5 当前保守方案

先接受混合执行：

```text
主干 + offset：BPU
注意力 Softmax：BPU
分类头 cls_fc2：CPU
```

优点：

- 不修改模型结构；
- 能快速完成第一版板端验证；
- 保留分类 logits 的 float 精度；
- 已成功生成 runtime BIN。

缺点：

- CPU 分类头可能增加延迟；
- 存在 BPU 与 CPU 间的数据传输；
- 无法做到全 BPU 推理。

适用条件：

```text
hb_perf 和板端实测表明帧率可接受
```

---

## 7. 解决分类头过大问题的几种思路

## 7.1 方案 A：拆成两个分类头，推荐

原结构：

```text
共享特征
├── cls_fc2: 71904
└── offset_fc: 224
```

改为：

```text
共享特征
├── cls_fc2_01: 321 × 56 × 2 = 35952
├── cls_fc2_23: 321 × 56 × 2 = 35952
└── offset_fc: 1 × 56 × 4 = 224
```

每个分类头满足：

```text
35952 < 65536
```

推荐 ONNX 输出：

```text
cls_01 [1,321,56,2]
cls_23 [1,321,56,2]
offset [1,1,56,4]
```

板端后处理：

```text
沿 lane 维拼接 cls_01 和 cls_23
→ 得到 cls_logits [1,321,56,4]
→ Softmax
→ no-lane 判断
→ Top-K soft-argmax
→ 加 offset
```

优点：

- 两个 `Gemm` 都有机会进入 BPU；
- 输出数量适中；
- 分类和 offset 语义明确；
- 后续更容易分别调试量化误差。

缺点：

- 必须真正修改 PyTorch 车道头；
- 需要正确迁移旧权重；
- 需要验证新旧 FP32 输出严格等价。

## 7.2 方案 B：拆成四个分类头

结构：

```text
cls_lane0 [1,321,56,1]
cls_lane1 [1,321,56,1]
cls_lane2 [1,321,56,1]
cls_lane3 [1,321,56,1]
offset    [1,1,56,4]
```

每个分类头输出：

```text
321 × 56 = 17976
```

优点：

- 输出尺寸远低于 BPU 限制；
- 每条车道独立；
- 调试最清晰。

缺点：

- 输出头数量更多；
- ROS 后处理需要管理更多 tensor；
- 导出与权重迁移更复杂。

建议仅在两个头仍不能顺利上 BPU 时采用。

## 7.3 方案 C：减少横向网格数量

保持一个分类头时，需要满足：

```text
(X + 1) × 56 × 4 ≤ 65536
```

因此：

```text
X ≤ 291
```

即把当前 320 个横向网格减少到最多约 291 个。

优点：

- 仍可保留单分类头；
- 模型结构简单。

缺点：

- 改变输出定义；
- 横向离散精度降低；
- 训练标签、损失和解码逻辑都需要同步修改；
- 通常需要重新训练。

因此不推荐作为首选方案。

## 7.4 方案 D：直接修改 ONNX 图

理论上可以：

```text
拆旧 Gemm 权重和 bias
→ 新建两个 Gemm
→ 新建两个 Reshape
→ 暴露多个输出
→ 删除旧大 Gemm 和尾部 Concat
```

但不推荐作为长期工程方案，原因：

- 容易切错权重的内存排列；
- initializer、节点连接和 shape 维护复杂；
- 重新训练后难以复用；
- PyTorch 源码与 ONNX 结构不一致；
- 调试成本高。

更合适的路径是：

```text
修改 PyTorch Lane Head
→ 加载旧权重
→ 精确切分权重
→ 重新导出 ONNX
→ 验证 FP32 数学等价
→ 再量化
```

## 7.5 方案 E：只在大 Gemm 后添加 Split，无效

错误思路：

```text
Gemm 输出 71904
→ Split 成两个 35952
```

这不能解决问题，因为报错发生在原始 `Gemm` 生成 71904 元素时。

必须把一个大 Linear/Gemm 真正改成多个较小 Linear/Gemm。

## 7.6 `hb_model_modifier` 不能解决该问题

`hb_model_modifier` 可以删除部分输入输出附近的格式节点，但不能：

- 把 71904 的 Gemm 自动拆成两个；
- 改变 Linear 的输出通道数；
- 绕过 BPU 的 65536 限制；
- 自动建立多个分类头。

而且当前输出同时包含分类 logits 和 offset：

```text
lane_output [1,322,56,4]
```

删除尾部 Reshape 或 Concat 可能改变输出 shape、数据类型和后处理契约。因此当前阶段不建议使用 `hb_model_modifier` 修改该模型。

---

## 8. 分类头权重迁移思路

如果在 PyTorch 中拆头，理论上不必重新训练，只要正确切分旧权重。

原分类层：

```text
Linear(in_features, 71904)
```

旧权重：

```text
W [71904, in_features]
b [71904]
```

需要根据旧 forward 中真实的 `view/reshape/permute` 顺序判断权重排列。

### 情况一：lane 维在扁平输出中连续

假设旧逻辑类似：

```python
cls = cls_fc2(x)
cls = cls.view(batch, 4, 56, 321)
cls = cls.permute(0, 3, 2, 1)
```

那么每条 lane 连续长度：

```text
56 × 321 = 17976
```

可以连续切：

```text
前两条 lane：0 ～ 35951
后两条 lane：35952 ～ 71903
```

### 情况二：lane 维在扁平输出中交错

假设旧逻辑直接：

```python
cls = cls.view(batch, 321, 56, 4)
```

则需要先把权重输出维 reshape 成：

```text
[321,56,4,in_features]
```

再沿 lane 维切：

```text
[:, :, 0:2, :]
[:, :, 2:4, :]
```

最后重新 flatten 给两个新 Linear。

### 必须进行的等价性验证

同一输入分别运行旧 ONNX 和新 ONNX：

```text
old_cls
old_offset

new_cls_01
new_cls_23
new_offset
```

合并：

```text
new_cls = concat(new_cls_01, new_cls_23, lane_axis)
```

验收标准：

```text
分类 logits 最大绝对误差接近 0
offset 最大绝对误差接近 0
解码后的车道点完全一致
```

若误差明显，说明权重切分顺序或 reshape 顺序错误。

---

## 9. 当前推荐路线

### 阶段一：保守部署验证

继续使用当前成功生成的混合模型：

```text
lane_robot_nv12_softmax_bpu.bin
```

配置：

```text
NV12 Runtime 输入
RGB NCHW 训练输入语义
/255
注意力 Softmax 指定 BPU
cls_fc2 保留 CPU float
offset_fc 位于 BPU INT8
单输出 lane_output [1,322,56,4]
```

目标：

```text
确认模型可在板端正常加载
确认输出 shape 和类型
完成 C++ 后处理
验证车道线精度
测试真实 FPS 和 CPU 占用
```

### 阶段二：优化输出头

如果当前 CPU 分类头造成明显性能问题：

```text
修改 PyTorch Lane Head
→ 一个大分类头拆成两个小分类头
→ 迁移旧权重
→ 导出三个输出
→ 验证新旧 ONNX 等价
→ 再做 RDK X5 量化
```

推荐最终输出：

```text
cls_01 [1,321,56,2]
cls_23 [1,321,56,2]
offset [1,1,56,4]
```

---

## 10. 后处理要求

板端不能照搬语义分割后处理。需要实现：

```text
读取输出
→ 拆分 cls_logits 和 offset
→ 对 321 个分类通道做 Softmax
→ no-lane 概率判断
→ 对前 320 个网格做 Top-K soft-argmax
→ 加 offset
→ 得到连续横向网格坐标
→ 还原到原图坐标
→ 可选多项式平滑
```

当前 Python 参数：

```text
exist_thr = 0.5
topk = 5
offset clip = [-0.5, 0.5]
poly degree = 2
poly blend = 0.5
```

如果使用 direct resize，坐标直接按模型宽高比例映射回原图。

如果后续启用自定义 letterbox，则必须同时保持：

```text
训练预处理
ONNX 推理预处理
量化校准预处理
板端预处理
坐标恢复
```

完全一致。

---

## 11. 当前量化配置参考

```yaml
model_parameters:
  onnx_model: /workspace/rdkx5_quant/lane_robot/model/model.onnx
  march: bayes-e
  layer_out_dump: false
  working_dir: /workspace/rdkx5_quant/lane_robot/output_softmax_bpu
  output_model_file_prefix: lane_robot_nv12_softmax_bpu
  node_info: {"/model/model.10/m/m.0/attn/Softmax": {"ON": "BPU"}}

input_parameters:
  input_name: images
  input_shape: 1x3x320x320
  input_batch: 1

  input_type_rt: nv12
  input_layout_rt: ""

  input_type_train: rgb
  input_layout_train: NCHW

  norm_type: data_scale
  scale_value: 0.003921568627451

calibration_parameters:
  cal_data_dir: /workspace/rdkx5_quant/lane_robot/calibration_source
  cal_data_type: ""
  calibration_type: default
  preprocess_on: true

compiler_parameters:
  compile_mode: latency
  debug: false
  optimize_level: O3
  jobs: 8
```

若使用提前生成的校准 BIN，则替换为：

```yaml
calibration_parameters:
  cal_data_dir: /workspace/rdkx5_quant/lane_robot/calibration_bin
  cal_data_type: float32
  calibration_type: default
  preprocess_on: false
```

---

## 12. 后续检查命令

### 检查模型信息

```bash
hb_model_info \
  /workspace/rdkx5_quant/lane_robot/output_softmax_bpu/\
lane_robot_nv12_softmax_bpu.bin
```

### 查看关键节点分配

```bash
grep -E \
"/model/model.10/m/m.0/attn/Softmax|/model/model.16/cls_fc2/Gemm|/model/model.16/offset_fc/Gemm" \
/workspace/rdkx5_quant/lane_robot/logs/makertbin_softmax_bpu.log
```

### 性能测试

```bash
mkdir -p /workspace/rdkx5_quant/lane_robot/perf
cd /workspace/rdkx5_quant/lane_robot/perf

hb_perf \
  /workspace/rdkx5_quant/lane_robot/output_softmax_bpu/\
lane_robot_nv12_softmax_bpu.bin
```

### 查看可删除节点，仅检查

```bash
hb_model_modifier \
  /workspace/rdkx5_quant/lane_robot/output_softmax_bpu/\
lane_robot_nv12_softmax_bpu.bin
```

当前仅建议查看，不建议执行删除。

---

## 13. 最终结论

本次量化已经解决：

```text
ONNX 输入输出确认
RGB + NCHW + /255 预处理确认
NV12 Runtime 输入配置
NV12 对应 pyramid 输入源问题
node_info YAML 解析问题
注意力 Softmax 指定 BPU
模型成功转换为 runtime BIN
```

当前核心遗留问题：

```text
cls_fc2 输出 71904，超过 BPU 65536 限制，只能运行在 CPU
```

当前最现实的两条路线：

```text
路线 1：先使用 CPU 分类头的混合模型完成板端验证
路线 2：将 PyTorch 分类头真正拆成两个较小 Linear，再重新导出 ONNX 和量化
```

推荐先完成路线 1 的精度和性能实测，再根据 CPU 占用和端到端 FPS 决定是否投入时间修改输出头。
