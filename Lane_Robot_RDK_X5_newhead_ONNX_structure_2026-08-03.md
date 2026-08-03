# Lane Robot RDK X5 双分类头 ONNX 结构说明

> 日期：2026-08-03  
> Git 分支：`quant_correct`  
> 实现方案：问题四的方案 A，将四线分类输出拆成两个分类头  
> 导出脚本：`export_onnx_newhead.py`  
> 推理脚本：`infer_onnx_newhead.py`  
> 当前模型：`runs/lane/lane_n_baseline-2/weights/best_newhead.onnx`

---

## 1. 修改目的

旧 ONNX 使用一个分类全连接层：

```text
cls_fc2: 512 -> 321 × 56 × 4 = 71904
```

RDK X5 工具链要求相关 Gemm 输出维度不超过 65536：

```text
71904 > 65536
```

因此旧节点：

```text
/model/model.16/cls_fc2/Gemm
```

无法进入 BPU，只能回退到 CPU float。

新结构将四条车道按 `0/1` 和 `2/3` 拆成两个分类头：

```text
cls_fc2_01: 512 -> 321 × 56 × 2 = 35952
cls_fc2_23: 512 -> 321 × 56 × 2 = 35952
```

两个分类 Gemm 均满足：

```text
35952 < 65536
```

---

## 2. 当前 ONNX 基本信息

以下数据从实际生成的 `best_newhead.onnx` 读取：

| 项目 | 数值 |
|---|---:|
| ONNX IR version | 6 |
| ONNX opset | 11 |
| 图节点数 | 760 |
| Initializer 数量 | 96 |
| 文件大小 | 173945608 bytes，约 165.89 MB |
| 输入类型 | float32 |
| 输入布局 | NCHW |
| 输入 shape | `[1,3,320,320]` |

输入名称：

```text
images [1,3,320,320]
```

新 ONNX 不再返回旧的单个 `lane_output [1,322,56,4]`，而是直接暴露三个输出。

---

## 3. 整体结构

主干网络和多尺度特征融合保持不变，只修改 `LaneRobotV2` 的最终分类投影和 ONNX 输出接口。

```text
images [1,3,320,320]
        │
        ▼
YOLO26 backbone + neck
        │
        ▼
融合特征 [1,256,20,20]
        │
        ▼
conv_1x1
256 -> 8 channels
输出 [1,8,20,20]
        │
        ▼
ONNX opset-11 兼容池化
Slice + ReduceMean + Concat
输出 [1,8,8,10]
        │
        ▼
Flatten
输出 [1,640]
        │
        ▼
cls_fc1/Gemm
640 -> 512
        │
        ▼
ReLU
共享特征 [1,512]
        │
        ├──────────────────────────┬──────────────────────────┐
        ▼                          ▼                          ▼
cls_fc2_01/Gemm              cls_fc2_23/Gemm              offset_fc/Gemm
512 -> 35952                 512 -> 35952                 512 -> 224
        │                          │                          │
        ▼                          ▼                          ▼
Reshape                      Reshape                      Tanh
        │                          │                          │
        ▼                          ▼                          ▼
cls_01                       cls_23                      Reshape × 0.5
[1,321,56,2]                [1,321,56,2]                offset [1,1,56,4]
```

ONNX 图中实际分类节点名称：

```text
/model/model.16/cls_fc2_01/Gemm
/model/model.16/cls_fc2_23/Gemm
```

旧的大分类节点已经不存在：

```text
/model/model.16/cls_fc2/Gemm
```

offset 分支保持为：

```text
/model/model.16/offset_fc/Gemm
512 -> 224
Tanh
Reshape [1,1,56,4]
乘以 0.5
```

因此 offset 数值范围仍为：

```text
[-0.5, 0.5]
```

---

## 4. 三个 ONNX 输出

### 4.1 `cls_01`

```text
名称：cls_01
类型：float32
shape：[B,321,56,2]
车道：lane 0、lane 1
```

维度含义：

```text
B：batch
321：分类通道
56：row anchors
2：车道 0 和车道 1
```

分类通道定义：

```text
0～319：320 个横向网格位置
320：no-lane 类别
```

### 4.2 `cls_23`

```text
名称：cls_23
类型：float32
shape：[B,321,56,2]
车道：lane 2、lane 3
```

分类通道定义与 `cls_01` 完全相同。

### 4.3 `offset`

```text
名称：offset
类型：float32
shape：[B,1,56,4]
车道顺序：lane 0、lane 1、lane 2、lane 3
数值范围：[-0.5,0.5]
```

offset 用于对分类得到的整数或 soft-argmax 网格位置进行亚网格微调。

---

## 5. 分类权重的拆分方式

旧分类层参数：

```text
weight [71904,512]
bias   [71904]
```

旧输出向量最终被解释为：

```text
[321,56,4]
即 [class,row,lane]
```

因此不能把 71904 行简单地从中间切成上下两半。正确迁移方式是先恢复 lane 交错布局：

```python
old_weight = old_weight.reshape(321, 56, 4, 512)
weight_01 = old_weight[:, :, 0:2, :].reshape(35952, 512)
weight_23 = old_weight[:, :, 2:4, :].reshape(35952, 512)

old_bias = old_bias.reshape(321, 56, 4)
bias_01 = old_bias[:, :, 0:2].reshape(35952)
bias_23 = old_bias[:, :, 2:4].reshape(35952)
```

代码在加载旧 `best.pt` 时自动完成迁移，并在删除旧 `cls_fc2` 前重新拼回原权重进行逐元素一致性检查。

该迁移不会改变分类 logits，也不要求为了结构拆分重新训练。后续继续训练时，两个新分类头会分别获得梯度和更新参数。

---

## 6. 训练侧接口

训练、损失和验证代码继续使用原来的完整输出协议：

```text
cls    [B,321,56,4]
offset [B,  1,56,4]
```

`LaneRobotV2.forward()` 在非 ONNX 分头导出模式下，会在 PyTorch 内部沿 lane 维拼接两个分类头：

```python
cls = torch.cat((cls_01, cls_23), dim=3)
```

因此以下逻辑不需要修改：

- 分类交叉熵和软标签损失；
- no-lane 存在性损失；
- Top-K local soft-argmax；
- offset SmoothL1 损失；
- 平滑和曲率损失；
- validator 和训练可视化。

---

## 7. ONNX 推理后处理变化

### 7.1 旧 ONNX

旧模型只返回一个张量：

```python
lane_output = session.run(["lane_output"], inputs)[0]

cls_logits = lane_output[:, 0:321, :, :]
offset = lane_output[:, 321:322, :, :]
```

### 7.2 新 ONNX

新模型返回三个张量：

```python
cls_01, cls_23, offset = session.run(
    ["cls_01", "cls_23", "offset"],
    {"images": input_tensor},
)
```

必须沿 lane 维，也就是 NCHW 张量的 `axis=3` 拼接：

```python
cls_logits = np.concatenate((cls_01, cls_23), axis=3)
```

结果：

```text
cls_01     [B,321,56,2]
cls_23     [B,321,56,2]
                         沿 axis=3 拼接
cls_logits [B,321,56,4]
```

不能沿分类通道 `axis=1` 拼接，否则会错误地得到 `[B,642,56,2]`，完全破坏类别和车道语义。

完成拼接后，其余后处理保持不变：

```text
cls_01 + cls_23
→ 沿 lane 维拼接为 cls_logits [B,321,56,4]
→ Softmax(axis=1)
→ 读取分类索引 320 的 no-lane 概率
→ 对分类索引 0～319 执行 Top-K soft-argmax
→ 加 offset[:,0,:,:]
→ no-lane 点设置为 -1
→ 可选多项式平滑
→ 恢复到原图坐标并绘制或保存标签
```

置信度定义也没有变化：

```text
点存在置信度 = 1 - P(no-lane)
位置置信度   = max(P(grid 0～319))
```

---

## 8. 预处理是否变化

预处理没有变化。

默认流程仍然是：

```text
读取图片并修正 EXIF 方向
→ RGB
→ 直接 resize 到 320×320，INTER_LINEAR
→ float32
→ 除以 255
→ HWC 转 CHW
→ 增加 batch
→ [1,3,320,320]
```

只有显式使用 `--letterbox` 时才启用保持宽高比的顶部补黑模式，并且必须与训练时的设置一致。

---

## 9. 导出方法

在 `lane_robot` Conda 环境中执行：

```bash
conda run -n lane_robot python export_onnx_newhead.py --verify-runtime
```

默认输入 checkpoint：

```text
runs/lane/lane_n_baseline-2/weights/best.pt
```

默认生成：

```text
runs/lane/lane_n_baseline-2/weights/best_newhead.onnx
```

导出脚本会检查：

1. checkpoint 必须是四线模型；
2. `x_grids` 默认必须为 320；
3. 两个分类 Linear 的输出必须分别为 35952；
4. 每个分类 Gemm 输出不能超过 65536；
5. opset-11 兼容池化改写前后数值一致；
6. ONNX checker 通过；
7. 图中存在且只存在一个 `cls_fc2_01/Gemm` 和一个 `cls_fc2_23/Gemm`；
8. 图中不存在旧 `cls_fc2/Gemm`；
9. 使用 `--verify-runtime` 时，三个 ONNX Runtime 输出分别与 PyTorch 比较。

---

## 10. 推理方法

```bash
conda run -n lane_robot python infer_onnx_newhead.py \
  --device cpu \
  --source test \
  --overwrite
```

`infer_onnx_newhead.py` 会强制检查输出布局。模型必须包含：

```text
cls_01 [B,321,56,2]
cls_23 [B,321,56,2]
offset [B,  1,56,4]
```

如果传入旧的单输出 ONNX，脚本会直接报错，避免把旧接口误认为新接口。

---

## 11. 当前验证结果

已经在 `lane_robot` Conda 环境完成以下验证：

```text
旧 checkpoint 权重迁移：PASS，逐元素完全一致
新分类头输出维度：35952 + 35952
RDK X5 单 Gemm 限制：每个分类头均小于 65536
ONNX checker：PASS
旧大 cls_fc2/Gemm：不存在
新 cls_fc2_01/Gemm：存在且唯一
新 cls_fc2_23/Gemm：存在且唯一
ONNX Runtime 三输出验证：PASS
真实图片推理、绘制和 TXT 保存：PASS
```

PyTorch 与新 ONNX Runtime 的最大绝对误差：

```text
cls_01：1.71661376953125e-05
cls_23：1.9073486328125e-05
offset：5.066394805908203e-07
```

使用同一张真实图片比较旧 `best.onnx` 和新 `best_newhead.onnx`：

```text
拼接后的分类 logits 最大绝对误差：0.0
offset 最大绝对误差：0.0
```

这说明当前 FP32 ONNX 的变化只在于分类输出被拆成两个接口，预测数值和语义没有变化。

---

## 12. RDK X5 转换时需要继续确认的内容

当前开发环境没有安装 `hb_mapper` 和 `hb_perf`，因此 ONNX 结构已经满足两个分类 Gemm 的尺寸条件，但仍需要在 OpenExplorer 环境重新完成转换并检查最终节点分配。

重点确认：

```text
/model/model.16/cls_fc2_01/Gemm -> BPU INT8
/model/model.16/cls_fc2_23/Gemm -> BPU INT8
/model/model.16/offset_fc/Gemm  -> BPU INT8
```

同时检查：

- 三个输出 tensor 的名称、shape 和排列顺序；
- 分类输出的量化余弦相似度；
- 两个分类 Gemm 是否仍出现 CPU fallback；
- 尾部 `Reshape`、`Tanh`、`Mul` 的执行位置；
- BPU 与 CPU 之间是否仍存在不必要的数据搬运；
- 板端后处理是否沿 lane 维正确拼接；
- `hb_perf` 延迟和真实板端帧率。

只有工具链节点表明确显示两个分类 Gemm 均进入 BPU，才能确认问题四在 RDK X5 Runtime BIN 中也已完全解决。
