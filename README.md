# YOLO26 Lane Robot V3B — zbn 分支

基于 Ultralytics YOLO26 与 Row-Anchor 思路改造的固定语义四线检测实验分支。

> 更新日期：2026-09-19  
> 当前分支：zbn  
> 核心结构：LaneRobotV3B + Learnable Row Sampling + Causal Row Conv  
> 标准输入：256x448  
> 关键实验：lane1 使用 lane0 预测几何作为条件，但 lane1 仍预测绝对坐标

本分支是在 V2 四槽位系统上继续修改 Head 的结构实验。它不是 quant_correct 的简单延伸：quant_correct 重点解决 V2 的 RDK X5 BPU 部署；zbn 重点解决远端行、弱特征和 lane1 语义依赖问题。

---

## 1. 四个固定语义槽位

| lane_id | 名称 | 语义 |
|---:|---|---|
| 0 | lane_follow | 跟随线 |
| 1 | lead_lane | 引导线 |
| 2 | channel_left | 黄色通道左边界 |
| 3 | channel_right | 黄色通道右边界 |

标签格式仍为：

~~~text
lane_id x1 y1 x2 y2 ... x56 y56
~~~

其中 x=-1 表示该 Row Anchor 没有有效点。

当前仍是固定四槽位模型，不是任意数量的实例曲线检测。

---

## 2. 最重要的代码事实：lane1 不是“直接回归 lane1-lane0 的 Δ”

项目讨论中容易把 zbn 记成：

~~~text
lane1 = lane0 + delta
直接监督 delta = lane1 - lane0
~~~

**当前代码并不是这个实现。**

LaneRobotV3B 当前做法是：

1. lane0 使用自己的分类器输出绝对网格 logits；
2. 对 lane0 logits 做 Top-K soft-argmax，得到每个 Row Anchor 的 lane0 预测 x；
3. 将这个 lane0 x 作为一个额外条件通道拼到 lane1 的特征上；
4. lane1 分类器根据“共享特征 + lane0 预测位置”输出 321 类 logits；
5. lane1 的最终输出仍然是 **绝对 x 网格坐标**；
6. Loss 仍然使用 lane1 的绝对 target_x，没有把 GT 转成 lane1-lane0。

代码逻辑可概括为：

~~~text
z [B,R,512]
  ├── cls_head[0](z)
  │      ↓
  │   lane0 logits
  │      ↓
  │   soft-argmax
  │      ↓
  │   lane0_x [B,R]
  │      ↓ detach
  │
  └── concat(z, lane0_x)
         ↓
      [B,R,513]
         ↓
      cls_head[1]
         ↓
      lane1 absolute logits [B,R,321]
~~~

因此更准确的名称是：

> **lane1 side-conditioned absolute prediction**

而不是：

> lane1 delta regression

另外，lane0_x 在送入 lane1 前调用了 detach，因此 lane1 的分类损失不会通过这个条件分支反向修改 lane0 的 soft-argmax 路径。

---

## 3. 为什么 lane1 要依赖 lane0

仓库中的数据分析记录表明，当前 lane1 / lead_lane 与 lane0 具有明显几何关系，旧数据中还曾混有不同语义的 lane1 标注。

基于这个现象，zbn 最终代码没有继续把 lane1 完全当成四个互不相关的视觉槽位之一，而是让 lane1 分类器显式看到 lane0 的预测位置。

需要注意：

仓库中的 docs/conclusions_2026-08-07.md 记录过“lane0 + 高斯先验 / delta”方向的设计讨论，但 **当前 head.py 最终实现没有使用固定高斯几何先验，也没有把输出改成 delta**。当前 README 以实际代码为准。

---

## 4. V3B Head 结构

标准输入：

~~~text
[B, 3, 256, 448]
~~~

Lane Head 接收到的融合特征目标尺寸：

~~~text
[B, 256, 16, 28]
~~~

Head 主链路：

~~~text
P4 + P5 fused feature
  ↓
Conv1x1: 256 -> 16
  ↓
Learnable row sampling
  ↓
56 Row Anchors
  ↓
每行：
16×28 展平 + 16 维 row mean
= 464 维
  ↓
CausalRowConv
  ↓
Linear 464 -> 512
  ↓
ReLU
  ↓
Learnable Row Embedding
  ↓
lane-specific heads
~~~

### 4.1 Learnable Row Sampling

row sampler 初始值由双线性采样矩阵生成，但实际参数是可学习的 grouped-conv 权重：

~~~text
row_conv_w
~~~

因此它不是永久固定的双线性采样。

Row Anchor 顺序为：

~~~text
r=0   -> y_end   -> 图像底部
r=55  -> y_start -> 图像上方
~~~

### 4.2 Causal Row Conv

当前模型配置：

~~~text
causal_kernel = 3
causal_layers = 2
~~~

目标是让更远的 Row Anchor 显式参考更近处的行特征，减轻远端弱纹理情况下逐行独立决策造成的抖动。

### 4.3 Lane-specific Heads

lane0 / lane2 / lane3 分类器输入：

~~~text
512 -> 321
~~~

lane1 分类器因为额外接收 lane0_x：

~~~text
513 -> 321
~~~

四个 offset head 都是：

~~~text
512 -> 1
~~~

最终：

~~~text
cls    [B, 321, 56, 4]
offset [B,   1, 56, 4]
~~~

offset 经过 tanh，并限制在：

~~~text
[-0.5, 0.5]
~~~

---

## 5. Loss 的当前实现

当前 Loss 仍然对所有四个槽位使用绝对 target_x。

### 5.1 分类

soft-label 的中心使用：

~~~text
round(target_x)
~~~

分类维仍是：

~~~text
0..319 : x grid
320    : no-lane
~~~

### 5.2 位置损失

解码位置：

~~~text
Top-K soft-argmax(cls) + offset
~~~

然后直接与绝对 target_x 比较。

### 5.3 Offset Plan-A

当前代码明确关闭独立 offset target：

~~~text
offset_loss = 0
~~~

offset head 只通过 lane_loc 的整体位置误差获得梯度。

所以 lane_offset 参数目前仍保留在配置接口中，但 Loss 不再单独计算传统的 offset SmoothL1 目标。

---

## 6. 一个需要特别注意的当前配置差异

ultralytics/cfg/default.yaml 中当前写的是：

~~~text
lane_end_weight      = 1.0
lane_end_weight_tail = 1.0
~~~

即远端额外加权关闭。

但 train_v3b.py 当前 CLI 默认值是：

~~~text
--lane-end-weight       1.0
--lane-end-weight-tail  6.0
~~~

并且 train_v3b.py 会把 CLI 值显式传给 Trainer。

因此：

> **直接运行 train_v3b.py 时，实际 tail 默认是 6.0，不是 default.yaml 中的 1.0。**

如果你想严格使用“完全不做远端额外加权”的 Plan-A / causal 基线，应显式运行：

~~~bash
python train_v3b.py \
  --lane-end-weight 1.0 \
  --lane-end-weight-tail 1.0 \
  --lane-end-no-lane-weight 1.0
~~~

这是当前代码的真实状态，后续实验记录必须写清楚实际命令，不能只引用 default.yaml。

---

## 7. 训练输入必须是 256x448

V3B 模型 YAML 目标输入为：

~~~text
height = 256
width  = 448
~~~

对应 Head feature map：

~~~text
16 x 28
~~~

LaneRobotTrainer / Validator 在该分支中已针对 lane 任务保留矩形 imgsz，避免通用 Ultralytics 逻辑把 [256,448] 强制变成正方形。

训练时应确认日志实际显示矩形尺寸，而不是只看命令行参数。

推荐入口：

~~~bash
python train_v3b.py \
  --img-height 256 \
  --img-width 448 \
  --name lane_v3b
~~~

---

## 8. train_v3b.py 当前默认配置

主要默认值：

~~~text
model        = yolo26s-lane-v3b.yaml
imgsz        = [256, 448]
epochs       = 500
patience     = 100
batch        = -1
workers      = 8
optimizer    = AdamW
lr0          = 3e-4
lrf          = 0.01
weight_decay = 0.01
seed         = 42
~~~

增强：

~~~text
hsv_h       = 0.002
hsv_s       = 0.05
hsv_v       = 0.05
degrees     = 2.0
translate   = 0.03
scale       = 0.05
fliplr      = 0.5

mosaic      = 0
mixup       = 0
cutmix      = 0
copy_paste  = 0
erasing     = 0
~~~

水平翻转仍依赖固定语义槽位交换逻辑，必须保证数据 YAML 中的 flip_lane_pairs 与任务语义一致。

---

## 9. lane1 旧权重处理

train_v3b.py 在加载预训练权重时注册 on_train_start callback。

当前代码会重新初始化：

~~~text
cls_heads[1]
offset_heads[1]
~~~

原因是 lane1 当前已经改成依赖 lane0 的 side-conditioned 结构，不应该直接继承旧 lane1 standalone classifier 的语义。

注意：

- backbone / neck / 其他兼容参数仍可加载；
- lane1 分类头和 offset 头会被 reset；
- 实际 transferred 数量应以启动日志为准。

---

## 10. 数据路径

zbn 支持环境变量：

~~~bash
export LANE_ROBOT_DATASETS=/absolute/path/to/datasets
~~~

train_v3b.py 还会默认：

~~~text
LANE_ROBOT_DATASETS = <repo>/datasets
~~~

这样在 tmux / nohup 等非交互 shell 中也不依赖用户 bashrc。

数据结构仍为：

~~~text
datasets/
├── images/
│   ├── train/
│   └── valid/
└── labels_corrected/
    ├── train/
    └── valid/
~~~

---

## 11. Validator

zbn 已在 Lane Validator 中加入逐槽位指标。

总体指标包括：

~~~text
metrics/lane_mae
metrics/lane_mae_px
metrics/lane_acc_valid_tol1
metrics/lane_acc_valid_tol3
metrics/lane_acc_valid_tol5
metrics/lane_exist_acc
~~~

并增加：

~~~text
metrics/lane0_*
metrics/lane1_*
metrics/lane2_*
metrics/lane3_*
~~~

其中：

~~~text
lane0 = lane_follow
lane1 = lead_lane
lane2 = channel_left
lane3 = channel_right
~~~

这对 zbn 尤其重要，因为 lane1 使用了独立结构，不能只看四槽位平均值。

---

## 12. ONNX 导出

V3B 专用导出脚本：

~~~text
export_onnx_v3b.py
~~~

标准输入：

~~~text
images [1, 3, 256, 448]
~~~

标准输出：

~~~text
lane_output [1, 322, 56, 4]
~~~

其中：

~~~text
0:321   -> cls logits
321:322 -> offset
~~~

opset：

~~~text
11
~~~

导出脚本会检查实际 Head 输入 feature map 是否为 16x28；如果输入尺寸会触发 adaptive-pool fallback，则拒绝标准部署导出。

示例：

~~~bash
python export_onnx_v3b.py \
  --weights runs/lane/lane_v3b/weights/best.pt \
  --imgsz 256 448 \
  --verify-runtime
~~~

---

## 13. ONNX Runtime 推理

V3B 专用入口：

~~~text
infer_onnx_v3b.py
~~~

它继承当前通用 Lane 后处理，并按 V3B 的 320 grids、56 anchors 和 256x448 输入检查模型。

训练、导出和推理必须统一：

~~~text
256x448
RGB
Direct Resize
float32 / 255
~~~

除非重新训练并完整验证，否则不要只在推理端切换 LetterBox。

---

## 14. 数据清理与 lane1 语义

当前分支还包含针对 lane1 的数据检查和清理工具，例如：

~~~text
tools/check_ds0806_labels.py
tools/check_label_image_match.py
tools/check_label_semantics.py
tools/clean_lane1.py
tools/gen_lane1_audit.py
tools/verify_lane1_semantics.py
~~~

这些工具来自 lane1 语义清理阶段。

README 不把某次远程数据目录中的修改结果当成仓库数据集本身；正式训练时仍应对你实际使用的数据重新做：

- lane1 是否必须伴随 lane0；
- lane1 直线性 / 跨度；
- train / valid 泄漏；
- 标签图片匹配；
- 每槽位数量；
- 每个 Row Anchor 有效点密度。

---

## 15. RDK X5 状态如何理解

zbn 的结构从设计上避免了 V2 单个 71904 输出的大 cls_fc2：

- 四个 lane 使用各自的分类器；
- 每个分类器共享 56 行；
- ONNX 输出仍合并为 lane_output [1,322,56,4]。

但 **仅从当前仓库代码本身，不能证明 zbn 的最终 Runtime BIN 节点分配和板端性能**。

因此本 README 只确认：

- V3B 训练代码存在；
- V3B ONNX Opset 11 导出入口存在；
- V3B ONNX Runtime 推理入口存在；
- 输出协议已固定。

如果要宣称 zbn 已完成 RDK X5 量化 / 全 BPU / 实际 FPS，需要对应的量化日志、Runtime BIN 信息或板端实测记录。

已经明确完成 BPU 部署验证的是 quant_correct 的 V2 双分类头方案。

---

## 16. 当前关键文件

~~~text
train_v3b.py
    V3B 训练入口

export_onnx_v3b.py
    V3B ONNX Opset 11 导出

infer_onnx_v3b.py
    V3B ONNX Runtime 推理

ultralytics/cfg/models/26/yolo26s-lane-v3b.yaml
    V3B 模型结构

ultralytics/nn/modules/head.py
    LaneRobotV3B
    Learnable Row Sampling
    CausalRowConv
    lane1 side conditioning

ultralytics/utils/loss.py
    absolute target_x
    Top-K soft-argmax + offset
    Plan-A offset supervision

ultralytics/models/yolo/lane/train.py
    矩形 imgsz 与 lane 训练配置

ultralytics/models/yolo/lane/val.py
    逐槽位验证指标

tools/
    lane1 数据检查与清理工具
~~~

---

## 17. 当前已知风险

1. lane1 依赖 lane0 的预测位置，因此 lane0 大幅错误时会给 lane1 提供错误条件。
2. lane0_x 被 detach，lane1 不能通过条件分支反向纠正 lane0。
3. lane1 仍是绝对 321 类分类，不是显式 delta 回归；后续比较实验时不要混淆两种方案。
4. train_v3b.py 的 lane_end_weight_tail 默认值与 default.yaml 不一致。
5. fliplr=0.5 必须与固定语义交换规则完全一致。
6. 256x448 与旧 V2 320x320 的校准数据、ONNX 和板端坐标恢复不能直接混用。
7. 数据清理工具中得到的历史统计不自动代表当前训练集。
8. RDK X5 最终节点分配必须以实际量化日志和板端运行结果为准。

---

## 18. 分支关系

| 分支 | 主要定位 |
|---|---|
| main | V2 原始四槽位基线，大分类 Head |
| quant_correct | V2 双分类头，RDK X5 量化和 BPU 部署已验证 |
| zbn | V3B：learnable row + causal row + lane1 side-conditioned on lane0 |
| lmm | 160-grid，四个完全独立单线 Head |

---

## 19. 后续建议

zbn 下一轮实验至少同时记录：

~~~text
commit SHA
dataset version
train/valid split
imgsz
实际 CLI
lane_end_weight / tail
fliplr
best epoch
lane0/1/2/3 指标
PyTorch vs ONNX 数值误差
~~~

对于 lane1，建议单独比较三种概念，不要混称：

~~~text
A. 完全独立 absolute classifier
B. 当前实现：conditioned absolute classifier
C. 真正的 delta regression: lane1 = lane0 + delta
~~~

当前代码属于 B。
