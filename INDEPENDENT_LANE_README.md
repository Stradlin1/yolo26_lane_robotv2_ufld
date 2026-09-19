# LMM 分支 — 160-grid 四任务独立 LaneRobotV2

> 更新日期：2026-09-19  
> 当前分支：lmm  
> 核心思路：共享 YOLO Backbone / Fusion，但四个 lane task 使用四套完整、互不共享参数的 LaneRobotV2 prediction branch  
> 当前默认训练输入：640x640  
> 当前横向网格：160

lmm 与 main / quant_correct / zbn 的四槽位联合 Head 思路不同。

它保留单任务 LaneRobotV2 的完整预测分支，并复制四份：

~~~text
shared backbone + fusion
  ├── task branch 0
  ├── task branch 1
  ├── task branch 2
  └── task branch 3
~~~

四个任务只共享 Backbone / Neck / Fusion；进入 prediction head 后，每个任务都有自己的 Conv1x1、FC、分类层和 offset 层。

---

## 1. 当前模型协议

当前配置：

~~~text
x_grids     = 160
classes/row = 161
row_anchors = 56
num_lanes   = 4
~~~

161 个分类状态含义：

~~~text
0..159 : 横向网格
160    : no-lane
~~~

训练 / PyTorch 输出：

~~~text
cls    [B, 161, 56, 4]
offset [B,   1, 56, 4]
~~~

LaneRobotV2Independent 设置 export=True 时，内部也支持合并为：

~~~text
[B, 162, 56, 4]
~~~

但当前根目录 export_onnx.py 实际导出的是 **两个独立 ONNX 输出**：

~~~text
cls_logits [B, 161, 56, 4]
offset     [B,   1, 56, 4]
~~~

因此使用 ONNX 时应以具体导出脚本的输出契约为准，不要只看 Head 的 export=True 行为。

---

## 2. Independent Head 结构

单个任务分支是：

~~~text
input feature
  ↓
Conv1x1
  ↓
AdaptiveAvgPool2d
  ↓
Flatten
  ↓
cls_fc1
  ↓
ReLU
  ├── cls_fc2
  └── offset_fc
~~~

SingleLaneRobotV2Branch 的默认参数：

~~~text
x_grids        = 160
row_anchors    = 56
reduce_channels= 8
hidden_dim     = 512
feat_h         = 8
feat_w         = 10
~~~

LaneRobotV2Independent 创建四个完全独立的 SingleLaneRobotV2Branch。

所以：

- 四个 task 的分类器不共享；
- 四个 task 的 offset head 不共享；
- 四个 task 的 Conv1x1 和 cls_fc1 也不共享；
- Softmax 只沿 161 个 x/no-lane 类别维执行；
- 四个 task 之间没有类别竞争。

---

## 3. 与其他分支的区别

| 分支 | Head 关系 |
|---|---|
| main | 四槽位共享一个 V2 大分类 Head |
| quant_correct | V2 分类 Head 拆成 0/1 与 2/3 两个较小 Head，面向 RDK X5 BPU |
| zbn | V3B 行级 Head；lane1 额外依赖 lane0 预测位置 |
| lmm | 四个完整单任务 LaneRobotV2 Head 完全独立 |

lmm 的设计目标不是减少 Head 参数，而是最大限度保留“每个单任务模型各自独立学习”的行为。

---

## 4. 标签格式

每张图可以包含 0～4 行：

~~~text
lane_id x1 y1 x2 y2 ... x56 y56
~~~

规则：

- lane_id 为 0～3；
- 每个 lane_id 最多一行；
- x 为归一化横坐标；
- x=-1 表示该固定 Row Anchor 没有有效点；
- 整条任务不存在时，该 lane_id 可以完全不出现。

当前 lane-robot-4tasks.yaml 中的任务名仍是通用名称：

~~~text
lane_task_0
lane_task_1
lane_task_2
lane_task_3
~~~

如果要与项目主线的四个固定语义完全对应，应在数据配置和实验记录中明确 task 0～3 实际代表什么，不能只靠顺序猜测。

---

## 5. 当前默认训练配置

ultralytics/cfg/default.yaml 当前实际值：

~~~text
task       = lane
model      = yolo26s-lane-independent.yaml
data       = ultralytics/cfg/datasets/lane-robot-4tasks.yaml

epochs     = 120
patience   = 25
batch      = 16
imgsz      = 640

optimizer  = AdamW
lr0        = 3e-4
lrf        = 0.02
weight_decay = 0.015
~~~

Lane 参数：

~~~text
lane_x_grids        = 160
lane_row_anchors    = 56
lane_num_lanes      = 4
lane_task_weights   = [1,1,1,1]

lane_ce             = 1.0
lane_loc            = 2.0
lane_exist          = 1.0
lane_smooth         = 0.03
lane_offset         = 3.0
lane_curv           = 0.02

lane_soft_label     = True
lane_soft_sigma     = 1.2
lane_softargmax_topk= 5
~~~

train.py 只是读取 default.yaml 并调用：

~~~python
model.train(cfg=default_yaml)
~~~

因此 **真正生效的训练配置以 default.yaml 为准**。

train.py 注释里仍写着 yolo26m-lane-independent.yaml，但当前 default.yaml 已经是 yolo26s-lane-independent.yaml；注释已经落后于实际配置。

---

## 6. 数据 YAML

当前：

~~~text
ultralytics/cfg/datasets/lane-robot-4tasks.yaml
~~~

主要配置：

~~~text
x_grids     = 160
row_anchors = 56
num_lanes   = 4

y_start = 0.333
y_end   = 1.0
~~~

也就是训练标签 / Dataset 的纵向范围按大约：

~~~text
1.0 -> 0.333
~~~

覆盖图像下方约 2/3 区域。

---

## 7. 当前存在一个重要的训练 / 推理 Row Anchor 不一致

这是 lmm 分支当前最需要注意的代码问题。

训练数据 YAML 使用：

~~~text
y_start = 0.333
y_end   = 1.0
~~~

但当前：

~~~text
infer.py
infer_onnx.py
~~~

都硬编码：

~~~python
Y_START = 0.67
Y_END = 1.0
~~~

也就是说：

~~~text
训练 / 标签：
1.0 -> 0.333

ONNX 推理绘制：
1.0 -> 0.67
~~~

这两个纵向范围并不一致。

影响：

- 分类 logits 本身仍然是 56 个 row；
- 但推理脚本给这些 row 分配的 y 坐标会不同；
- 映射回原图后的点位置会被压缩到更靠下的区域；
- 可视化和后续控制坐标可能与训练标签语义不一致。

因此在修正这处代码前，不应把 infer.py / infer_onnx.py 的 y 坐标输出直接当成与 lane-robot-4tasks.yaml 完全一致的结果。

本次只更新 README，没有修改推理代码。

---

## 8. 训练

直接运行当前 default.yaml：

~~~bash
python train.py
~~~

启动后建议确认：

~~~text
model = yolo26s-lane-independent.yaml
imgsz = 640
x_grids = 160
row_anchors = 56
num_lanes = 4
~~~

并确认四个 task 的数据量，而不是只看总 loss。

---

## 9. 从四个单任务 checkpoint 初始化

脚本：

~~~text
init_independent_lane_from_single_models.py
~~~

用法：

~~~bash
python init_independent_lane_from_single_models.py \
  --model ultralytics/cfg/models/26/yolo26n-lane-independent.yaml \
  --base task0_best.pt \
  --task-weights task0_best.pt task1_best.pt task2_best.pt task3_best.pt \
  --output independent_4task_init.pt
~~~

逻辑：

- base checkpoint 提供共享 Backbone / Neck / Fusion 的兼容参数；
- 每个 task checkpoint 提供自己的：
  - conv_1x1
  - cls_fc1
  - cls_fc2
  - offset_fc
- 每个单任务 Head 被复制到对应 independent branch。

脚本会检查 missing / shape mismatch；迁移验证失败时不会写输出 checkpoint。

---

## 10. ONNX 导出

当前 export_onnx.py 使用固定配置：

~~~text
IMG_SIZE = 640
OPSET    = 18
~~~

输出：

~~~text
cls_logits [1,161,56,4]
offset     [1,1,56,4]
~~~

导出脚本当前使用硬编码权重和输出路径，因此换实验时需要先检查脚本顶部：

~~~text
WEIGHTS
OUTPUT
~~~

再执行：

~~~bash
python export_onnx.py
~~~

与 main / quant_correct 的 Opset 11 不同，lmm 当前根目录导出脚本是 Opset 18。

---

## 11. ONNX Runtime 推理

静态图片入口：

~~~text
infer_onnx.py
~~~

摄像头入口：

~~~text
infer.py
~~~

当前共同配置：

~~~text
IMG_SIZE    = 640
X_GRIDS     = 160
ROW_ANCHORS = 56
NUM_LANES   = 4

CONF_THR    = 0.15
NO_LANE_THR = 0.5
TOPK        = 5
OFFSET_CLIP = 0.5
~~~

默认预处理是：

~~~text
BGR image
-> resize 640x640
-> RGB
-> float32 / 255
-> NCHW
~~~

两者都只画有效点，不强制把所有点连成折线，这一点对保留断点更安全。

但再次强调：当前 Y_START=0.67 与训练 YAML 的 0.333 不一致。

---

## 12. 当前 README 旧版本中已经失效的内容

旧 INDEPENDENT_LANE_README.md 曾写：

~~~text
model: yolo26m-lane-independent.yaml
~~~

但当前 default.yaml 实际是：

~~~text
yolo26s-lane-independent.yaml
~~~

旧 README 还引用：

~~~text
verify_independent_lane.py
~~~

当前 lmm 分支树中没有这个脚本，因此不能再把它写成现有验证入口。

---

## 13. 预标注脚本的当前状态

lmm 根目录存在：

~~~text
prelabel_onnx_v3b.py
~~~

但该脚本当前：

~~~python
import infer_onnx_v3b as v3b
~~~

而 lmm 分支树中没有 infer_onnx_v3b.py。

因此按当前仓库独立 checkout：

> prelabel_onnx_v3b.py 不是自包含可运行入口。

除非另外提供它依赖的 V3B 推理模块，否则不应把它当成 lmm independent 模型的正式预标注工具。

---

## 14. 手工标签修正工具

最新提交加入 / 更新了：

~~~text
manual_fix_56anchors_v11_class1_789_update.py
scripts/manual_fix_56anchors_v11_class1_789_update.py
~~~

该工具面向固定 56 anchors 的人工修正，并支持：

- x=-1 缺失行；
- 多段可见线；
- 遮挡段断开；
- labels_corrected 输出；
- 原子保存。

但工具内部支持的 class id 范围包含 0～4，而 independent 四任务模型仍只有：

~~~text
num_lanes = 4
=> task 0..3
~~~

因此用于模型训练的数据仍要保证最终 lane_id 只落在 0～3；class 4 若用于人工标注辅助，不能未经处理直接送入四任务 Dataset。

---

## 15. 当前关键文件

~~~text
ultralytics/nn/modules/head.py
    SingleLaneRobotV2Branch
    LaneRobotV2Independent

ultralytics/cfg/models/26/
    yolo26*-lane-independent.yaml

ultralytics/cfg/datasets/lane-robot-4tasks.yaml
    160-grid / 56-anchor / 4-task 数据配置

ultralytics/cfg/default.yaml
    当前实际训练超参数

train.py
    使用 default.yaml 启动训练

init_independent_lane_from_single_models.py
    四个单任务 checkpoint -> independent model

export_onnx.py
    640x640 / Opset 18 / 两输出导出

infer_onnx.py
    图片 ONNX 推理

infer.py
    摄像头 ONNX 推理

count_lane_classes.py
    标签 task 数量统计

manual_fix_56anchors_v11_class1_789_update.py
    手工标签修正
~~~

---

## 16. 当前已知风险

1. **最高优先级：训练 YAML y_start=0.333，但两个推理脚本 Y_START=0.67。**
2. train.py 注释中的默认模型名称落后于 default.yaml。
3. 根目录路径大量硬编码到具体机器，换机器前要检查 WEIGHTS / OUTPUT / ONNX_PATH / IMAGE_PATH。
4. export_onnx.py 当前 Opset 18，与 RDK X5 主线使用的 Opset 11 不是同一部署口径。
5. prelabel_onnx_v3b.py 依赖当前分支不存在的 infer_onnx_v3b.py。
6. 手工修正工具可处理 class 4，但四任务模型只接受 0～3。
7. lane-robot-4tasks.yaml 的 names 仍是 generic task 名，需要和真实任务语义建立明确映射。
8. 四个独立 Head 参数量更大，部署性能不能直接套用 quant_correct 的 BPU 结论。

---

## 17. 当前建议工作顺序

### P0：统一 Row Anchor y 范围

先决定 lmm 最终要使用：

~~~text
1.0 -> 0.333
~~~

还是：

~~~text
1.0 -> 0.67
~~~

然后让以下全部一致：

~~~text
训练标签
dataset YAML
PyTorch predictor
ONNX inference
可视化
控制坐标恢复
~~~

### P1：冻结 independent baseline

记录：

~~~text
commit SHA
default.yaml
dataset version
task 0..3 语义
train/valid split
best epoch
四个 task 各自指标
~~~

### P2：再比较三个结构方向

在同一数据划分下比较：

~~~text
lmm            : 四个完全独立 Head
main           : V2 联合 Head
zbn            : V3B + lane1 conditioned on lane0
~~~

RDK X5 部署性能对照应另外加入已经验证 BPU 的 quant_correct。
