# YOLO26 Lane Robot V2 — quant_correct / RDK X5 BPU 部署分支

基于 Ultralytics YOLO26 与 UFLD / Row-Anchor 思路改造的固定语义四线检测工程。

> 更新日期：2026-09-19  
> 当前分支：quant_correct  
> 分支定位：解决 V2 大分类 Gemm 无法进入 RDK X5 BPU 的部署问题  
> 当前结论：双分类头量化链路已经跑通，并已验证可在 RDK X5 BPU 部署

本分支不是新的任务定义，而是 main 的 V2 四槽位模型的 **部署结构修正版**。训练侧仍保持原来的四槽位输出和损失接口；核心变化是把原来一个过大的分类 Linear 真正拆成两个较小 Linear，使分类计算满足 RDK X5 BPU 的单 Gemm 输出维度约束。

---

## 1. 任务定义

模型识别四个固定语义槽位：

| lane_id | 名称 | 语义 |
|---:|---|---|
| 0 | lane_follow | 跟随线 |
| 1 | lead_lane | 引导线 |
| 2 | channel_left | 黄色通道左边界 |
| 3 | channel_right | 黄色通道右边界 |

模型仍属于：

> 固定四种语义、每种语义在一张图中最多一条曲线的多线检测模型。

标签格式仍为：

~~~text
lane_id x1 y1 x2 y2 ... x56 y56
~~~

其中：

- lane_id 只能是 0～3。
- x 为归一化横坐标。
- x=-1 表示该 Row Anchor 上没有有效点。
- 一条线整张图不存在时，可以不写该 lane_id。
- Row Anchor 数量固定为 56。

---

## 2. 为什么要有 quant_correct

main 中的 V2 分类头一次性输出：

~~~text
321 × 56 × 4 = 71904
~~~

对应旧节点：

~~~text
/model/model.16/cls_fc2/Gemm
~~~

RDK X5 工具链对该类 Gemm 的相关输出维度限制为 65536，因此：

~~~text
71904 > 65536
~~~

旧结构会导致该分类 Gemm 回退到 CPU。

quant_correct 将四个槽位拆成两组：

~~~text
lane 0 + lane 1
lane 2 + lane 3
~~~

得到两个分类 Linear：

~~~text
cls_fc2_01: 512 -> 321 × 56 × 2 = 35952
cls_fc2_23: 512 -> 321 × 56 × 2 = 35952
~~~

两个输出都满足：

~~~text
35952 < 65536
~~~

这不是在旧大 Gemm 后面加 Split，而是 **真正把一个大 Linear/Gemm 改成两个较小的 Linear/Gemm**。

---

## 3. 当前 Head 结构

整体结构：

~~~text
输入
  ↓
YOLO26 Backbone / Neck
  ↓
P4 + P5 融合
  ↓
LaneRobotV2
  ↓
Conv1x1
  ↓
Pool / Flatten
  ↓
cls_fc1 -> ReLU
  ↓
共享 512 维特征
  ├── cls_fc2_01 -> lane 0/1
  ├── cls_fc2_23 -> lane 2/3
  └── offset_fc  -> lane 0/1/2/3
~~~

当前参数：

~~~text
x_grids     = 320
row_anchors = 56
num_lanes   = 4
~~~

训练和普通 PyTorch 推理仍保持原协议：

~~~text
cls    [B, 321, 56, 4]
offset [B,   1, 56, 4]
~~~

也就是说，训练、Loss、Validator 和现有四槽位解码逻辑不需要因为拆头而改成两套任务。

---

## 4. 旧权重如何迁移

旧分类层权重是：

~~~text
weight [71904, 512]
bias   [71904]
~~~

旧输出实际按：

~~~text
[class, row, lane]
= [321, 56, 4]
~~~

排列，因此不能简单把 71904 从中间切成两半。

代码中的迁移逻辑先恢复为：

~~~text
[321, 56, 4, 512]
~~~

再沿 lane 维切：

~~~text
lane 0:2 -> cls_fc2_01
lane 2:4 -> cls_fc2_23
~~~

bias 同理。

这样可以保持旧 checkpoint 的 FP32 分类输出语义不变。分支中的导出逻辑还会检查旧大分类节点是否已经消失，并检查两个新 Gemm 是否存在。

---

## 5. ONNX 输出协议

本分支用于 RDK X5 的新 Head 导出脚本是：

~~~text
export_onnx_newhead.py
~~~

默认静态输入：

~~~text
images [1, 3, 320, 320]
opset 11
~~~

新 ONNX 暴露三个输出：

~~~text
cls_01 [B, 321, 56, 2]
cls_23 [B, 321, 56, 2]
offset [B,   1, 56, 4]
~~~

板端或 ONNX Runtime 后处理必须沿 lane 维拼接：

~~~python
cls_logits = np.concatenate((cls_01, cls_23), axis=3)
~~~

恢复成：

~~~text
cls_logits [B, 321, 56, 4]
~~~

然后继续执行：

~~~text
Softmax(class 维)
-> no-lane 判断
-> Top-K soft-argmax
-> 加 offset
-> 坐标恢复
~~~

不能沿 class 维拼接。

---

## 6. RDK X5 量化状态

### 已验证结论

本分支的双分类头方案已经完成 RDK X5 量化，并验证 **BPU 可以部署该结构**。

因此项目当前对 V2 的部署结论应区分为：

~~~text
main
└── 单个 cls_fc2 = 71904
    └── 大分类 Gemm 回退 CPU
    └── BPU + CPU 混合基线

quant_correct
├── cls_fc2_01 = 35952
├── cls_fc2_23 = 35952
└── 量化已跑通
    └── 已验证 BPU 可部署
~~~

这证明 main 中的 CPU fallback 是旧 Head 结构造成的，不是 Lane Robot 四线任务本身不能部署到 BPU。

### 需要准确理解“BPU 可部署”

“BPU 可部署”表示拆分 Head 后，量化和板端 BPU 部署链路已经被实际验证。

它不自动等价于：

- ONNX 图中所有节点 100% 都在 BPU；
- 整个 runtime 图完全没有 CPU 后处理或格式转换；
- 已经完成最终机器人整机 FPS / 功耗 / 控制闭环评测。

最终性能仍应以实际 Runtime BIN 的节点分配、hb_perf 和板端端到端实测为准。

---

## 7. 关键文件

~~~text
ultralytics/nn/modules/head.py
    LaneRobotV2 双分类头实现
    旧 cls_fc2 权重迁移逻辑

export_onnx_newhead.py
    拆头 ONNX Opset 11 导出
    输出 cls_01 / cls_23 / offset
    检查旧大 Gemm 不再存在

infer_onnx_newhead.py
    新三输出 ONNX 图片推理入口

infer_onnx_newhead_camera.py
    摄像头实时推理与可视化

infer_onnx_newhead_points.py
    点形式输出 / 可视化入口

infer_external_dataset_newhead.py
    外部数据集新 Head 推理入口

Lane_Robot_RDK_X5_newhead_ONNX_structure_2026-08-03.md
    双分类头结构和 ONNX 迁移说明

Lane_Robot_RDK_X5_quantization_issues_and_solutions_2026-08-03.md
    旧 V2 大 Gemm 问题的背景记录
~~~

注意：旧量化说明文档记录的是“拆头前”的问题背景；当前 quant_correct 的最终状态应以本 README 和当前代码为准。

---

## 8. ONNX 导出

示例：

~~~bash
conda activate lane_robot

python export_onnx_newhead.py \
  --weights runs/lane/lane_n_baseline-3/weights/best.pt \
  --output runs/lane/lane_n_baseline-3/weights/best_newhead.onnx \
  --imgsz 320 320 \
  --verify-runtime
~~~

导出脚本会检查：

- num_lanes 必须为 4；
- x_grids 默认应为 320；
- 两个分类 Linear 的输出各为 35952；
- 单分类 Gemm 输出不超过 65536；
- ONNX 图中不存在旧 cls_fc2/Gemm；
- ONNX 图中存在 cls_fc2_01/Gemm 和 cls_fc2_23/Gemm；
- 可选比较 PyTorch 与 ONNX Runtime 输出。

---

## 9. ONNX Runtime 推理

图片 / 目录推理使用新 Head 入口，不要把三输出模型当成旧单输出模型：

~~~bash
python infer_onnx_newhead.py \
  --model runs/lane/lane_n_baseline-3/weights/best_newhead.onnx \
  --source test \
  --device cpu \
  --overwrite
~~~

摄像头：

~~~bash
python infer_onnx_newhead_camera.py \
  --model runs/lane/lane_n_baseline-3/weights/best_newhead.onnx
~~~

新 Head 推理脚本会检查输出布局，避免把旧 lane_output 单张量模型误当成三输出模型。

---

## 10. 预处理约束

当前 V2 / quant_correct 基线按以下口径工作：

~~~text
RGB
-> Direct Resize 320x320
-> float32
-> /255
-> NCHW
~~~

量化、ONNX Runtime 和板端预处理必须和训练 / 导出保持一致。

如果权重不是按照 LetterBox 训练，就不要在部署端单独改成 LetterBox。

---

## 11. 与其他分支的关系

| 分支 | 核心用途 |
|---|---|
| main | V2 原始四槽位基线；旧大分类 Head；用于训练/推理/部署对照 |
| quant_correct | V2 双分类头；解决 RDK X5 大 Gemm 问题；BPU 部署已验证 |
| zbn | V3B 结构实验；Causal Row + lane1 依赖 lane0 的条件建模 |
| lmm | 160-grid、四个完全独立的单线任务分支 |

quant_correct 的目标不是替代 zbn 的模型结构实验，而是把已经成熟的 V2 四线方案做成可实际量化部署的版本。

---

## 12. 当前已知风险与后续工作

1. 训练和量化输入尺寸必须保持一致，当前部署基线为 320x320。
2. 三输出 ONNX 的拼接轴必须是 lane 维 axis=3。
3. 新旧 checkpoint 迁移依赖正确的 [class, row, lane] 排列，不能自行按扁平向量一刀切。
4. 后处理仍需正确处理 x=-1 / no-lane，不能跨大段无效 Anchor 强行连线。
5. BPU 可部署已经验证，但最终整机性能仍应记录 Runtime BIN、hb_perf、CPU/BPU 占用、真实 FPS 和控制延迟。
6. 如果后续修改 x_grids、row_anchors 或 num_lanes，需要重新检查 Gemm 输出尺寸和板端输出协议。

---

## 13. 上游与许可证

本项目基于 Ultralytics 源码继续修改。许可证见仓库 LICENSE。

提交数据集、权重、量化模型或第三方代码前，请分别确认对应授权和许可证要求。
