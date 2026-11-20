# FHAPD XJ / JS 评估接口 v1

本文件记录数据适配阶段的指标接口。当前尚无训练好的 FHAPD 模型、正式测试结果或 checkpoint；合成掩码单元测试仅用于验证指标与聚合逻辑。

## 固定设置与数据身份

- 协议标识：`HBGNet-metrics-FHAPD-256-3px-spatial-v1`。
- 输入：原始 RGB uint8，256×256；除以 255 后 ImageNet 归一化。
- 决策：概率严格 `>0.5`；禁止将未经 sigmoid 的 logits 作为概率送入接口。布尔预测掩码可直接使用。
- TTA、小目标删除与其他形态学后处理均关闭。
- 每个模型/seed 使用同一区域已冻结的 train/val/test 清单；XJ 382/83/89，JS 592/117/142。
- `expected_counts`、尺寸和边界像素容差由配置读取；评估流须完整、ID 唯一、顺序与清单完全相同。
- 当前边界容差为 3 像素。PNG 无可信地理元数据，GSD 未确认，不能称为 3 米；配置保留请求的 3m 作为待确认目标，实际物理字段为 null。

## 共用指标

`metrics.py` 与既有 HBGNet Protocol B 指标文件逐字节一致。`evaluation_adapter.py` 负责输入校验与全局聚合，不修改指标定义。

1. F1、IoU、OA、Precision、Recall：在全部图像上累计 TP/FP/FN/TN 后计算；不以逐图平均替代 micro。
2. BF1、Boundary Completeness、Boundary Correctness：从区域预测及区域真值提取 1 像素 inner boundary，使用配置的圆盘像素容差；累计匹配/总数后统一计算。训练使用的 Canny Boundary target 不替代正式区域派生边界。
3. GOC、GUC、GTC：预测和真值均按 8 连通域标记；每个预测对象匹配与其重叠面积最大的真值对象，计算错误并按预测对象面积在整个 split 上加权，三者越低越好。
4. 窄前景、窄背景专门指标没有冻结定义，保留缺失，不填零。

## 空标签与指标边界

- 空标签图保留。其误检进入 FP、预测边界和预测对象误差；漏检进入 FN。
- 预测对象与全部真值对象均无交集时，该预测对象的 OC/UC/TC 各为 1，仍按其预测面积加权。
- 没有任何预测前景时，GOC/GUC/GTC 为 null，表示没有可评估的预测对象，不能解释为完美结果。
- 原指标仅遍历预测对象，完全漏掉且没有预测重叠的真值地块不进入对象误差分母。漏检必须结合 FN、Recall、Completeness 等一起报告。
- 空图不增加 micro 的人工逐图权重；若整个 split 的预测与 GT 均为空，沿用共用函数的 F1/IoU/BF1=1 特殊约定。因此所有结果应同时发布 pixel_counts、boundary_counts、预测对象面积及空标签数量。
- 逐图记录包含 pixel_counts、boundary_counts、object_numerators 和 predicted_object_area，可独立复算全局聚合。缺失值不能填 0。

## 与 AI4Boundaries 的区别

这是新建的 FHAPD 空间划分，非官方 FHAPD 划分。256 网格和隔离带证明的是文件名坐标下同一 parent 的块间距离；缺少跨 parent 地理位置证据，不声称完整地理独立性。

输入、归一化、增强和指标算法与现有接口一致，但标签来源不同，边界物理容差尚未确认。不能宣称与 AI4Boundaries 3m 协议完全相同，也不能将指标直接解释成对 HBGNet 论文官方划分的复现。

## 后续运行包约束

数据配置的 `adaptation.run_config_ready=false`。训练包需独立冻结模型/seed/run_name、训练损失监督落点、源码 manifest、工程验收和数据指纹；各消融组共用已冻结数据。配置中的训练配方是参考，未授权通过本适配工具启动训练。

正式测试需另行明确授权，使用对应新 run 的 best.pt、完整本区域 test 清单和独立 test_name。训练完成不自动启动测试；不得用 AI4Boundaries 的 checkpoint 身份、3014/3067 计数或输出目录冒充 FHAPD。
