# HBGNet-metrics-ProtocolB-256-3m 统一评测协议

本协议用于 AI4Boundaries_1m Protocol B 上的所有区域分割与区域-边界多任务模型。它借鉴 HBGNet 的指标体系，但使用本研究从官方父级划分确定性裁剪得到的 256 x 256 parcel-focused 数据，因此不将结果表述为对 HBGNet 论文表 4 的直接复现。

## 固定推理设置

- 测试划分：`/root/wyk2/dataset/AI4Boundaries_1m_parcel_focused_256/manifests/test.txt`，共 3014 张 256 x 256 图像块。
- 输入尺寸：256 x 256，原始空间分辨率 1 m/像素。
- Protocol B 沿用 Protocol A 的父影像 train/val/test 划分；每张 512 x 512 父影像无重叠裁剪为四张 256 x 256 图像块，并且只删除区域前景像素为零的图像块。
- 最终区域输出经过 sigmoid，以 0.5 二值化。
- 不使用测试时增强（TTA）。
- 不进行小目标删除或其他形态学后处理。
- 区域指标在全部测试像素上累计 TP、FP、FN、TN 后统一计算（micro）。
- 边界指标在全部测试图上累计匹配边界像素数和边界像素总数后统一计算。
- 对象指标汇总全部预测地块，并按照预测地块面积进行全测试集加权。
- 测试集只用于最终报告，不用于选择阈值或 checkpoint。

## Checkpoint 选择

- 只在完整验证集上选择 checkpoint。
- 第 1 轮、每 2 轮以及最后 10 轮执行完整验证；第 100 轮必定验证。
- `checkpoint_score = 0.7 * val_region_iou + 0.3 * val_region_boundary_f1`。
- `val_region_boundary_f1` 与正式测试一致：从阈值 0.5 的区域预测与区域真值提取 1 像素内边界，并使用 3 m（3 pixels）容差。
- 验证端采用经过逐计数等价性测试的 GPU 批量边界统计；正式测试端保留 CPU 实现，两者使用相同的 4 邻域内边界和 3-pixel 圆盘容差。
- SEANet 直接边界头的严格逐像素 F1 仅作为训练诊断指标，不参与 checkpoint 选择。

## 主表指标

1. `F1`：由全测试集累计 TP、FP、FN 计算的区域二分类 micro-F1。
2. `IoU`：由全测试集累计 TP、FP、FN 计算的区域二分类 micro-IoU。
3. `F_bdy`：边界 F1。分别从预测区域和真值区域提取 1 像素内边界；AI4Boundaries_1m 使用 3 m，即 3 像素匹配容差。`Completeness` 表示真值边界落入预测边界缓冲区的比例，`Correctness` 表示预测边界落入真值边界缓冲区的比例，二者调和平均为 `F_bdy`。HBGNet 论文原设置为 5 m，本研究为提高精细边界评价的区分度而固定改为 3 m，因此不得把本研究的 `F_bdy` 与论文表中 5 m 结果直接比较。
4. `GOC`、`GUC`、`GTC`：对预测区域进行 8 邻域连通域标记。每个预测地块与重叠面积最大的真值地块匹配；按照 HBGNet 论文公式计算单地块 OC、UC、TC，再按预测地块面积加权。

## 诊断指标

`OA`、`Precision`、`Recall`、边界 `Completeness` 和 `Correctness` 保存在测试日志中，但不进入论文主对比表。逐图 F1、IoU 和 F_bdy 的宏平均只作为失败案例诊断，不作为正式结果。

## 特殊情况

- 空白测试图不删除。空白图上的误检进入全局 FP 和预测边界总数；漏检进入全局 FN。
- 预测与真值均为空的图不会人为给正式 F1/IoU 增加一个 1，也不会增加边界或对象权重。
- 预测对象与所有真值对象都没有交集时，该预测对象的 OC/UC/TC 均记为 1，并按其面积进入全局对象指标。

## 输出

- 主指标和诊断指标：`z_2_test/AI4Boundaries_1m/<test_name>/metrics.json`
- 逐图指标及汇总：`z_2_test/AI4Boundaries_1m/<test_name>/accuracy.txt`
- 二值预测图：`z_4_results/AI4Boundaries_1m/<test_name>/predictions/`

后续对比模型必须复用上述定义，不得单独启用 TTA、改变阈值、修改容差或增加模型专属后处理。
