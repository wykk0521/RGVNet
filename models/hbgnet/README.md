# HBGNet：AI4Boundaries Protocol B 对比模型

## 当前范围

目标目录：`/root/wyk5/RGVNet/models/hbgnet`。独立 RUN_NAME：`hbgnet_protocol_b_v1`。

本包是 **官方 HBGNet 在当前 SEANet Protocol B 上的适配复现**，不是论文原数据、原预算、原表格数字的逐项复现。工程验收和正式就绪状态分别见 `verification.json`、`status.json`。本次不启动完整训练或正式测试；不修改 SEANet 根模型、不修改或中断 v6。

## 官方来源和适配边界

- 农业地块 HBGNet：<https://github.com/NanNanmei/HBGNet>。
- 论文：*A large-scale VHR parcel dataset and a novel hierarchical semantic boundary-guided network for agricultural parcel delineation*，ISPRS Journal of Photogrammetry and Remote Sensing，2025。
- `upstream/` 保留下载的官方原文件；`source_manifest.json` 记录 8 个文件 SHA256，并与 jsDelivr 文件列表哈希核对。GitHub commit API 不可用，因此不声称已锁定某个 commit。
- `implementation/model.py` 只改包内 import、可配置预训练路径及严格加载检查；所有 forward 保持官方原文。`losses.py` 和 `lib/pvtv2.py` 与官方文件逐字节一致，差异见 `upstream_adaptation.patch`。
- 保留 PVT-v2-B2、Dropout2d(0.4)、可训练 Laplace 初始化卷积、近/远上下文、边界引导、多尺度卷积及 CBAM。未添加 SEANet 小波候选、关系路由、keep 或轮廓损失。
- 输出依次是 Region 原始 logits `[B,1,H,W]`、边界双类 log-softmax `[B,2,H,W]`、距离原始预测 `[B,1,H,W]`。距离不擅自加 sigmoid。
- 上游有未使用参数，保留以避免结构改变，不将这些参数的无梯度视作接线错误。
- 未在上游目录列表发现 LICENSE；公开可见不等于已获公开再分发许可。本包用于用户指定的私有实验，不宣称 MIT 等许可。

## 比较配置

| 项目 | 设置 |
|---|---|
| 数据 | AI4Boundaries_1m / parcel_focused / Protocol B |
| 划分 | 固定 train/val/test 清单：13890/3067/3014，不随机重分 |
| 输入 | RGB [1,2,3]，256×256，/255 后 ImageNet normalization |
| 标签 | mask band 1 Region、band 2 Boundary、band 3 Distance |
| 增强 | 原水平翻转、垂直翻转、90°旋转，各概率 0.5 |
| 预算 | 40 epochs，seed 41，batch 8，累计 2，有效 batch 16，workers 8，AMP false |
| 损失 | 官方 LossF：[1,1,1] × (BCEDiceLoss + 双类 NLL + MSE) |
| 优化 | 官方 Adam，lr 1e-4，weight decay 0 |
| 调度 | 官方 CosineAnnealingLR，T_max=1e10，eta_min=1e-5；40 轮内几乎恒定 |
| 验证 | epoch 1、每 2 轮、最后 5 轮 |
| 评估 | Region 阈值 0.5，Region 派生一像素内边界，3px/3m disk 容差 |
| 聚合 | Region/Boundary micro；对象 8 连通、预测对象面积加权全局汇总 |

相对官方脚本的明确差异：40 轮替代默认 100 轮；固定父图拆分清单替代随机 80/20；累计 2；使用本项目数据增强、三波段原生标签和评估/选模规则。不同模型保留各自官方优化器和损失，因此属于**共同数据与训练预算下的模型配方比较**，不是只替换结构、其余全部相同的控制实验。

训练验证沿用已有 `sigmoid >= 0.5`，最终评估沿用已有 `sigmoid > 0.5`；恰好等于 0.5 时两者存在边界差异，未静默更改历史实现。辅助 edge 阈值 0.84 留在共享配置中，但不用于正式 Region BF1。无 TTA、无小目标删除、无额外后处理。本项目 3m 容差不同于论文 5m，不能直接对照论文数值。

## 权重选择与恢复

- `best.pt`：最大化验证 `0.7×IoU + 0.3×BF1`，同分保留较早 epoch。
- `best_iou_constrained.pt`：在验证 `IoU >= 0.8205741376917083` 中按 BF1、IoU 选择，同分保留早轮。无合格 epoch 就不生成该文件。此线沿用 v6 的 seed41 v5 对照，不是 HBGNet 论文要求。
- 与 v6 比较时，HBGNet `best.pt` 对 v6 `best_legacy_score.pt`；HBGNet `best_iou_constrained.pt` 对 v6 `best.pt`。不可跨选模规则混比，额外 checkpoint 不意味着允许反复用 test 选模型。
- 只有本 RUN 的 `last.pt` 可恢复；包含 Adam、scheduler、epoch、Python/NumPy/Torch/CUDA 及 train/val loader RNG、源文件/清单哈希和预训练来源。
- `last.pt` 是 epoch 事务的提交依据；可恢复提交后未物化的指标/最佳文件。提交前残留的孤立 pending 记录会明确拒绝接续，需审查，不能静默丢弃。
- 新训练拒绝已有日志/权重目录，训练锁防重复。不会自动启动测试。正式测试须在 40 轮完整结束后显式指定已选 best；完整 ID 集、3014 PNG 尺寸/值域和总像素计数通过才标记完成。

## 环境与隔离

Python：`/root/miniconda3/envs/py3.11torch2.0.1cu118/bin/python`。

新增纯 Python 依赖仅位于 `models/hbgnet/_deps`：timm 0.6.13、huggingface_hub 0.25.2、fsspec 2024.6.1。使用 `--no-index --no-deps --target` 安装项目内 wheel；不升级 Conda 的 torch/torchvision，不修改 `.python_packages`。

```bash
ROOT=/root/wyk5/RGVNet/models/hbgnet
PY=/root/miniconda3/envs/py3.11torch2.0.1cu118/bin/python
export PYTHONPATH="$ROOT/_deps:/root/wyk5/RGVNet/.python_packages:/root/wyk5/RGVNet"
```

## 必需的 PVT 预训练权重

位置：`pretrained/pvt_v2_b2.pth`，来源登记：同目录 `pvt_v2_b2.json`。新正式训练缺少权重/来源或键缺失即报错，**不允许随机初始化回退**。

已核实官方来源：

- PVT 官方 README：<https://github.com/whai362/PVT/blob/v2/classification/README.md>，PVTv2 on ImageNet-1K 表中的 **PVT-V2-B2**，不是 B2-Linear。
- 官方 release：<https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth>。
- HBGNet README 自身链接：<https://drive.google.com/file/d/1uzeVfA4gEQ772vzLntnkqvWePSw84F6y/view?usp=sharing>。
- PVT 官方 Drive 备选：<https://drive.google.com/file/d/1snw4TYUCD5z4d3aaId1iBdw-yUKjRmPC/view?usp=sharing>。

本次下载受阻：本地 GitHub/Drive 超时，服务器 GitHub TLS 证书验证失败。没有绕过 TLS，没有用不明镜像或随机权重冒充预训练。

取得官方文件并放到上述路径后，可用下式严格检查并登记操作者提供的来源（不会训练）：

```bash
"$PY" -B "$ROOT/register_pretrained.py" \
  --source-url https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth
```

登记脚本使用 `weights_only=True`、CPU、所有 backbone key/shape/finite 检查，只允许分类 head 两个额外 key。SHA256 固定本地字节；没有发布者 checksum 时，不能把本地 SHA 当成发布者身份认证。

## 可复用入口（本次未启动正式实验）

CPU 工程验收（随机初始化，只读少量 train/val；不读 test 标签）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PY" -B -u "$ROOT/verify_hbgnet.py"
```

正式训练（须先补齐官方权重、确认 GPU 容量并获得训练任务授权）：

```bash
bash "$ROOT/run_train.sh"
# 断点恢复只允许 last.pt：
bash "$ROOT/run_train.sh" --resume /root/wyk5/RGVNet/z_3_weights/AI4Boundaries_1m/hbgnet_protocol_b_v1/last.pt
```

完整训练结束后，显式执行一次匹配选模规则的正式测试：

```bash
bash "$ROOT/run_evaluate.sh" \
  --checkpoint /root/wyk5/RGVNet/z_3_weights/AI4Boundaries_1m/hbgnet_protocol_b_v1/best.pt \
  --test-name hbgnet_protocol_b_v1_best
```

训练日志/权重和测试日志/预测分别在当前 `z_1_trainning_log`、`z_3_weights`、`z_2_test`、`z_4_results` 的 AI4Boundaries_1m 下独立 RUN/TEST 子目录，不覆盖 SEANet 实验。
