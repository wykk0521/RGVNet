# HBGNet SR v2：双输出边界邻域类别监督

独立实现v2首选主实验及核心监督消融，当前正式训练/测试未启动。是否工程通过以当前源码指纹匹配的verification.json为准，不复用v1验收结果。

## 本次实现范围

| 配置 | 运行名 | Region相关目标 |
|---|---|---|
| config_protocol_b.yaml | hbgnet_protocol_b_sr_v2 | .5原始Region＋.5最终Region＋.125原始band＋.125最终band |
| config_raw.yaml | hbgnet_protocol_b_sr_v2_raw | .5原始Region＋.5最终Region |
| config_band.yaml | hbgnet_protocol_b_sr_v2_band | 最终Region＋.25最终band |
| config_disabled.yaml | hbgnet_protocol_b_sr_v2_disabled | 无SR，原Region目标 |
| config_nosr_band.yaml | hbgnet_protocol_b_sr_v2_nosr_band | 无SR，原Region＋.25band |

各组另加原Boundary NLL和Distance MSE各一次。原Region=.5 BCEWithLogits＋per-image Dice，smooth1e-5。band为GT图内四邻接异类对的两端，经欧氏disk2膨胀；不虚构图外背景，类别标签不变。band损失按每图非空类别分别归一化后平均，再对全部图像平均；空band图贡献0。

保留v1的3941参数结构、对称五点/两侧/上下文、有效组平均、`delta=valid_any*2*tanh(a)`及zero-init。主实验**联合反传**；只detach输入旁路的Region/Boundary概率提示，final仍通过相加项和旁路F影响主干。它不是完全梯度隔离，不保证raw不退化或连通性正确。

默认forward保持三输出；`forward(image,return_training_aux=True)`显式返回三输出与连图raw Tensor。`last_auxiliary`仅用于detach诊断，不能当raw训练目标来源。

## 尚未实现的设计消融

`stopraw`、`stopall`、`linear`和`finalband`仅在设计文档中预留；当前配置守卫拒绝它们，没有静默fallback。本次不加入新的梯度隔离、无界残差、Mamba或band4。

## 运行与保护

仅在用户后续明确授权训练时使用 `run_train.sh --config <本包配置>`；训练不自动测试。中断只用本run last.pt。正式评估入口仅接受本run best.pt，保留原>0.5和3px协议，无TTA/后处理，不覆盖结果。best_iou_constrained仍是训练事务的历史次要产物，不能用于本版正式测试。

40轮、seed41、batch8累积2、Adam1e-4与原调度、RGB/256及固定split全部保留，从相同PVT预训练开始；禁止HBGNet best额外40轮冒充同预算。不同loss总值不直接横比。shared依赖在原HBGNet/_deps、项目.python_packages和Conda中，不复制/修改。

模块独立于v1：`tools/local_separation_refinement_v2.py`（当前与v1模块字节相同）和`tools/region_boundary_band_loss_v2.py`。不修改v1工具或tools/__init__.py。基线配置/损失/数据/指标/训练事务副本保留原字节。

## 验收与诊断

verify_sr.py执行GT band单测、公式/折叠/原loss等价、连图raw梯度、输入提示detach、开启/关闭/RNG/参数等价、模型序列化、锁与续训拒绝、两批16个真实train样本两次工程Adam更新（第二次复用同样本）、单张val指标链路。工程临时权重和optimizer状态不保留为正式实验。

训练日志输出loss分项；验证同时输出raw/final指标与FP/FN、GT band错误、远边界错误等诊断。单张val工程值不是性能结论。设计文档中的各地区全面比较和多种子确认不由工程验收替代。

完整设计保留在 `z_6_不同消融组代码/HBGNet/局部分隔结构细化/v2/v2设计方案.md`。该原始设计快照不覆盖；实现另存归档。文献查询未完成系统查新，当前不声称新颖性或性能提升。
