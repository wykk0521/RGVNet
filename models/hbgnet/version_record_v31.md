

### V31：HBGNet 对比模型 Protocol B 独立接入与 CPU 工程验收

- 2026-09-12，按用户“在这个model里根据现有配置、配置复现hbgnet 作为对比模型”执行。确认根 `model/` 不存在、`models/` 为空后，在 `/root/wyk5/RGVNet/models/hbgnet` 独立部署。未修改 SEANet 根模型，也未修改、中断或重启正在运行的 v6；52 项根代码/v6 源码配置受保护哈希前后完全一致。
- 官方来源为农业地块 NanNanmei/HBGNet，不是同名机器人抓取模型。8 个上游文件与 jsDelivr 目录哈希逐文件核验；保留原文件、适配 diff、source_manifest 和官方 PVT README 来源证据。GitHub commit API 不可用，不虚构 commit；未找到 LICENSE，不宣称 MIT 或公开再分发许可。
- 模型：官方 Field/PVT-v2-B2，channel32、边界2类、dropout0.4；保留可训练 Laplace、近远上下文、边界引导、多尺度融合与 CBAM。30,453,790 参数。只改包内 import 与严格可配置预训练加载，全部 forward AST 保持一致；loss/backbone 与上游逐字节一致。Region 原始 logits、双类 boundary log-softmax、distance 原始值；不添加 SEANet 小波/关系/轮廓损失。
- 独立配置 `models/hbgnet/config_protocol_b.yaml`，SHA256=`cd234aa0b758b48df384a7d4a74d3fceaf4f177e575a18cffec1efd4d91d79da`。RUN_NAME=`hbgnet_protocol_b_v1`；固定 Protocol B/parcel_focused/256/RGB/13890-3067-3014 清单；40epochs/seed41/batch8/累计2/workers8/AMPfalse。共享 loader、增强、归一化、原生三波段标签、阈值及评估。官方 LossF 权重[1,1,1]：BCEDice + 双类NLL + MSE；官方 Adam1e-4/weight_decay0，保留 Cosine T_max1e10/eta_min1e-5（近恒定学习率）。40轮/固定划分/累计2等是本项目适配，不冒充论文原预算或表格复现；本项目边界3m亦不同于论文5m。
- best.pt 保留原0.7IoU+0.3BF1；额外 best_iou_constrained.pt 在 IoU≥0.8205741376917083 下按 BF1/IoU/较早epoch 选择。对比v6须匹配对应选模规则，不交叉混比。last.pt 含 optimizer/scheduler、所有 RNG、源码/清单哈希与预训练来源；只有该run last可续训，最佳权重不用于接续。epoch事务以last为依据恢复已提交但未写完的记录/最佳文件，孤立的未提交证据明确拒绝覆盖。
- 训练/测试入口分别 train.py/evaluate.py 与 run_train.sh/run_evaluate.sh。禁止已有输出覆盖，训练锁防重复；不自动测试。最终评估仅Region sigmoid>0.5、3px disk、micro与对象全局面积加权，无TTA和后处理；完整样本ID、PNG尺寸/uint8/0-255和总像素数核验后才标complete。训练验证沿用历史>=0.5口径，披露等于0.5时差异。
- 仅本目录 `_deps` 安装 timm0.6.13、huggingface_hub0.25.2、fsspec2024.6.1；torch仍2.0.1+cu118，不更改Conda/.python_packages。37个包文件部署SHA核验成功。
- CPU隐藏CUDA、2线程、nice10下16项工程验收全部通过（exit0）：官方状态/256前向精确一致、输出契约、官方loss、Laplace、真实train一batch反向和Adam更新、478参数张量梯度有限、全模型独立实例checkpoint严格恢复、val一batch、共享指标5类样例、配置/缺预训练/缺key反例、RNG及Adam下一步精确恢复、epoch事务故障注入恢复/孤立证据拒绝、PNG正反例。未读test标签、未生成正式日志或权重、未保留smoke权重。仅TypedStorage弃用提示。验收结果在 verification.json/verification_console_01.log/verification_exit_01.txt；随机初始化的工程指标不作为精度结论，未验收batch8的GPU容量。
- **当前正式训练阻塞：缺官方ImageNet PVT-v2-B2预训练权重。** 已确认 PVT 官方release `https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth` 与官方Drive备选，但本地请求超时，服务器TLS证书验证失败；没有关闭TLS、没有不明镜像、没有随机初始化冒充预训练。要求 `pretrained/pvt_v2_b2.pth` 和有来源/SHA的receipt；register_pretrained.py 可在取得权重后严格检查。真实train入口反例确认缺权重即拒绝且不创建正式输出。
- 本次完成代码、配置、隔离依赖和工程接入，未启动HBGNet正式训练、未进行完整test，不存在HBGNet精度对比结果。v6继续正常写日志，交付检查时已完成7轮、正在后续轮次；未以本次HBGNet任务擅自启动v6测试。
