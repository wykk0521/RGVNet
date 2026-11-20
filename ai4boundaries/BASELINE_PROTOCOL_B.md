# Protocol B 固定基线

- 正式名称：`seanet_protocol_b_baseline`
- 原实验名称：`seanet_protocol_b_baseline_v2_1`
- 状态：后续小波候选与关系验证实验的固定主基线
- 配置：`/root/wyk5/RGVNet/ai4boundaries/config_baseline.yaml`
- 训练日志：`/root/wyk5/RGVNet/z_1_trainning_log/AI4Boundaries_1m/seanet_protocol_b_baseline/`
- 权重：`/root/wyk5/RGVNet/z_3_weights/AI4Boundaries_1m/seanet_protocol_b_baseline/`
- 正式测试：`/root/wyk5/RGVNet/z_2_test/AI4Boundaries_1m/seanet_protocol_b_baseline_best/`
- 预测图：`/root/wyk5/RGVNet/z_4_results/AI4Boundaries_1m/seanet_protocol_b_baseline_best/predictions/`
- 完整清理前备份：`/root/wyk5/RGVNet/ai4boundaries/backups/baselines_pre_cleanup_20260907.tar`
- 校验清单：`/root/wyk5/RGVNet/ai4boundaries/baseline_protocol_b_manifest.json`

训练固定为 Protocol B / 256、40 epochs、seed 41、batch size 8、梯度累计 2。主 checkpoint 仍按 `0.7 × Region IoU + 0.3 × region-derived Boundary F1` 选优，`best.pt` 来自 epoch 37。

正式测试结果：F1=`0.8952677684`、IoU=`0.8103934536`、F_boundary=`0.5053516961`、GOC=`0.0809226030`、GUC=`0.2283095632`、GTC=`0.1965538306`。后续正式消融必须使用相同数据、训练预算、seed、选优规则和测试协议，新增实验使用独立名称，不覆盖本基线目录。
