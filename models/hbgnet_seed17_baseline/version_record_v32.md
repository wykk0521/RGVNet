

### V32：用户上传 PVT 权重后，HBGNet 正式40轮训练启动

- 2026-09-12，用户“已经拖入 开始训练”，确认与现有协议的任务/轮数等一致后，再次授权“启动hbgnet训练”。按当前配置从第1轮启动，不从工程预检状态或SEANet权重接续，不自动测试。
- 用户放入 `models/hbgnet/pretrained/pvt_v2_b2.pth`，101501299 bytes，SHA256=`80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d`。CPU weights_only读取，334项中332项backbone严格加载，额外仅分类head权重/偏置，所有张量finite。登记 `pvt_v2_b2.json`：明确为用户提供的文件、官方预期链接，未独立核实实际下载来源或发布者checksum；不把本地SHA说成发布者身份验证。无随机回退。
- 预启动发现 train.main 持锁后又调用会再次加锁的 train 函数，属于工程入口问题；最小改为直接调用已持锁的 `_train_locked`，保留 CLI/程序调用各自的单进程保护；并在原config_snapshot.json外新增同字节config_snapshot.yaml。旧源码、manifest、verification和状态已存 `launch_fix_backup/`。模型/损失/优化器/数据/阈值/轮数均不变，配置SHA仍为`cd234aa0b758b48df384a7d4a74d3fceaf4f177e575a18cffec1efd4d91d79da`。更新源码manifest后重跑16项CPU验收，exit0；另验证CLI实际进入一次、重复锁被拒绝。
- GPU预检用新实例加载用户PVT权重，真实batch8两批、累计2执行一次Adam更新，再验证一批val8；backbone、Laplace、Region/edge/distance三个head参数均更新，loss/grad/参数finite。峰值allocated6775.56MiB、reserved8564MiB，未写入工程权重、未读取test标签。正式训练另建实例并重置seed41，不继承预检更新。
- 正式RUN/tmux=`hbgnet_protocol_b_v1`，入口`models/hbgnet/train.py`，监督脚本`models/hbgnet/launch_training.py`。UTC2026-09-12T18:51:51启动，主PID435175、supervisor435173。独立train.lock和launch.lock防重复，console/launch/exit evidence记录于`models/hbgnet/formal_launch_01`；训练日志目录中console.log指向本次真实输出。不改模型训练过程、不自动测试。
- 当前正式配置：HBGNet/PVT-v2-B2，Region/Boundary/Distance三任务；Protocol B/parcel_focused、256/RGB、固定13890/3067/3014清单；40epochs、seed41、batch8、累计2、workers8、AMPfalse。官方LossF[1,1,1]，Adam1e-4/weight_decay0，cosine T_max1e10/eta_min1e-5。best按0.7IoU+0.3BF1，附加IoU约束best供匹配v6选模规则比较；只last可续训。
- 启动核验：首轮batch20与40连续推进，loss分别1.455669和1.395488，finite；没有Traceback、NaN/Inf、CUDA OOM或数据异常。主进程与8个数据加载worker不能误判为重复训练。A40上HBGNet约8920MiB、v6约18278MiB、原显存占据9604MiB，空闲约11721MiB，GPU利用率100%。v6主PID426205继续推进，52项受保护根代码/v6源码配置哈希不变。
- 训练日志：`z_1_trainning_log/AI4Boundaries_1m/hbgnet_protocol_b_v1`，含config_snapshot.json/yaml、TensorBoard、launch/verification/preflight/权重来源记录。权重：`z_3_weights/AI4Boundaries_1m/hbgnet_protocol_b_v1/{last.pt,best.pt,best_iou_constrained.pt}`。当前首轮未结束，尚无正式val、best epoch或checkpoint；不宣称首轮落盘已验收。首轮末val3067样本并按事务保存last/best；无合格IoU时额外约束best不生成属正常。
- 仅既有TypedStorage弃用及CuDNN nvrtc.so workaround提示，未为警告修改依赖。一次等待日志的SSH工具调用在宿主30秒处超时，随后只读复查确认训练正常，没有因此重启或重复启动。正式退出码仅在进程结束后写入。
