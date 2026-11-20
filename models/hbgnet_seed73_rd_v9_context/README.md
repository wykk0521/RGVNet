# Seed73 rd_v9_context independent experiment

run_name: `hbgnet_protocol_b_seed73_rd_v9_context`; seed: 73; physical GPU: 2.
Start from the official PVT-v2-B2 pretraining, never from a v8 or SR checkpoint.

# HBGNet rd_v9_context fullloss + SR v2

唯一可运行配置：`config_protocol_b.yaml`，run=`hbgnet_protocol_b_seed73_rd_v9_context`。本包必须先通过独立工程验收与
GPU smoke，才允许启动正式训练；本包不自动启动 test。

GPU smoke 的候选语义（`verify_rd_v9_context.py --real-smoke`）：SR refiner 末 head 零初始化，
所以初始 `delta_sr == 0`、`sr == raw`，自然阶段可能一个候选像素都没有，此时 verifier loss 是连图
零、context 分支合理地收不到梯度——这是被记录的正常事实，不是失败。只有当某方向真的出现候选像素
时才要求该方向 head 必须移动。若自然阶段缺候选，smoke 会在一个**独立构建、不写 checkpoint**的
探针模型上按 `±1.0 / ±2.0` 顺序做强制候选探针（每次尝试新建 optimizer，避免上一轮动量造成假位移；
step 前先要求目标方向 head 梯度 finite 且非零，只改探针模型的 refiner bias；探针结束后用 `torch.equal`
校验 Python/CUDA RNG 已精确恢复，并对被验收模型做覆盖全部参数/缓冲名称、dtype、shape 与原始字节的
SHA256 state digest 比对），逐方向记录真实 batch 证据；若某方向在真实 batch 中仍无候选，再用**合成
detached CUDA 输入**（raw/sr/delta/truth 直接构造，与训练一样 detach，单独构造只让一个方向有候选、
并校验另一方向梯度恒为 0）验证该方向的梯度与更新，报告中以 `evidence` 字段明确区分两类证据。
**cut 与 fill 两个方向都必须被验证**，否则 smoke 失败、不允许启动；自然模型、自然 optimizer 与 val
路径始终不被扰动（探针 never 复用自然 optimizer）。

`fullloss` 指设计的完整联合目标，不是给 gated final 另加 Region loss：
`L_total = L_sr_v2 + 1.0 * mean_present_directions(L_cut, L_fill)`。`L_cut/L_fill` 是普通
`BCEWithLogits`，只在真实候选像素上计算，并只平均当前 batch 中实际出现的方向（无候选方向是连图
零；单个方向出现时不做二次折半），再对出现的方向取平均。

## 模块路由

SR v2 从 `F、z0、edge_logp` 产生局部 Region 残差，得到 `zs = z0 + delta_sr`。rd_v9_context
只读取实际发生二值变化的位置：cut 是前景到背景，fill 是背景到前景。验证器输入是 detached 的
`F、tanh(z0)、tanh(zs)、tanh(delta_sr)、|tanh(delta_sr)|、edge probability、cut/fill mask 及上下文`，
输出 `[B,2,H,W]` 的 cut/fill logits。严格 `logit > 0` 才接受；接受时保留完整 SR 残差，拒绝时精确
回到 raw，不发生候选之外的新编辑。v9 与 v8 的唯一差别是新增的上下文残差：

`z9,d = z_local,d + r_d`，其中 `z_local,d` 是逐字保留的 rd_v8_grok 评分（v8 模块被直接
import，v9 是其子类），`r_d` 来自方向私有的轻量上下文分支。

## 上下文分支规格（每方向 131 通道描述子）

- 共享可学习 1x1 投影：detached F64 → 8 通道嵌入 `E`（只由 verifier loss 更新）。
- 中心块 14 通道：`E`(8) + `tanh(raw)/tanh(sr)/tanh(delta)/|tanh(delta)|`(4) + 前景 edge 概率(1)
  + 本方向候选标志(1)。
- 支持块 117 通道 = 3 掩码 × 3 尺度 × 13 通道。3 掩码为本方向候选 `C_d`、所有候选外的 raw 前景
  `S_fg`、所有候选外的 raw 背景 `S_bg`；尺度固定 3/7/15；每个 (掩码,尺度) 的 13 通道为
  掩码加权 `E` 均值(8)、掩码密度(1)、掩码加权 `tanh(raw)/tanh(sr)/tanh(delta)`(3)、掩码加权前景
  edge 概率(1)。密度分母只计窗口内落在图像内的像素（`count_include_pad=False`），越界取零、
  不回卷；空支持均值置 0、密度置 0 显式标识缺失。支持区只是 raw 与 SR 未翻转的位置，不是真值。
- 方向私有编码器：1x1 131→16、channel LayerNorm、SiLU、depthwise 3x3 残差、1x1 16→1 输出 `r_d`；
  末 head 的 weight/bias 零初始化，因此加载同一 v8 参数后新增分支初始输出精确为 0，`z9` 初始
  逐位等于 v8 的 `z_local`。首步只有 head 得到非零梯度、上游 context encoder 梯度为 0，这是零
  初始化 head 的性质；head 更新后整支学习，单元测试两阶段都验收。

参数量：v8 验证器 6,074，新增上下文 5,154，v9 验证器合计 11,228；Field 合计 30,468,959。两个
head 的零初始化只保证初始候选全部拒绝，不保证联合训练后性能保持。

## 规格变更记录（相对 v9 设计草案）

| 项目 | 设计草案 | 实现 | 原因 |
| --- | --- | --- | --- |
| 描述子宽度 | 126 | **131** | 为保证对照与主变体的中心完全一致，中心块显式加入中心 margin
`tanh(raw)/tanh(sr)/tanh(delta)/|tanh(delta)|`(4) 与前景 edge 概率(1)，`9+117 → 14+117`；主 agent
已认可 131 规格 |
| 上下文参数量 | 4,994 | **5,154** | 上述 +5 通道 × 16 宽 1x1 stem 无 bias，共 +80/方向，cut/fill 两方向
合计 +160 |

中心块与支持块的 channel 索引由 `descriptor_layout()` 与 `center_channel_index` /
`support_channel_index` 固定并验收；两个 aggregation 变体共用同一 `(center, support)` 通道布局。

## 对照与校准（本轮不训练对照）

- `aggregation="center_only"` 是中心对照：中心块相同，支持块由同样的 13 通道配方在像素本身
  （窗口 1x1）上计算并按掩码门控，因而骨架与名义参数预算相同且没有死零输入；但三个尺度 slot
  承载相同统计量，第一层存在线性冗余，所以它只是「同骨架、同名义预算的 1x1 聚合消融」，不能据此
  宣称有效容量已完全对齐，也不能排除容量因素。本轮只训练 `multi_scale_support` 主变体，对照身份
  在 `RESERVED_RUNS` 中保留、不训练。
- 两个 aggregation 变体的参数名与形状完全相同，因此 verifier 注册了持久 buffer
  `aggregation_code`（主变体 0 / 对照 1），并在 `_load_from_state_dict` 中先于任何张量拷贝校验；
  跨变体加载（含先 load 再改 `aggregation` 属性）会直接抛错，不会被静默当成同构权重。
- 方向独立单调 affine 校准 `a_d > 0, b_d` 与 BR/HA 曲线工具只用于诊断：`a>0` 保持排序，校准
  只把工作点沿同一条 BR–HA 曲线移动（`calibrated > t` 等价于 `z > (t - b_d)/a_d`），因此分方向
  报告零门限与曲线时不得混为排序提升。系数在**转成 logits dtype 之后**校验 `a>0` 与有限性
  （half 下溢到 0、上溢到 inf 均拒绝），校准输出若非有限则抛错而不是返回 inf。

## 监督和阈值

Region/SR 监督为：`0.5 Lregion(raw) + 0.5 Lregion(sr) + 0.125 Lband(raw) + 0.125 Lband(sr)
+ Lboundary + Ldistance`。验证器只在真实 cut/fill 候选像素上使用普通 BCE；每个有候选的方向各自
归一化，再对出现的方向取平均；无候选方向是连图零。Verifier 输入 detach，Verifier loss 只更新
verifier 参数；不加 final Region loss、不加正类重权、不改门限、不做 test 调参。

路由使用严格 `logit > 0`，验证指标保留冻结历史口径 `sigmoid >= 0.5`，正式 test（未来单独授权后）
保留 `sigmoid > 0.5`。

`protocol/candidate_diagnostics.py` 的四格表是 **benefit × accept** 候选诊断（候选是否真的有益 ×
打分是否接受），描述当前打分器的 BR/HA；它不是 v8→v9 的迁移表，两者的行列表义不同，不得互相
解释。本轮不运行设计 A 阶段的固定候选机制实验。

checkpoint 的有限性扫描（`evaluate.py`）按张量逐个检查：整数 `aggregation_code` 这类整型 buffer
只做类型检查、不做 `isfinite`，浮点张量仍必须有限，因此新增 buffer 不会让原扫描漏检或误报。

## 四组消融身份

原 baseline / baseline+RD / baseline+SR v2 / baseline+RD+SR v2 身份在 `common.ABLATION_GROUPS`
中保留。baseline 和 baseline+SR v2 指向已有冻结历史包；baseline+RD 在本 post-SR verifier 架构下
没有候选编辑，明确标为 `reserved_not_representable`。本包只允许 combined 主组进入训练。

## 运行保护

配置固定使用 `/root/wyk3/dataset/AI4Boundaries_1m_parcel_focused_256`、包内官方 PVT-v2-B2 预训练
权重及 wyk3 日志/权重根目录。正式起点只能是官方 PVT-v2-B2，不加载任何历史 HBGNet、SR、RD 或
工程权重，禁止从 v8 checkpoint resume。训练入口要求 `package_manifest.json` 与 `verification.json`
和源码完全一致，拒绝配置漂移、输出覆盖和重复进程；中断只允许本 run 的 `last.pt` 续训，`best.pt`
只留给未来明确授权的 test。

本次只启动 `hbgnet_protocol_b_seed73_rd_v9_context`，绑定物理 GPU2；不自动 test、不启动其他消融、不修改或删除旧资产。
v9 设计已知边界：设计草案的 A 阶段固定候选诊断未执行，本轮按用户决定直接正式重训；重训后
raw/SR 分支与候选集合不承诺与 v8 逐位相同。

启动审计回执的语义：`finally` 只做清理（移除 forward hook、还原被打补丁的 `train.build` /
`train._finite_step` / `train.loss_fn`），不写 `audit_complete`、不发出 `training_continues`；
只有 `maybe_finish_audit` 在观察到 >=5 个 loss、>=2 次 optimizer 更新且 refiner/verifier 各
>=5 次前向之后才标记成功。因此启动验收必须同时核对 `activity.jsonl` 的 5 loss/2 update 记录、
进程存活且无 `exit.json`，不能只看 `audit_complete` 布尔值。`verify_v9_diagnostics.py` 的
`launcher_audit_receipts` 用桩模块复现早退路径，断言早退不会被标成成功。
