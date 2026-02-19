# SpatialVLA-LLaVA3D 当前框架深度总结（架构 + 算法）

## 1. 目标与当前状态
本项目当前目标是让策略网络在 LIBERO 场景下对语言与视觉输入均保持有效敏感性，并稳定输出 7 维动作轨迹。  
你最近的 sensitivity 结果已经证明修复后系统具备良好区分能力：

- `baseline_l2 = 0.0`（同输入同输出）
- `image_l2 = 0.471532`（固定指令、换图像后有非零差异）
- `synonym_l2 = 0.030927`（同义改写差异较小）
- `opposite_l2 = 1.485739`（相反语义差异显著增大）

结论：基础“可感知性”和“语义方向性”已经建立，下一阶段可以重点做算法增益。

## 2. 端到端架构（训练/推理主链路）
数据流主链路：

1. 数据加载（图像 + 指令 + 状态 + 动作标签）
2. VLM/Processor 处理多视角图像与文本，构建语言-视觉 token
3. 融合模型输出多层隐藏状态（含几何分支信息）
4. Action Head（Flow Matching + DiT + Cross-Attn）读取多层表示并预测动作序列
5. 训练时优化 `action_loss`；推理时从噪声轨迹积分生成动作

关键模块位置：

- 框架入口：`/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py`
- VLM 接口：`/Users/bazinga/code/my-starvla/starVLA/model/modules/vlm/MapAnythingLlava3D.py`
- Processor：`/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/processing_mapanything_llava3d.py`
- 融合模型：`/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py`
- 几何编码包装：`/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything.py`
- Action Head：`/Users/bazinga/code/my-starvla/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`
- DiT/Cross-Attn：`/Users/bazinga/code/my-starvla/starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py`
- 训练脚本：`/Users/bazinga/code/my-starvla/starVLA/training/train_starvla.py`
- 数据配置：`/Users/bazinga/code/my-starvla/starVLA/dataloader/gr00t_lerobot/data_config.py`
- 数据集实现：`/Users/bazinga/code/my-starvla/starVLA/dataloader/gr00t_lerobot/datasets.py`

## 3. 模块级算法解析

### 3.1 Framework（MapAnythingLlava3D_PI）
文件：`/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py`

- 负责组装 VLM + Action Head，读取 LLM 隐藏层维度与层数。
- 训练 `forward` 中将 VLM 输出的层特征按配置选择（first/last），再喂给 action head。
- 推理 `predict_action` 支持 deterministic seed，并返回丰富 `debug_info`（token signature、padding 模式、layer RMS、DiT 统计等），这是你做 sensitivity 诊断的关键可观测性。

### 3.2 VLM Interface
文件：`/Users/bazinga/code/my-starvla/starVLA/model/modules/vlm/MapAnythingLlava3D.py`

- 负责加载 MapAnything-LLaVA3D 权重、processor、配置项。
- 关键开关：`use_geometric_branch`、`image_token_joiner`、instruction normalization。
- 提供缺省相机内参 fallback，保证输入完整性。

### 3.3 Processor（图像 token 注入逻辑核心）
文件：`/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/processing_mapanything_llava3d.py`

- 定义并校验 `<image>` special token。
- 自动探测 joiner（`""` 或 `" "`），避免 tokenizer 切分异常。
- 将每个 `<image>` 扩展为 `image_seq_length * num_images` 占位 token，并在编码后与视觉特征位置对齐。

### 3.4 融合模型（2D + 3D）
文件：`/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py`

- 2D 语义特征来自 LLaVA vision tower（fallback 到 SigLIP）。
- 3D 几何特征来自 `MapAnythingWrapper`。
- 当前融合策略的主要形态：几何特征投影后全局平均，再广播到视觉 token，随后拼接并线性投影。
- 训练支持 prefix dropout（图像/语言 token dropout），有助于鲁棒性但可能引入信息抖动。

### 3.5 Action Model（Flow Matching + DiT）
文件：`/Users/bazinga/code/my-starvla/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`

- 学习目标：速度场（velocity）回归，噪声轨迹 `x_t=(1-t)z+t a`，目标 `v=a-z`。
- 训练损失为 velocity MSE，时间采样来自 Beta 分布。
- 推理从高斯噪声轨迹出发，用 Euler 离散积分迭代更新。
- Layerwise 机制：每个 DiT 层对齐读取对应 VLM 层（cross-attention）。

### 3.6 训练环节
文件：`/Users/bazinga/code/my-starvla/starVLA/training/train_starvla.py`

- 主要优化 `action_loss`（及内部 `action_dit_loss`），包含非有限值保护和梯度统计。
- 早停和日志以动作相关指标为核心，适合当前阶段。

### 3.7 数据接口
文件：`/Users/bazinga/code/my-starvla/starVLA/dataloader/gr00t_lerobot/data_config.py`、`/Users/bazinga/code/my-starvla/starVLA/dataloader/gr00t_lerobot/datasets.py`

- 输入包含多视角图像、语言、可选状态；动作为 7 维（xyz + rpy + gripper）。
- 训练和推理对 state 维度一致性要求较高，需要持续关注 pad 策略与归一化统计一致性。

## 4. 当前算法优势

- 可观测性强：`debug_info` 足够细，定位 tokenizer/image-token/padding 问题效率高。
- 结构解耦清晰：VLM 与 action head 边界明确，便于单独替换/升级。
- 已具备语言敏感性与图像敏感性，且同义/反义语义差异符合预期。

## 5. 当前算法瓶颈（最可能影响上限）

- 3D 几何融合粒度偏粗：全局池化后广播，细粒度空间对应关系可能被抹平。
- VLM 层利用方式较硬：层选择规则固定，缺少样本自适应路由。
- Flow Matching 推理器较基础：Euler + 少步数，精度/稳定性仍有优化空间。
- 语言模板与 token 化仍是风险源：虽然已修复，但需要持续 regression guard。

## 6. 优先级最高的改进方向（建议先做）

P0（先做，预计 ROI 最大）：

1. 细粒度 2D-3D 融合：从“全局池化广播”升级到“token-to-token/region-to-region 对齐融合”。
2. 可学习层路由：对多层 VLM 表示做 gating 或 MoE-style 加权，而不是固定取 last/first。
3. 推理积分器增强：比较 Euler / Heun / DPM-Solver 风格离散，在相同步数下提升动作质量。

P1（中优先）：

1. 时间一致性约束：对预测动作轨迹加 smoothness/jerk 正则与任务边界约束。
2. 状态-视觉一致性建模：引入 state-conditioned modulation，减少仅靠语言/视觉的模糊性。
3. 指令稳健训练：同义改写一致性损失 + 难负样本（反义、干扰）对比训练。

P2（后续）：

1. 数据混合采样策略优化（任务难度分层、失败轨迹重采样）。
2. 自蒸馏或 teacher-free consistency 训练，提升低步推理稳定性。

## 7. 推荐的实验验证顺序（最短闭环）

1. 先固定当前设置，记录统一基线（eval_libero + sensitivity 三元组：baseline/synonym/opposite）。
2. 只替换“融合模块”，其余不动，跑短程训练验证增量。
3. 在最佳融合版本上再替换“层路由”。
4. 最后做“推理积分器/flow solver”替换和步数-性能折中。

## 8. 继续开发时应长期监控的核心指标

- 任务成功率：LIBERO 套件 success rate（主指标）
- 动作误差：`action_dit_loss`、eval rollout 级别动作稳定性
- 指令敏感性：`opposite_l2`、`synonym_l2`、`image_l2`
- 稳定性指标：训练 NaN/Inf 次数、梯度范数异常率
- 代价指标：吞吐、显存、每步推理延迟

## 9. 快速结论
你的系统已经通过“输入变化 -> 动作显著变化”的关键门槛。  
下一步最值得投入的是“更细粒度的 2D-3D 融合 + 可学习层路由 + 更强 flow 求解器”，这三项最可能带来可复现、可量化的算法增益。

