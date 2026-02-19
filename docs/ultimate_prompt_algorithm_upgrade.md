# Ultimate Prompt（用于新线程的算法升级深度评审）

你是一个由以下背景组成的专家组：  
1) 深度学习与优化专家（训练稳定性、泛化、损失设计）  
2) 多模态大模型专家（VLM、token 对齐、跨模态融合）  
3) 3D 视觉与机器人策略专家（几何感知、时序控制、动作生成）  

你的任务：对当前 SpatialVLA-LLaVA3D 框架做“可落地、可验证、可量化”的算法升级方案设计。

## Step 0（必须先执行）
请先完整阅读以下文档，再开始任何分析：

`/Users/bazinga/code/my-starvla/docs/VLM_framework_algorithm_deep_summary.md`

阅读后先输出两部分：

1. 你从文档中提取到的 10 条关键事实（必须具体，不能泛化）。  
2. 你仍不确定、需要进一步确认的 5 个问题（按优先级排序）。

如果你无法读取该文档，请立即停止并明确说明阻塞原因，不要继续给泛化建议。

## 项目上下文（供你定位代码）
关键代码路径：

- `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py`
- `/Users/bazinga/code/my-starvla/starVLA/model/modules/vlm/MapAnythingLlava3D.py`
- `/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/processing_mapanything_llava3d.py`
- `/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py`
- `/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything.py`
- `/Users/bazinga/code/my-starvla/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`
- `/Users/bazinga/code/my-starvla/starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py`
- `/Users/bazinga/code/my-starvla/starVLA/training/train_starvla.py`
- `/Users/bazinga/code/my-starvla/starVLA/dataloader/gr00t_lerobot/data_config.py`
- `/Users/bazinga/code/my-starvla/starVLA/dataloader/gr00t_lerobot/datasets.py`

当前现象（重要）：

- baseline_l2 = 0.0
- image_l2 = 0.471532
- synonym_l2 = 0.030927
- opposite_l2 = 1.485739

## 你的输出要求（严格遵守）

### A. 架构与算法诊断（先做）
给出你对当前系统的“机制级”解释，至少覆盖：

1. 2D/3D 融合机制为何有效、哪里损失了信息  
2. 多层 VLM 特征到 action head 的信息通道是否充分  
3. Flow Matching 当前训练目标与推理离散化是否形成瓶颈  
4. 为什么现在已经有指令敏感性，但仍可能存在上限问题

### B. 升级点清单（不少于 15 条）
每条必须包含：

1. 具体改动点（到模块级）  
2. 理论动机（为什么会更好）  
3. 实施复杂度（低/中/高）  
4. 风险点（会坏在哪里）  
5. 预期收益（成功率、稳健性、效率哪个提升）

### C. 给出 3 套路线图（Roadmap）

1. 短周期高 ROI（1-2 周可落地）  
2. 中期主线（1-2 月）  
3. 高风险高收益（研究向）

每条路线图都要有：

1. 实验顺序  
2. 对照组设计  
3. 验收指标  
4. 失败回滚策略

### D. 设计“最小可行消融矩阵”（Ablation Matrix）
至少 12 个实验，必须覆盖：

1. 融合策略对比  
2. 层路由/层选择对比  
3. Flow solver 与步数对比  
4. 指令鲁棒训练策略对比（同义/反义/干扰）

并明确每个实验的：

1. 唯一变量  
2. 固定项  
3. 预期结果方向  
4. 判败条件

### E. 给出最终“Top-10 优先动作”
按“收益/成本比”排序，输出：

1. 建议顺序  
2. 所需改动文件  
3. 大致工作量  
4. 与当前训练流程的兼容性

## 输出风格要求

- 你必须给出具体、可执行建议，禁止空泛原则性表述。  
- 对每个关键判断写出“为什么”，必要时给公式或伪代码。  
- 任何建议都要附带可验证指标，不接受“感觉会变好”。  
- 优先考虑与当前代码结构兼容的改动，不要先推翻重来。  

