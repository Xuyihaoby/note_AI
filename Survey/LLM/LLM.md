#### mHC: Manifold-Constrained Hyper-Connections

mHC 的核心思想是对残差映射施加结构化约束，将其限制在双随机矩阵构成的流形上。双随机矩阵具有行和与列和均为 1 的性质，使得残差映射在幅值上具备天然的稳定性。更重要的是，该集合在矩阵乘法下保持封闭，这意味着即使经过多层复合，整体残差传播仍然处于受控范围内，从而显著缓解深层网络中的数值不稳定问题。在此基础上，mHC 还对输入映射和输出映射施加非负性约束，以保证信息流的可解释性和数值一致性。所有映射仍然由当前层的隐藏状态动态生成，从而保留 HC 原有的自适应特性。当然，deepseek出品，必须得有算子和pipeline上的优化。

#### oai gpt-oss model card

模型采用维度为2880的残差流，并在注意力块和混合专家（MoE）块之前使用RMSNorm，整体为Pre-LN结构。MoE模块由多个专家和线性路由层组成，每个token仅选择得分最高的4个专家进行加权计算，专家内部采用门控SwiGLU。注意力层在带宽为128的窗口注意力与全密集注意力之间交替，并使用分组查询注意力（64个查询头、8个键值头），结合旋转位置编码与YaRN将上下文长度扩展至131k，同时在softmax中引入可学习偏置以缓解注意力退化问题。训练全程使用基于BPE的o200k_harmony分词器，词表规模约20万。

#### DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models

提出DSA（Dynamic Sparse Attention）整体结构包含三个核心组件：

Lightning Indexer：为每个查询 Token 生成轻量级查询向量，用于快速估计查询与历史 Token 之间的相关性。多头索引分数经加权聚合后，形成近似的注意力相关性度量。Top-k 选择器：根据索引分数筛选 Top-k 个最相关的键值对，构成稀疏上下文集合，从而避免在全量序列上执行高成本注意力计算。核心注意力层：仅在筛选后的稀疏上下文上执行注意力计算，生成最终上下文表示。

密集预热阶段：冻结主模型，仅训练Lightning Indexer。以完整注意力分布作为监督信号，通过 KL 散度使Lightning Indexer预测的相关性分布逼近真实注意力分布。稀疏训练阶段：启用 Top-k 选择机制并解冻全模型参数，索引器的训练仅基于筛选后的稀疏集合进行KL优化，而主模型通过语言建模目标进行优化，两者的计算图处于分离状态，进而实现优化目标解耦。

在对齐与强化学习阶段，引入规模化 GRPO。同时，采用无偏 KL 估计以修正由旧策略采样带来的分布偏移。对优势为负且策略偏差较大的序列进行掩码以降低离策略噪声；在训练阶段保持与推理阶段一致的专家路由路径；并复用采样时的 Top-k / Top-p 截断掩码，确保新旧策略在相同动作子空间内进行比较。

#### Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models

EnGRAM 通过将 N-gram 与哈希检索结合，实现了一种算存分离的上下文记忆机制，在表示构建上，EnGRAM 并不直接对 N-gram 原始 tokenizer 输出的 token ID 进行建模，而是通过一个基于文本规范化（如 NFKC、小写化等）的词汇投影函数，将等价 token 折叠到共享的规范标识符空间中，从而提升 N-gram 的语义密度并减少冗余。基于该压缩后的 token 序列，模型构造后缀 N-gram 作为检索单元。

由于直接参数化全部 N-gram 组合在规模上不可行，EnGRAM 采用多头哈希机制：对每个 N-gram 阶数使用多个独立哈希头，将 N-gram 映射到固定大小的稀疏嵌入表中。不同阶数与哈希头对应的嵌入向量被拼接，形成位置相关的记忆表示，从而在参数受限的条件下覆盖大规模上下文模式。

在交互方式上，EnGRAM 使用当前隐藏状态作为动态查询，对检索到的记忆进行上下文感知门控。记忆向量经线性投影后作为键和值，通过归一化内积计算门控系数，对值向量进行调制，以确保梯度稳定并避免过度放大。随后，引入一个短程的因果卷积模块以扩大感受野并增强非线性建模能力，最终结果通过残差方式注入主干网络。

在多分支骨干结构中，EnGRAM 进一步扩展为共享存储、分支特定检索的形式：所有分支共享同一稀疏嵌入表和价值投影，而每个分支拥有独立的键投影矩阵，从而使不同残差流能够以不同方式访问同一记忆内容。

#### DAPO: An Open-Source LLM Reinforcement Learning System at Scale

DAPO 的优化目标定义为：


$$
\mathcal{J}_{\text{DAPO}}(\theta)=
\mathbb{E}_{(q,a)\sim\mathcal{D},{o_i}_{i=1}^G\sim\pi_{\theta_{\text{old}}}(\cdot|q)}
\left[
\frac{1}{\sum_{i=1}^G |o_i|}
\sum_{i=1}^G \sum_{t=1}^{|o_i|}
\min!\left(
r_{i,t}(\theta)\hat A_{i,t},
\operatorname{clip}!\left(r_{i,t}(\theta),1-\varepsilon_{\text{low}},1+\varepsilon_{\text{high}}\right)\hat A_{i,t}
\right)
\right],\\
\text{s.t.}\quad
0<\left|{o_i\mid \text{is}_{\text{equivalent}}(a,o_i)}\right|<G
$$


DAPO 相较于朴素 PPO / GRPO 主要包含以下改进：

Clip-Higher: 为缓解训练过程中的熵坍缩问题，引入不对称裁剪策略，解耦 $\varepsilon_{\text{low}}$ 与 $\varepsilon_{\text{high}}$。通过增大 $\varepsilon_{\text{high}}$，为低概率 token 的概率提升提供更大的探索空间。

动态采样（Dynamic Sampling）: 通过过采样并过滤准确率为 0 或 1 的 prompt，仅保留具有有效梯度的样本，在保持 batch 中 prompt 数量恒定的同时，提高训练效率与稳定性。

Token-Level Policy Gradient Loss: 针对长思维链场景中 prompt 级损失无法有效约束局部不合理生成的问题，引入 token 级策略梯度损失，以实现更细粒度的优化。

Overlong Reward Shaping: 为避免截断样本带来的奖励噪声，首先屏蔽超长截断样本的损失以稳定训练；进一步提出软超长惩罚机制，根据响应长度施加连续惩罚：


$$
R_{\text{length}}(y)=
\begin{cases}
0, & |y|\le L_{\max}-L_{\text{cache}},\\
\dfrac{(L_{\max}-L_{\text{cache}})-|y|}{L_{\text{cache}}}, & L_{\max}-L_{\text{cache}}<|y|\le L_{\max},\\
-1, & |y|>L_{\max}.
\end{cases}
$$


其中，$|y|$ 表示响应长度，$L_{\max}$ 为最大生成长度，$L_{\text{cache}}$ 为软惩罚区间长度。



