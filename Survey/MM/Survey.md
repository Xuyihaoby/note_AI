### Pretrain

#### BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

整体的与训练架构如图所示：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220154259922.png" alt="image-20260220154259922" style="zoom:80%;" />

使用到了图文对比损失函数，图像匹配损失函数以及语言生成的损失函数，为了高效训练，文本的encoder与decoder是共享参数的；作者又设计了额外的流程来提高数据集的质量：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220154512802.png" alt="image-20260220154512802" style="zoom:67%;" />

caption generator和caption filter都是从相同的上述预训练模型初始化的，并在 COCO 数据集上单独进行微调。caption generator是一个基于图像的文本解码器。它通过 LM 目标进行微调，以解码给定图像的文本。给定网络图像 $I_w$，caption generator生成合成文本 $T_s$，每个图像一个文本。caption filter是一个基于图像的文本编码器。它根据 ITC 和 ITM 目标进行了微调，以了解文本是否与图像匹配。caption filter会去除原始网络文本 $T_w$ 和合成文本 $T_s$ 中的噪声文本，其中如果 ITM 头预测文本与图像不匹配，则文本被认为是噪声文本。

#### BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

本篇工作相较于第一代就开始像主流的多模态结构上靠拢了，视觉模型与大语言模型都冻住，并且只使用两个transformer模块Q-former（这两个模块共享参数）进行训练。

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220155054341.png" alt="image-20260220155054341" style="zoom:80%;" />

训练的时候分为两阶段：图像-语言表征阶段，只需要使用到视觉编码器与Q-former模块，像BLIP一样进行学习分为图文对比，图文匹配与图文生成三个loss，这三个任务需要将mask进行对应处理。第二步再加上语言模型，进行进一步语言能力的提升；

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220155251732.png" alt="image-20260220155251732" style="zoom: 80%;" />

#### CoCa: Contrastive Captioners are Image-Text Foundation Models

整体架构如下图所示，通过结合contrastive learning和captioning loss的方式，高效地预训练图像-文本编码器-解码器模型：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220155903866.png" alt="image-20260220155903866" style="zoom:80%;" />

在解码器的前半部分，省略了交叉注意力机制，仅使用自注意力机制来编码纯文本表示；在解码器的后半部分，加入了交叉注意力机制，使其能够同时关注图像编码器的输出，从而学习图像-文本的联合表示。

为了适应不同的训练目标和下游任务，CoCa 引入了任务特定的注意力池化机制，通过一个单头注意力层，将图像编码器的输出池化为不同长度的嵌入，以满足对比学习和生成式学习的需求：对于对比学习，使用单个查询向量（[CLS] token）来池化图像嵌入。对于生成式学习，使用更长的查询序列（如 256 个查询向量）来提取更细粒度的图像特征。

#### DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment

针对LiT训练方法，冻结visual encoder微调text encoder使得视觉与文本对齐。主要改进点：1、结合[cls]，与平均池化的patch构成全局与局部特征；2、额外添加几层可训练的vision block；3、数据集的处理结合分层kmeans以及metaclip2。整体流程如下所示：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220165432272.png" alt="image-20260220165432272" style="zoom:67%;" />

#### Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks

BEIT-3 采用模块化的 Multiway Transformer，实现了单架构适配所有视觉 / 视觉 - 语言任务。

网络结构核心设计：所有模态（视觉、语言、跨模态）共享同一套自注意力层，负责学习模态内的依赖和跨模态的语义对齐，实现深度融合。每层 Transformer 配备视觉专家（V-FFN） 和语言专家（L-FFN），输入的视觉 token / 文本 token 会被路由到对应专家层，捕捉模态特异性信息；顶层 3 层额外增加视觉 - 语言专家（VL-FFN），强化跨模态融合。

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220210837932.png" alt="image-20260220210837932" style="zoom:67%;" />

通过对 Multiway Transformer 的灵活复用，BEIT-3 可无缝切换为不同编码器，适配所有下游任务：仅激活 V-FFN，用于图像分类、目标检测、语义分割等纯视觉任务；仅激活 L-FFN，用于文本相关建模；激活 VL-FFN，用于 VQA、视觉推理（NLVR2）等需要深度跨模态交互的任务；分别编码图像 / 文本，用于高效的图像 - 文本检索；序列到序列编码器：用于图像 caption 等生成式视觉 - 语言任务。

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220210902546.png" alt="image-20260220210902546" style="zoom:67%;" />

 预训练策略：统一的掩码数据建模（Masked Data Modeling）

BEIT-3 的预训练仅用一种任务—— 掩码后预测，摒弃了传统多模态模型的 “对比学习 + 匹配学习 + 掩码学习” 多任务混合模式。

预训练数据：图像 - 文本对：2100 万对（来自 CC12M、CC3M、SBU、COCO、VG）；单模态图像：1400 万张（ImageNet-21K）；单模态文本：160GB 英文语料（维基百科、BookCorpus、OpenWebText 等）。

掩码规则（差异化掩码，适配不同模态特性）：纯文本：随机掩码 15% 的 token；图像 - 文本对中的文本：随机掩码 50% 的 token；图像：采用块掩码策略随机掩码 40% 的 patch（BEITv2 的视觉 tokenizer 将图像编码为离散视觉 token，作为重建目标）。

### MLLM

#### DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding

整体流程如下所示：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220163109358.png" alt="image-20260220163109358" style="zoom: 67%;" />

结构：vision enocder（“SigLIP-SO400M-384” ）；vision-language adaptor；Deepseek-MOE，

该论文主要是将Deepseek-v2拓展到多模态领域，使用了动态切片的策略（和其他多模态系列工作核心思想相似），先将现有图片进行resize处理，使其符合vision encoder输入的分辨率要求$C_R = \{ m \cdot 384, n \cdot 384 \mid m \in \mathbb{N}, n \in \mathbb{N}, 1 \leq m, n, mn \leq 9 \}$，之后就可以将该图片分成$m_i \times n_i$，之后将这些分块图片以及一个整体图片，输送给vision encoder，流程图如下所示：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220163244098.png" alt="image-20260220163244098" style="zoom:67%;" />

“Vision-Language Adaptor” 为简单的两层MLP，并且在这之前还使用了pixel shuffle策略。

训练分为三个阶段：VL alignment,只优化vision encoder与adaptor，数据集构建：“ShareGPT4V”；VL pretraining，全部模块参与训练；supervised fine-tuning (SFT)，只对回答部分进行训练。

#### Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling

InternVL-2.5

整体架构：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220200245667.png" alt="image-20260220200245667" style="zoom:67%;" />

训练过程框架：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220200317245.png" alt="image-20260220200317245" style="zoom:67%;" />

数据处理：

数据增强：仅对图像数据集启用 JPEG 压缩增强，视频数据集禁用，保证视频帧质量一致性；

采样与分辨率控制：通过`n_max`（最大分块数）控制图像 / 视频帧分辨率，适配不同数据集；通过`r`（重复因子，0<r≤4）调节数据集采样频率，r<1 下采样降权重，r>1 上采样增训练轮数；

训练效率优化：采用多模态数据打包策略，拼接多个样本为长序列，减少填充、提升 GPU 利用率；

严格数据过滤：针对 LLMs 对噪声高敏感特性设计双模块过滤，仅处理纯文本数据：①按领域用预训练 LLM 打 0-10 分，移除低于阈值样本；②LLM 识别重复模式并人工审核，移除低分样本；③启发式规则过滤长度异常、含超长零序列等异常文本；

分阶段数据组合：预训练，扩充领域特定数据，将非对话类数据（图像字幕、OCR 等）转为对话格式；阶段 1/1.5 仅训练 MLP/MLP+ViT，纳入高低质量混合数据，丰富模型世界知识、提升泛化性；微调，数据规模翻倍至 1630 万，覆盖通用问答、科学、医疗、代码等多领域，包含单 / 多图像、视频、文本等多模态，保障数据多样性。

#### Gemma3

多模态大模型，整体结构继承自Gemma2

decoder-only结构，GQA, postnorm与prenorm且均为RMSNorm，QK-Norm，locallayer与global layer以5：1的比例进行交错；支持128k上下文，RoPE的基由10k增加到1M；视觉编码器使用SigLIP，固定输入分辨率为896*896；Gemma 3 的 27B 模型训练了 14T tokens，12B 版本训练了 12T tokens，4B 版本训练了 4T tokens，1B 版本训练了 2T tokens。增加 token 数量是为适应预训练期间图像和文本的混合使用，同时提高了多语言数据的比例，以增强语言覆盖范围，还添加了单语和并行数据，并采用相关策略处理语言表示的不平衡问题。使用与 Gemini 2.0 相同的基于 SentencePiece 的分词器，支持数字分割、保留空格和字节级编码，最终生成的词汇表包含 262k 个条目，对非英语语言更为均衡。

### Vision Encoder

#### A Single Transformer for Scalable Vision-Language Modeling

图像不经过vision encoder直接输入给LLM，框架如下：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220113606906.png" alt="image-20260220113606906" style="zoom: 80%;" />

baseline是基于Mistral-7B-v0.1的LLM，图像处理算法：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220113723026.png" alt="image-20260220113723026" style="zoom: 80%;" />

预训练分为3个阶段：

- **第一阶段**：使用ImageNet21K进行图像分类预训练，为后续阶段打下基础。如果不加上这个阶段，第二阶段训练出来的效果不好。
- **第二阶段**：扩展到Web规模数据，包括图像-标题对和合成网页，以增强模型的多样性。
- **第三阶段**：退火处理，为指令微调阶段做准备。选择一些高质量数据让模型从带有大量噪声的网络数据转到高质量数据生成

#### ConvLLaVA: Hierarchical Backbones as Visual Encoder for Large Multimodal Models

将vision encoder常用的ViT类转换成卷积类ConvNeXt，这么做的好处可以降低计算量，并且分辨率能够下降的更多。但是convnext的优化方法和ViT也有一些出入：ConvNeXt 预训练于低分辨率（256）且训练数据质量低于 ViT，直接应用于高分辨率会导致性能下降。采用三阶段训练更新 ConvNeXt：用 558k 字幕数据初始化视觉 - 语言投影器；用 ShareGPT4V-PT 高质量数据训练整个视觉 - 语言模型（包括 ConvNeXt 的后 18 个块）；用 665k LLaVA 指令数据微调。当分辨率超过 768 时，即使是 4 阶段 ConvNeXt 也会产生过多令牌。所以新增 1 个 ConvNeXt 阶段（6 层），将特征压缩比从 32 倍提升至 64 倍。

#### DeepStack: Deeply Stacking Visual Tokens is Surprisingly Simple and Effective for LMMs

整体架构流程如下所示：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220164722489.png" alt="image-20260220164722489" style="zoom:67%;" />

### Caption

#### End-to-End Transformer Based Model for Image Captioning

transfomer提出早期，将该结构应用到image caption任务中的一次尝试。摒弃了传统的两阶段范式中以来目标检测预训练等问题。整体架构如下所示：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220193002002.png" alt="image-20260220193002002" style="zoom:80%;" />

#### Beyond a Pre-Trained Object Detector: Cross-Modal Textual and Visual Context for Image Captioning

传统 Image Captioning 模型的核心流程为「预训练目标检测器提取独立物体视觉特征→语言解码器生成描述」，存在两大关键问题：仅提取孤立物体特征，忽略图像内物体的空间关联、场景语义上下文；未利用文本模态的语义先验辅助视觉特征理解，导致生成描述易逻辑断裂、物体错配。论文提出视觉上下文模块（Visual Context Module, VCM） 和跨模态文本上下文模块（Cross-modal Textual Context Module, CTM） 构建，弱化对预训练检测器的依赖（可兼容轻量检测器甚至基础视觉特征提取器），通过双模块的特征交互与融合，输出更具语义关联性的跨模态特征，再送入解码器生成描述

针对预训练检测器输出的物体特征（含位置、类别、视觉特征向量），通过图神经网络显式建模物体间的上下文关联：将每个物体作为图节点，以空间距离+ 语义相似度构建节点间的边权重，通过 GNN 的消息传递机制，聚合相邻物体的特征，得到融合上下文的视觉特征，解决孤立物体特征的缺陷。为让文本语义先验辅助视觉特征理解，设计双向跨模态注意力机制，实现视觉特征与文本上下文的互作，具体分两步：先通过预训练语言模型（PLM） 初始化文本语义空间，挖掘图像描述的句式先验、词汇共现规律；再构建跨模态注意力层，让融合上下文的视觉特征与文本语义特征进行双向交互：视觉特征引导文本特征聚焦与图像相关的语义，文本语义特征则对视觉特征进行语义重加权，最终输出跨模态融合特征。

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220152716236.png" alt="image-20260220152716236" style="zoom:80%;" />

#### GPT4RoI: Instruction Tuning Large Language Model on Region-of-Interest

提出**空间指令微调**方法，在指令中引入对感兴趣区域（RoI）的指代。在将指令输入大语言模型之前，指代标记会被替换为感兴趣区域特征，并与语言嵌入向量交错组合为一个序列。

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220205551406.png" alt="image-20260220205551406" style="zoom:80%;" />

### OCR

#### DeepSeek-OCR: Contexts Optical Compression

虽然是OCR类的工作，但是它可以在压缩长文本信息，某种程度上能够进一步提升LLM的效率。整体架构如下所示：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220162407649.png" alt="image-20260220162407649" style="zoom:80%;" />

架构就是SAM编码器加上CLIP的串联。多分辨率支持：通过动态插值位置编码来满足上述需求，并设计了几种分辨率模，DeepEncoder主要支持两种主要输入模式：原生分辨率和动态分辨率。原生分辨率支持四种子模式：Tiny、Small、Base和Large，对应的分辨率和标记数量分别为512×52（64）、640×640（100）、1024×1024（256）和1280×1280（400）；Tiny和Small模式，图像通过直接调整原始形状进行处理。Base和Large模式，为了保持原始图像的纵横比，图像会被填充至相应尺寸。动态分辨率可以由两个原生分辨率组合而成：Gundam模式由n×640×640 tiles和一个1024×1024全局视图组成。 tiles方法遵循InternVL2.0。DeepEncoder在Gundam模式下的输出视觉标记数量为：n × 100 + 256，其中n是tiles数量。对于宽高均小于640的图像，n设为0，即Gundam模式将退化为Base模式。

#### Glyph: Scaling Context Windows via Visual-Text Compression

相较于deepseek ocr 这篇工作更加直观的说明了vl压缩带来的好处；将文本渲染成图像以此来支持更长的上下文；但是从准确率来说比deepseek ocr还是要差一点，baseline为GLM。

整体训练分为三个部分，如下所示：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220203654927.png" alt="image-20260220203654927"  />

Continual Pre-Training：训练VLM理解并推理以多种视觉风格渲染的长文本；LLM-Driven Rendering Search：自动探索并发现适用于下游任务的最佳渲染配置；Post-Training：在确定的最优配置下，通过监督微调（SFT）和强化学习（RL）进一步提升模型的长上下文能力。

持续预训练：设计了三类持续预训练任务，OCR：模型需重建一个或多个渲染页面上的全部文本；Interleaved Language Modeling：部分文本段落被渲染为图像，其余保持文本形式，训练模型在模态间无缝切换。生成任务：给定部分渲染页面（如开头或结尾），模型需补全缺失内容。

LLM驱动的渲染搜索：在持续预训练之后，执行LLM驱动的遗传搜索，以自动识别用于后训练阶段的最优渲染配置 $\theta^*$。（注：每种渲染方式由一个配置向量 $\theta$ 定义：

$$
\theta = (\text{dpi}, \text{page\_size}, \text{font\_family}, \text{font\_size}, \text{line\_height}, \text{alignment}, \text{indent}, \text{spacing}, \text{h\_scale}, \text{colors}, \text{borders}, ...)
$$
）

从一个初始候选配置种群 $\{\theta_k\}$（从预训练配置中采样）开始迭代执行以下步骤：渲染数据，使用每个配置 $\theta_k$ 渲染验证集，得到视觉输入。验证集评估，在渲染后的数据上进行模型推理，测量任务准确率和压缩比，并更新结果。LLM分析与批评，使用一个大语言模型（LLM）基于当前种群和验证结果，提出有潜力的变异（mutation）和交叉（crossover）操作。搜索历史记录，记录所有配置及其性能，对候选者进行排序和采样，用于下一轮迭代。

后训练：通过两个互补的优化阶段，SFT和RL提升 Glyph-Base 的性能，并辅以一个辅助的OCR对齐任务。构建了一个高质量的文本SFT语料库，并使用最优配置 $\theta^*$ 对其长上下文输入进行渲染。每个响应采用“思考式”格式（thinking-style），即示例中包含明确的推理过程（例如，“<think>...</think>”）。这鼓励模型在阅读海量token上下文时进行逐步推理。Reinforcement Learning，在SFT之后， 使用GRPO来进一步优化策略，辅助OCR对齐任务，在SFT和RL两个阶段中，引入一个辅助的OCR对齐任务，以增强模型从图像中识别文本内容的能力，从而更好地对齐其视觉与文本表示，

### Distillation

#### EM-KD: Distilling Efficient Multimodal Large Language Model with Unbalanced Vision Tokens

针对高效学生模型与原始教师模型间视觉 Token 数量不平衡的问题，提出了名为EM-KD；整体架构如下图所示：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220190625784.png" alt="image-20260220190625784" style="zoom: 80%;" />

为解决视觉 Token 数量不平衡 + 空间错位的核心步骤，将其转化为集合二分匹配任务：把师生的视觉 Token 通过 LM Head 解码为视觉 Logits（词汇空间，含语义）；计算师生视觉 Logits 间的曼哈顿距离，构建代价矩阵；用匈牙利算法求解代价最小的排列，为师生 Token 建立最优一对一对应关系，实现非对齐 Token 的精准匹配。采用反向 KL 散度作为损失函数，让学生模型的视觉 Logits 分布逼近教师模型；对匹配后的师生视觉 Token，分别计算其与文本 Token 的余弦相似度，构建视觉 - 语言亲和矩阵；用平滑 L1 损失让学生的亲和矩阵逼近教师，保证师生在视觉 - 语言关联上的一致性；

### Downstream Tasks

#### Beyond Bounding Box: Multimodal Knowledge Learning for Object Detection

使用languige model结合目标检测模型，构造了一个目标检测器，语言模型和目标检测模型之间的loss使用的是对比学习；利用语言模型的能力，提升目标检测器的检测性能。

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220153845539.png" alt="image-20260220153845539" style="zoom: 67%;" />

#### General Object Foundation Model for Images and Videos at Scale

GLEE 的核心目标是：构建统一输入输出范式，融合多源数据与多粒度监督，实现对任意目标的检测、分割、跟踪、定位与识别。

整体架构如下图所示：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220201049612.png" alt="image-20260220201049612" style="zoom:67%;" />

基于MaskDINO构建目标解码器，并设计一个动态类别头。该动态类别头通过计算检测器输出的目标嵌入与文本编码器输出的文本嵌入之间的相似度，实现对目标类别的判定。
