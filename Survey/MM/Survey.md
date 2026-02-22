### Pretrain

#### Learning Transferable Visual Models From Natural Language Supervision

CLIP

#### MetaCLIP 2: A Worldwide Scaling Recipe

目前CLIP训练主要还是使用英文的训练数据，一使用多语种，性能就会产生下降；这篇文章中，使用互联网上所有的语料进行训练。实证表明，CLIP中的"多语诅咒"是由于缺乏合适的数据清洗和模型训练的方法而导致的。

训练方法基于metaclip，并且模型架构与openai的clip同样尽可能相似，本文的主要创新点如下：将英文 MetaCLIP 的元数据扩展到维基百科和多语言 WordNet 上的 300 多种语言。构建了逐语言的子串匹配和平衡机制，以便为非英文数据筛选出与英文数据相似的概念分布；设计了首个全球范围的 CLIP 训练框架，包括在训练过程中按新增非英文数据量成比例增加已见图像 - 文本对，以及针对从全球规模数据中学习所需的最小可行模型容量展开研究。

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

 预训练策略：统一的掩码数据建模（Masked Data Modeling）；BEIT-3 的预训练仅用一种任务—— 掩码后预测，摒弃了传统多模态模型的 “对比学习 + 匹配学习 + 掩码学习” 多任务混合模式。

预训练数据：图像 - 文本对：2100 万对（来自 CC12M、CC3M、SBU、COCO、VG）；单模态图像：1400 万张（ImageNet-21K）；单模态文本：160GB 英文语料（维基百科、BookCorpus、OpenWebText 等）。

掩码规则（差异化掩码，适配不同模态特性）：纯文本：随机掩码 15% 的 token；图像 - 文本对中的文本：随机掩码 50% 的 token；图像：采用块掩码策略随机掩码 40% 的 patch。

#### ImageBind: One Embedding Space To Bind Them All

以图像为 “枢纽”，通过对比学习，将文本、音频、深度、热成像、IMU 这 5 种模态的特征，全部 “投射” 到同一个高维向量空间中。

#### LiT: Zero-Shot Transfer with Locked-image text Tuning

传统的多模态模型（如 CLIP, ALIGN）通常是从头开始同时训练图像编码器和文本编码器。而 LiT 提出了一种更高效的策略：利用已经预训练好的强大图像模型，仅训练文本模型来适配该图像模型。

### MLLM

#### Improved Baselines with Visual Instruction Tuning

LLaVA-1.5在结构上将视觉语言连接部分从原本的线性投影层改为两层mlp；数据层面：加入学术任务导向的 VQA 数据，并设计标准化的响应格式提示词，设计明确的格式提示词，解决输出失衡问题；原版 LLaVA 用 224px 分辨率，LLaVA-1.5 提升至CLIP-ViT-L-336px；模型缩放：将基础 LLM 从 Vicuna-7B 升级为Vicuna-13B；数据缩放：加入 ShareGPT 语言对话数据，提升模型的自然语言理解能力；同时对训练数据做轻量化处理（如合并同图 QA 为单对话、截断长对话、按模态采样批次），训练速度提升 25%，且不影响最终性能。

#### LLaVA-OneVision: Easy Visual Task Transfer

方法架构 (Modeling)：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260222165621646.png" alt="image-20260222165621646" style="zoom:80%;" />

主要由三部分组成：视觉编码器，使用 SigLIP；大语言模型，选用 Qwen-2 系列（0.5B, 7B, 72B）；Projector，使用一个简单的 2层 MLP，将视觉特征映射到语言模型的词嵌入空间。

视觉表示 (Visual Representations) 为了平衡性能和计算成本，并支持跨场景迁移，提出了 Higher AnyRes 策略：

对于宽度为 $a$、高度为 $b$ 的 AnyRes 配置，它将图像划分为 $a \times b$ 个裁剪块（crops），每个裁剪块的形状为 $(a, b)$。每个裁剪块都具有适合视觉编码器的相同分辨率。假设每个裁剪块有 $T$ 个 token，则视觉 token 的总数为 $L = (a \times b + 1) \times T$，考虑一个阈值 $\tau$，如果需要，会使用双线性插值来减少每个裁剪块的 token 数量：


$$
T_{new} = \begin{cases} \frac{\tau}{a \times b + 1} & \text{if } L > \tau \\ T & \text{if } L \le \tau \end{cases}
$$


定义了一组空间配置 $(a, b)$ 以指定各种图像裁剪方法，从而适应不同分辨率和纵横比的图像：单图，采用高分辨率裁剪策略，分配大量视觉 token（模拟视频的多帧表示），以便更好地迁移到视频理解；多图，仅使用基础分辨率，节省计算资源；视频，将每帧调整为基分辨率，并使用双线性插值减少每帧的 token 数量，从而允许输入更多帧数。

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

数据增强：仅对图像数据集启用 JPEG 压缩增强，视频数据集禁用，保证视频帧质量一致性；

采样与分辨率控制：通过`n_max`（最大分块数）控制图像 / 视频帧分辨率，适配不同数据集；通过`r`（重复因子，0<r≤4）调节数据集采样频率，r<1 下采样降权重，r>1 上采样增训练轮数；训练效率优化：采用多模态数据打包策略，拼接多个样本为长序列，减少填充、提升 GPU 利用率；

严格数据过滤：针对 LLMs 对噪声高敏感特性设计双模块过滤，仅处理纯文本数据：按领域用预训练 LLM 打 0-10 分，移除低于阈值样本；LLM 识别重复模式并人工审核，移除低分样本；③启发式规则过滤长度异常、含超长零序列等异常文本；

分阶段数据组合：预训练，扩充领域特定数据，将非对话类数据（图像字幕、OCR 等）转为对话格式；阶段 1/1.5 仅训练 MLP/MLP+ViT，纳入高低质量混合数据，丰富模型世界知识、提升泛化性；微调，数据规模翻倍至 1630 万，覆盖通用问答、科学、医疗、代码等多领域，包含单 / 多图像、视频、文本等多模态，保障数据多样性。

#### InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models

大多数现有的 MLLM通常是先训练一个纯文本的大语言模型（LLM），然后再通过多阶段流程强行“插入”视觉能力。这种方法容易导致模态间的对齐困难，且可能损害原有的语言能力。InternVL3采用原生多模态预训练。模型在单一的预训练阶段，就同时接触大量的纯文本语料和多模态数据。

可变视觉位置编码 (Variable Visual Position Encoding, V2PE)为了支持更长的多模态上下文（例如高分辨率图像或多帧视频），同时不过度消耗位置编码的窗口。传统的编码中，每个 token的位置索引都增加 1。V2PE 为视觉 token 使用一个较小的增量 $\delta$ ($\delta < 1$)，而为文本 token 保持增量 1。

在预训练之后，InternVL3 采用了两阶段的后训练来进一步提升对话和推理能力：监督微调 (SFT)，使用了比前代更高质量、更多样化的数据，数据量从 1630 万增加到 2170 万条。混合偏好优化 (Mixed Preference Optimization, MPO)：为了解决训练和推理之间的分布差异，引入了 MPO。它结合了偏好损失（DPO）、质量损失（BCO）和生成损失，利用正负样本对来对齐模型的响应分布，显著增强了链式思维（CoT）推理能力。

测试时扩展 (Test-Time Scaling)：引入了 Best-of-N 策略，配合专门训练的 VisualPRM作为裁判。 在推理阶段，模型生成多个候选答案，VisualPRM 对每一步推理进行打分，最终选择得分最高的答案。这一策略在数学和复杂推理任务上带来了显著的性能提升，即使是小参数模型也能受益。

#### InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency

InternVL 3.5 相比前代在以下三个点上进行创新：级联强化学习 (Cascade RL)：采用“由粗到细”的两阶段训练策略。离线 RL (Offline RL)：使用混合偏好优化 (MPO) 进行高效预热，确保模型收敛稳定并提供高质量的 rollout（采样数据）。在线 RL (Online RL)：使用 GSPO 算法基于模型自生成的数据进行精细化对齐，进一步推高性能上限。视觉分辨率路由器 (Visual Resolution Router, ViR)：动态调整每个图像块（patch）的视觉 token 压缩率。模型会根据图像内容的语义丰富度，自动选择是将图像块压缩为较少的 token（低分辨率）还是保留较多 token（高分辨率）。解耦的视觉 - 语言部署 (Decoupled Vision-Language Deployment, DvD)：将视觉编码器（ViT）和语言模型（LLM）部署在不同的 GPU 服务器上。视觉部分并行处理图像生成特征，然后异步传输给语言部分。

模型架构：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260222153727028.png" alt="image-20260222153727028" style="zoom:80%;" />

训练过程分为三个主要阶段：预训练 (Pre-training)，使用约 250B token 的多模态和纯文本数据进行联合训练，支持 32K 长上下文。监督微调 (SFT)，使用高质量指令数据、思维链（Thinking mode）推理数据以及新能力数据（如 GUI 交互、具身智能、SVG 生成）进行微调。后训练 (Post-training)，应用 Cascade RL 提升推理。应用 ViCO 训练 ViR 模块以构建 Flash 版本。

#### Mini-InternVL: A Flexible-Transfer Pocket Multimodal Model with 5% Parameters and 90% Performance

较为小型的多模态模型，具体结构如下图所示：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260222200619400.png" alt="image-20260222200619400" style="zoom:80%;" />

#### InternLM-XComposer2: Mastering Free-form Text-Image Composition and Comprehension in Vision-Language Large Model

基于InternLM的多模态大模型，主要创新点在于使用了partial lora将图片信息与语言信息对齐，并且训练时使用了更高质量，更加多样的数据。

视觉编码器：OpenAI ViT-Large；Large Language Model ：InternLM2-7B-ChatSFT

Partial Low-Rank Adaptation (蓝色部分是视觉token，灰色部分是语言token)：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260221154450992.png" alt="image-20260221154450992" style="zoom:80%;" />

公式表示：


$$
\begin{align*}
\hat{x}_{t} &= W_{0}x_{t} + B_{0} \\
\hat{x}_{v} &= W_{0}x_{v} + W_{B}W_{\Lambda}x_{v} + B_{0} \\
\hat{x} &= [\hat{x}_{v}, \hat{x}_{t}]
\end{align*}
$$


其中，对于每个线性层 $L_0$ 在大型语言模型（LLM）块中，表示其权重矩阵 $W_0∈R^{C_{out}×C_{in}}$ 和偏置 $B_0∈R^{Cout}$，其中 $C_{in}$ 和 $C_{out}$ 是输入和输出维度。其对应的部分低秩适配（Partial LoRA）包含两个低秩矩阵 $W_Λ∈R^{C_r×C_{in}}$ 和 $W_B∈R^{C_{out}×C_r}$。

#### InternLM-XComposer2-4KHD: A Pioneering Large Vision-Language Model Handling Resolutions from 336 Pixels to 4K HD

为了能够让大模型处理更高的分辨率，加入了动态分辨率的概念：给定一个最分块数 $H$，尺寸为 $[h,w]$ 的图像 $x$ 被调整大小并填充到新图像 $\hat{x}$，其尺寸为 $[p_h×336,p_w×336]$。这个过程受到以下约束：


$$
p_w \times p_h \leq \mathcal{H}; \quad p_h = \left\lceil \frac{p_w \times h}{w} \right\rceil \qquad (1)
$$


这里，$p_w$ 和 $p_h$ 分别代表每行和每列的块数。然后将 $\hat{x}$ 分割成 $p_h×p_w$ 个不重叠的块。每个块是一个尺寸为 $336×336$ 的小图像，将这些块作为ViT的单独输入。需要注意的是，这里的 $H$ 是最大分区数，$h$ 和 $w$ 是原始图像的高和宽，$p_h$ 和 $p_w$ 是调整后图像的高和宽的块数。公式中的 $⌈⋅⌉$ 表示向上取整。使用多尺度的技术，除了将图片分块之外，还额外将整张图片尺寸缩小成336x336，并且进行分块处理。

整体过程如下图所示：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260221154233915.png" alt="image-20260221154233915" style="zoom:80%;" />

为了保证长宽比，每一行patch后面会加’\n‘而不同尺度后面会加’separate’

#### InternLM-XComposer-2.5: A Versatile Large Vision Language Model Supporting Long-Contextual Input and Output

整体架构如下所示：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260221152017564.png" alt="image-20260221152017564" style="zoom: 80%;" />

图像输入分辨率：560560，图像分片策略为： 给定最大分区数 \(\mathcal{H}\)，尺寸为 \([h, w]\) 的图像 \(x\) 会被调整大小并填充为新图像 \(\hat{x}\)，其尺寸为 \([p_h \times 560, p_w \times 560]\)。这一过程需满足以下约束：


$$
\begin{align} p{w1} &= \left\lfloor p{w1} \times \frac{h}{w} \right\rfloor \leq \mathcal{H}, \tag{1} \\ p{w2} &= \left\lfloor w \times \frac{s}{560} \right\rfloor, \tag{2} \\ p_w &= \min(p{w1}, p_{w2}); \quad p_h = \left\lceil p_w \times \frac{h}{w} \right\rceil \tag{3} \end{align}
$$


其中 \(s\) 是缩放因子，\(p_w\) 和 \(p_h\) 分别表示每行和每列的图像块数量。对于多图像输入，为每张图像分配一个索引 \( i \in \{1, 2, 3, \dots\} \)，并以交错格式将图像和文本进行格式化。 从给定视频中采样帧，并沿帧的短边将它们拼接，从而得到一张高分辨率图像。图像索引也会写入图像中，以提供时间关系。 音频处理部分采用 Whisper将音频转录为文本；对于音频输出，利用 MeloTTS将文本转换回音频。在post training中使用DPO来提升训练质量

#### JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation

将多模态理解与文生图任务通过自回归模型统一起来。整体架构如下所示（在生成模型中，encoder与decoder使用long skip connection进行连接，并且省略了SDXL-VAE）：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260221160442763.png" alt="image-20260221160442763" style="zoom: 67%;" />

在图像编码阶段并没有使用同一个encoder进行编码，多模态理解的图像编码pre-trained SigLIP-Large-Patch/16，$g_{enc}$与$g_{dec}$为ConvNext模块。生成编码器包含一个2×2的patchify层、两个ConvNeXt块和一个线性层；生成解码器包含两个ConvNeXt块、一个pixel-shuffle层和一个线性层。

训练阶段如下所示：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260221160753661.png" alt="image-20260221160753661" style="zoom: 67%;" />

除了常规的配置自回归与生成的loss之外，还引入了REPA，进行进一步正则。

#### Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation

在统一多模态理解与图像生成的基础上，将图像理解的编码器与图像生成的编码器进行解耦：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260221155646245.png" alt="image-20260221155646245" style="zoom:80%;" />

图像理解的编码器使用的是SigLIP，图像生成的编码器使用的是VQ tokenizer，训练的几个阶段如下图所示：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260221155727162.png" alt="image-20260221155727162" style="zoom:80%;" />

#### Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling

结构相较于Janus并没有明显改动，使用SigLIP编码器从图像中提取高维语义特征，将这些特征从2D网格展平为1D序列，并通过（understanding adaptor）映射到LLM的输入空间。使用VQ tokenizer将图像转换为离散ID，将ID序列展平为1D后，通过（generation adaptor）将每个ID对应的codebook嵌入映射到LLM的输入空间。

增加Stage I的训练步数，使模型在ImageNet数据上充分训练。增加了约9000万样本，包括图像字幕数据集（如YFCC）以及表格、图表和文档理解数据（如Docmatix）。在Stage II中，直接使用正常文本到图像数据训练模型，放弃ImageNet数据。增加了更多数据集，如MEME理解、中文对话数据以及提升对话体验的数据集。Janus-Pro通过将模型规模从1.5B扩展到7B

#### Gemma3

多模态大模型，整体结构继承自Gemma2。decoder-only结构，GQA, postnorm与prenorm且均为RMSNorm，QK-Norm，locallayer与global layer以5：1的比例进行交错；支持128k上下文，RoPE的基由10k增加到1M；视觉编码器使用SigLIP，固定输入分辨率为896*896；Gemma 3 的 27B 模型训练了 14T tokens，12B 版本训练了 12T tokens，4B 版本训练了 4T tokens，1B 版本训练了 2T tokens。增加 token 数量是为适应预训练期间图像和文本的混合使用，同时提高了多语言数据的比例，以增强语言覆盖范围，还添加了单语和并行数据，并采用相关策略处理语言表示的不平衡问题。使用与 Gemini 2.0 相同的基于 SentencePiece 的分词器，支持数字分割、保留空格和字节级编码，最终生成的词汇表包含 262k 个条目，对非英语语言更为均衡。

#### Kimi-VL Technical Report

整体架构如下所示：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260221162542192.png" alt="image-20260221162542192" style="zoom:80%;" />

Kimi-VL 由视觉编码器（MoonViT）、MLP投影和MoE语言模型三部分组成，具体如下：处理多分辨率图像，采用NaViT打包方法将图像划分为小块并转化为一维序列；基于SigLIP-SO-400M预训练权重（可学习位置编码，支持高分辨率插值），引入二维旋转位置嵌入（RoPE）提升细粒度位置表示；连接MoonViT与语言模型，通过2×2（pixel shuffle）对图像特征下采样，再经两层MLP将特征投影至语言模型嵌入维度；基于Moonlight预训练中间检查点初始化（已处理5.2T纯文本标记，上下文长度8K），后续经2.3T多模态+纯文本数据联合预训练。采用增强版Muon优化器，新增权重衰减并调整参数更新规模；结合ZeRO-1优化策略实现分布式部署，优化内存效率、降低通信开销，用于训练模型所有参数。

预训练：1 基于图像-文本对训练，文本含alt文本、合成标题等，训练目标为SigLIP损失+交叉熵损失；用SigLIP SO-400M初始化编码器，采用渐进式分辨率采样，经2T标记训练后，再用0.1T标记对齐MoonViT与MoE模型（仅更新前两者）。2 结合纯文本与多模态数据训练，基于现有LLM检查点续训，额外消耗1.4T标记；渐进增加多模态数据比例，兼顾语言能力与视觉理解能力。3 采用高质量语言+多模态数据训练，语言部分纳入合成QA对（提升推理、代码等能力），多模态部分将学术数据源转化为QA对；控制QA对比例避免过拟合。4 将上下文长度从8K扩展至128K，重置RoPE频率至800,000；分两子阶段（每阶段上下文长度扩4倍），各子阶段长数据占比25%、短数据占比75%。

后训练阶段：采用ChatML格式，优化语言模型、投影器与视觉编码器；分两阶段训练（32K→128K序列长度）。基于高质量长CoT预热数据集（含文本/图像推理路径），进行轻量级SFT使模型内化多模态推理策略。采用在线策略镜像下降变体算法，优化目标含奖励项与KL正则化；引入长度奖励惩罚冗余回答，结合课程采样、优先采样提升训练效率。

#### KIMI-VL TECHNICAL REPORT

Keye-VL 专为短视频理解而设计，同时在通用的视觉 - 语言任务上也保持了强大的能力。

数据策略 (Data)：构建了超过 6000 亿 token 的大规模数据集。数据处理：采用了严格的过滤、重标注（re-captioning）和帧级标注流程。去污染：在训练前后进行了严格的数据去重和去污染，避免基准测试数据泄露。

模型架构 (Architecture)：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260221165626268.png" alt="image-20260221165626268" style="zoom:80%;" />

基于 Qwen3-8B 作为语言解码器，使用 SigLIP 作为视觉编码器。原生分辨率 (Native-Resolution)：视觉编码器支持原生动态分辨率，保留图像的原始宽高比，避免复杂的切割操作。引入了 2D RoPE (Rotary Position Embedding) 增强视觉信息的建模，并使用 3D RoPE 统一处理文本、图像和视频的时间位置信息，确保对视频时序变化的精确感知。为图像预留了充足的 token 缓冲（最多 16384 个），视频则采用动态分辨率策略，平衡帧数和总 token 数。

预训练 (Pre-Training) - 四阶段渐进式策略：Stage 0: 视觉编码器继续预训练（适应内部数据分布）。Stage 1: 跨模态对齐（冻结视觉和语言模型，训练投影层）。Stage 2: 多任务预训练（解锁所有参数，涵盖 OCR、定位、VQA 等）。Stage 3: 退火与模型合并（Annealing & Model Merging），通过混合不同数据配比训练的模型权重来减少偏差，增强鲁棒性。

后训练 (Post-Training)：阶段一，非推理训练 (No-Reasoning Training)，通过监督微调 (SFT) 和混合偏好优化 (MPO) 建立基础指令遵循能力。阶段二，冷启动 (Cold-Start)，使用包含五种模式的混合数据（“思考”、“非思考”、“自动思考”、“带图思考/Agent”、“高质量视频”）教会模型何时以及如何思考。强化学习 (RL)：使用 GRPO 算法进一步增强复杂推理能力，特别是针对短视频理解进行了专门的 RL 优化。迭代对齐：纠正重复输出和逻辑错误，提升用户体验。

#### MiMo-VL Technical Report

小米提出的多模态模型，整体结构如下所示：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260222200048261.png" alt="image-20260222200048261" style="zoom:80%;" />

post-training部分采用创新的混合在线强化学习（MORL）框架，将可验证奖励强化学习（RLVR）与人类反馈强化学习（RLHF）无缝融合，在提升模型解决高难度推理任务能力的同时，实现模型输出与人类偏好的对齐。

#### MiniCPM-V: A GPT-4V Level MLLM on Your Phone

模型架构 (Model Architecture)

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260222212544962.png" alt="image-20260222212544962" style="zoom:80%;" />

整体结构： 由视觉编码器（SigLIP）、压缩层（Perceiver Resampler）和大语言模型（LLM，基于 Llama3-Instruct 8B）组成。

自适应视觉编码 (Adaptive Visual Encoding)：图像切片 (Image Partition)， 为了处理高分辨率和任意长宽比的图像，模型将图像切分成多个小块（Slices），每个小块的大小适配视觉编码器的预训练设置。切片编码与压缩，每个切片编码后产生的 token 数量较多，通过压缩层将每个切片的 token 压缩为固定数量（如 96 个），大大减少了输入给 LLM 的视觉 token 总数，从而降低显存占用并提高推理速度。空间模式 (Spatial Schema)，引入特殊标记来指示每个切片在原图中的位置，保留空间信息。

训练策略 (Training Recipes)

训练分为三个阶段：预训练 (Pre-training)：阶段 1： 预热压缩层，连接视觉编码器和 LLM。阶段 2： 扩展视觉编码器的输入分辨率（从 224 扩展到 448）。阶段 3： 引入自适应视觉编码策略，并加入 OCR 数据以增强文字识别能力。数据清洗： 使用辅助模型对低质量的网络图文对进行重写（Caption Rewriting），并采用数据打包（Data Packing）策略提高训练效率。

监督微调 (SFT)：使用高质量的视觉问答（VQA）和多模态指令数据进行微调。数据分为两部分：Part-1 侧重基础识别能力，Part-2 侧重复杂交互和长文本生成。多语言泛化： 利用多语言 LLM 作为枢纽，仅用少量多语言数据即可将多模态能力泛化到 30+ 种语言。RLAIF-V (基于 AI 反馈的强化学习)：为了解决“幻觉”问题，模型生成多个回答，利用开源强模型（如 LLaVA-NeXT-Yi 34B）对回答中的原子事实进行打分。基于这些评分构建偏好数据集，使用 DPO（直接偏好优化）算法对模型进行对齐，显著降低了幻觉率。

端侧部署优化 (End-side Deployment)：量化 (Quantization)： 使用 4-bit 量化（Q4_K_M），将显存需求从 16GB+ 降至约 5GB。内存优化： 采用顺序加载策略（先加载 ViT 处理图像，释放后再加载 LLM），避免内存溢出。编译优化： 针对目标设备的指令集进行编译，显著提升推理速度。配置优化： 自动搜索最佳的后端配置参数。NPU 加速： 利用手机自带的 NPU（神经网络处理器）加速视觉编码部分，进一步降低延迟。


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

#### DeepSeek-OCR 2: Visual Causal Flow

从结构上来看，将v1版本的CLIP结构换成了LM结构作为新的视觉编码器：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260221110547724.png" alt="image-20260221110547724" style="zoom: 80%;" />

输入策略：因果查询令牌的数量与视觉令牌的数量相等，计算公式为$\frac{W ×H}{16^{2} ×16}$，其中$W$和$H$分别表示输入至编码器的图像宽度和高度。采用multi-crop策略，在预定义分辨率下使用固定的查询配置。全局视图采用1024×1024的分辨率，对应256个查询嵌入，记为$query_{global}$；局部裁剪采用768×768的分辨率，裁剪数量$k$的取值范围为0至6（当图像的长和宽均小于768时，不执行裁剪操作）。输入至大语言模型（LLM）的重排序视觉令牌总数为$k ×144+256$，取值范围为256至1120。

从实验数据上来说的话其实和专用OCR模型还是有差别的，专用模型的效果优于通用模型

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

#### LLaVA-KD: A Framework of Distilling Multimodal Large Language Models

三个阶段的蒸馏方法，如下图所示：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260222163500968.png" alt="image-20260222163500968" style="zoom:80%;" />

文章提出了两种互补的蒸馏方法，分别作用于输出分布和视觉特征的关系建模。

响应蒸馏损失 ($\mathcal{L}_{res}$)： 使用  (KLD) 最小化教师模型 ($l$) 和学生模型 ($s$) 在生成响应 token 时的概率分布差异；视觉蒸馏损失 ($\mathcal{L}_{vis}$)： 同样使用 KLD 最小化视觉 token 输出分布的差异；关系蒸馏 (Relation Distillation, RDist)：过构建视觉 token 的自相关矩阵来传递结构知识。

 DPT 阶段损失：在预训练阶段，除了标准的自回归预测损失 ($\mathcal{L}_{PT}$)，还加入了上述三种蒸馏损失；DFT 阶段损失：在微调阶段，总损失包含正则化损失 ($\mathcal{L}{reg}$，即标准的 SFT 损失) 和蒸馏损失。

#### MiniLLM: Knowledge Distillation of Large Language Models

传统的KD方法通常最小化前向KL散度（Forward KLD,$ KL[p∣∣q]$）。在开放式的文本生成任务中，教师模型的分布非常复杂（多模态），而学生模型容量有限。最小化前向KL散度会迫使学生在教师分布的低概率区域（“空洞”）也分配高概率，导致生成质量下降、出现幻觉或不合理的样本。

![image-20260222213927408](https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260222213927408.png)

为了解决上述问题，提出了 MiniLLM 方法，其核心创新点包括

反向KL散度（Reverse KLD）：

将优化目标从最小化前向KL散度改为最小化反向KL散度（$KL[q∣∣p]$ ）。优势： 反向KL散度具有“模式搜索（mode-seeking）”特性，它鼓励学生模型专注于教师分布的主要模式（高概率区域），而忽略低概率的长尾噪声。这使得生成的文本更准确、更可靠，避免了覆盖教师分布中的“空洞”。

在线策略优化（On-Policy Optimization）：由于反向KL散度的梯度无法直接通过标准的监督学习获得，作者利用策略梯度（Policy Gradient）定理推导了梯度公式，采用在线训练方式（即采样学生模型的输出来计算梯度）。

三大优化策略： 为了稳定训练并解决策略梯度常见的问题（如高方差、奖励黑客、长度偏差），作者引入了三种策略：

单步分解（Single-Step Decomposition）, 将梯度分解，直接计算单步生成质量的期望，减少方差，加速收敛。教师混合采样（Teacher-Mixed Sampling）,在采样时混合教师和学生的分布，防止学生模型通过生成退化文本（如重复短语）来欺骗奖励函数（Reward Hacking）。长度归一化（Length Normalization）, 消除对短序列的偏好，防止模型生成空响应。

### Downstream Tasks

#### Beyond Bounding Box: Multimodal Knowledge Learning for Object Detection

使用languige model结合目标检测模型，构造了一个目标检测器，语言模型和目标检测模型之间的loss使用的是对比学习；利用语言模型的能力，提升目标检测器的检测性能。

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220153845539.png" alt="image-20260220153845539" style="zoom: 67%;" />

#### General Object Foundation Model for Images and Videos at Scale

GLEE 的核心目标是：构建统一输入输出范式，融合多源数据与多粒度监督，实现对任意目标的检测、分割、跟踪、定位与识别。

整体架构如下图所示：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260220201049612.png" alt="image-20260220201049612" style="zoom:67%;" />

基于MaskDINO构建目标解码器，并设计一个动态类别头。该动态类别头通过计算检测器输出的目标嵌入与文本编码器输出的文本嵌入之间的相似度，实现对目标类别的判定。

#### MDETR -- Modulated Detection for End-to-End Multi-Modal Understanding

同样是transformer引入多模态领域早期所带来的工作。

MDETR 基于DETR框架进行了扩展：

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260222190918360.png" alt="image-20260222190918360" style="zoom:80%;" />

<img src="https://note-ai-survey.oss-cn-hangzhou.aliyuncs.com/image-20260222190900293.png" alt="image-20260222190900293" style="zoom:80%;" />

MDETR 不使用传统的分类标签（如“猫”、“狗”），而是直接预测文本中指向该物体的词 span。主要包含两个关键损失函数：

软 Token 预测损失 (Soft Token Prediction Loss)：对于每个匹配到的物体，模型被训练去预测原文本中指向该物体的所有 token 位置的分布（类似DFL）；未匹配到物体的查询则预测“无物体”（no object, $\emptyset$）。

文本 - 查询对比对齐损失 (Text-Query Contrastive Alignment Loss)：基于 InfoNCE 损失，强制使物体的视觉嵌入与其对应的文本 token 嵌入在特征空间中更接近，而与不相关的 token 更远。这增强了图文对齐的鲁棒性。

预训练数据：构建了一个包含130 万对齐图文对的大规模数据集。数据来源包括 Flickr30k、MS COCO 和 Visual Genome。



