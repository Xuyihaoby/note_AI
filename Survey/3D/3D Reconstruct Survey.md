# 3D Reconstruct Survey

## Nerf

### Method

#### NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

利用MLP对物体进行神经辐射场建模，输入几张照片后以及观察角度利用几层MLP结合体渲染公式就能重建出整个场景，nerf类工作一个很明显的缺点就是输出结果是隐式的，无法直接提取mesh。nerf在训练的时候会有两个模型一个coarse（沿着每条射线均匀采样 N个点），一个fine（根据粗网络输出的权重 $w_i=T_i α_i$ 建一个新的分布，利用重要性采样再从这个分布中采样 N'个点）。

在这里给出最重要的体渲染公式为：

$$
\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \alpha_i \mathbf{c}_i, \quad \text{其中} \quad T_i = \exp\left( -\sum_{j=1}^{i-1} \sigma_j \delta_j \right), \quad \alpha_i = 1 - \exp(-\sigma_i \delta_i),\quad δ_i=t_{i+1}−t_i
$$

#### NeRF++: Analyzing and Improving Neural Radiance Fields

分析了辐射场与颜色之间的歧义性，就算是模型预测的错误密度，也能因为后续预测的错误color达到负负得正的效果，nerf的分阶段输入以及小容量MLP一定程度上缓解了这一点，并且着重去解决无边界场景问题，近景照常处理，远景使用**逆球面参数化**的方式

#### Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields

通过把原版的射线改成视锥的方式，并不单纯采样视线上具体的某一个点，而是一段圆锥台内的所有点去表征，解决nerf只能应用在单尺度，单分辨率，渲染图像很容易模糊或者出现锯齿

#### Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields

针对nerf在进行无边界场景的重建困难：利用种基于视差（与距离成反比）的采样机制，将无边界的3D场景参数化为有边界的场景；使用在线蒸馏的思想，coarse mlp只需要预测密度权重，并且利用所提出的损失函数加以限制$L_{prop}$进行约束，使其预测尽量接近nerfMLP；将weight约束成一种单峰分布的形式，以缓解floater的问题

#### Volume Rendering of Neural Implicit Surfaces

想要将隐式神经场显式化（想要进行表面重建）改进密度的表示方法，并不依靠神经辐射场直接输出密度信息，而是预测符号距离场并通过laplace累积分布的形式给出密度信息；优化采样方式，在采样点个数与laplace分布的$\beta$中寻求平衡。

#### NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction

推导出了一个全新的、基于SDF的体渲染公式。设计了一个权重函数 `w(t)`，该函数满足 **“无偏性”** 特性：当采样无限密集时，权重函数的峰值会**收敛**到真实的物体表面。这避免了VolSDF中可能存在的偏差，使得重建的表面更加精确和细节丰富。

#### Plenoxels: Radiance Fields without Neural Networks

一种不需要额外神经网络参数的nerf，和后续的gaussian splatting有很多异曲同工的地方，该方法认为场景一个稀疏的体素网格，每个体素网格的格点存储着不透明度与每个颜色通道的球谐函数sigma。

#### Differentiable Point-Based Radiance Fields for Efficient View Synthesis

nerf开始转向splatting的一种中间形态。

#### TensoRF: Tensorial Radiance Fields

将场景看成4D张量，张量中的每一项代表一个体素，体素内包含一个密度以及特征向量（用于代表颜色信息）并使用张量分解的形式，将张量拆分为Matrix与vectors的外积和。通过这种方式降低模型的大小同时，提升模型的训练效率。

#### Strivec: Sparse Tri-Vector Radiance Fields

基于TensorRF，但是TensorRF以及之前的其他方法总是将注意力平等的散布在空间中。本文首先使用DVGO获取粗略的几何分布信息，再使用该信息利用CP分解进行局部场景表示。

#### InstantNGP

将空间划分为多个分辨率，高分辨率下使用hash映射的方式每个顶点的可学习向量（2维）映射到对应的内存中，由于hash表长度会比点的个数少，可能会存在一些隐式的冲突，但是由于本身空间中物体的分布就不是均匀的，所以这种隐式的冲突并不会影响到最终的效果。

#### UniVoxel: Fast Inverse Rendering by Unified Voxelization of Scene Representation

提出一种统一体素化的快速逆渲染方法，核心是将场景表示统一体素化为辐射场体素网格，结合神经辐射场（NeRF）与体素哈希技术加速逆渲染过程。方法通过可微渲染器将输入图像与预测的体素辐射场进行对齐，利用体素哈希结构动态分配计算资源，仅在高细节区域密集采样，低细节区域稀疏表示，从而减少内存占用和计算量。训练时采用多尺度体素更新策略，从粗到细优化体素网格的密度和辐射值，同时引入相机姿态优化模块同步调整位姿参数。该方法无需显式 3D 重建，直接在体素空间中通过反向传播优化场景表示，实现对输入图像的快速逆渲染，适用于动态场景和实时交互应用。

#### MobileNeRF: Exploiting the Polygon Rasterization Pipeline for Efficient Neural Field Rendering on Mobile Architectures

这项工作旨在解决 NeRF 渲染速度慢、难以与传统渲染管线结合的问题。核心思路是将神经辐射场与传统的多边形渲染、Z-buffer 和延迟着色技术相结合，以实现高效的并行化渲染。首先将场景表示为一个可渲染的多边形网格，并在每个面片上存储特征而非颜色。在渲染时，采用两阶段的延迟渲染流程：第一阶段使用 GPU 光栅化生成特征图像（即延迟缓冲区），记录每个像素的几何特征和视角信息；第二阶段在片段着色器中运行一个小型 MLP，将这些特征映射为最终的像素颜色。为简化深度排序并提升效率，不透明度被离散化为二值（0/1），从而避免半透明物体的处理复杂性。

整个训练过程分为三个阶段：首先在连续表示下联合优化网格顶点和 MLP 参数，以重建颜色并抑制漂浮伪影；随后使用直通估计器将不透明度离散化并稳定训练，实现从体渲染向显式表面过渡；最后仅保留可见面片，将特征和离散不透明度烘焙到纹理图中，得到一个可直接由传统 GPU 渲染流水线加速的“特征纹理网格”。

### Framework

#### Nerfstudio: A Modular Framework for Neural Radiance Field Development

工程上开源了nerf相关的framework，造福开源社区。

### Style transfer

#### ARF: Artistic Radiance Fields

给定特定风格图片对3D物体进行风格迁移（对NerF基础上进行微调）：提出NNFM loss避免GRAM类loss只关注全局信息忽略局部高频信息；延迟渲染的技术，先关闭梯度计算整幅图片loss，之后再进行分块渲染。

### Large Scale

#### BungeeNeRF: Progressive Neural Radiance Field for Extreme Multi-scale Scene Rendering

通过逐级加模块的方式训练多尺度（不同高度），进行大规模场景重建。

### Mesh extraction

#### Delicate Textured Mesh Recovery from NeRF via Adaptive Surface Refinement

从nerf中提取mesh，分为两个部分：基于Instant-NGP训练一个NeRF模型，并将外观颜色解耦为漫反射（与视角无关）和高光（与视角相关）两部分；使用Marching Cubes从训练好的密度场中提取一个粗糙网格。通过可微渲染，联合优化网格顶点位置和已解耦的外观模型，以最小化2D渲染误差。根据面的渲染误差动态调整网格面密度：对高误差区域进行细分以增加细节，对低误差区域进行简化以提高效率。将优化后的漫反射和高光特征“烘焙”成标准纹理贴图，并利用集成到着色器中的小型MLP实现高光效果的实时渲染（mobile nerf中的技术）。

### 4D

#### KFD-NeRF: Rethinking Dynamic NeRF with Kalman Filter

 4D 辐射场建模，将卡尔曼滤波器插件和tri-plane表示融入NeRF，采用逐渐释放时间信息以促进学习动态系统的新颖策略进行训练；并且规范空间中的正则化，用于增强浅层观测 MLP 的学习能力。

## Gaussian Splatting

### Method

#### EWA volume splatting

首次在splating中引入显示的高斯核函数（gaussian splatting在非深度学习时代的前身），并利用高斯函数的封闭性进行计算，将世界坐标系转为相机坐标系，再将相机坐标系转为射线空间坐标系（使用泰勒展开进行线性化处理），得到2D平面与3D空间gaussian的协方差对应关系：


$$
\mathbf{\Sigma}_k = \mathbf{J} \mathbf{\Sigma}_k' \mathbf{J}^T = \mathbf{J} \mathbf{W} \mathbf{\Sigma}_k'' \mathbf{W}^T \mathbf{J}^T
$$


#### Surface splatting

EWA splatting的进一步改进，或者是更接近3DGS的前身；图像渲染的做法是将uvmap与3D空间的位置结合，之后再将其映射到2D屏幕空间去。本文认为没有必要再3D空间中找纹理函数，可以直接再物体空间中存储相关的颜色信息。即每个点云由一个基函数与三个通道的权重系数去表示颜色

#### 3D Gaussian Splatting for Real-Time Radiance Field Rendering

该方法完全抛弃了nerf的神经辐射场对整个场景的隐式建模，使用splatting技术来显示建模整个场景（在笔者看来这是图形学splatting与深度学习技术非常震撼的一次结合，同时从coding的角度来说这也是极为出色的工程项目）：提出了各向异性的3d高斯分布作为非结构化的辐射场表示方法；提出了3d高斯分布的优化方法-利用自适应密度控制算法去创建高质量的表示；针对GPU进行可微优化，并且将图片分割成16x16的tile利用cuda并行处理。

每100个iter去除alpha小于阈值的高斯，在自适应密度控制过程中，如果高斯的位置产生较大的梯度，说明当前高斯并不能够很好地拟合当前场景，如果是高斯尺度不够大的情况，就额外克隆高斯将其沿着梯度的方向分布，如果高斯尺度过大，就分裂高斯，尺度因子除以1.6。为了减少漂浮物的影响，每训练3000个iter，就会将所有高斯分布的透明度置零，让他们重新进行训练

优化过程中，不透明度是经过sigmoid确保输出在0-1。scale的输出是经过exp激活，3D高斯分布初始化的时候使用邻近三个点的均值进行各向同性的初始化，程序运行过程中，会将图片划分成16x16个小块tile，方便充分利用GPU的性能进行并行处理，每一个tile都会维持一个列表，具体哪些高斯splatiing与他们相交，并且使用64位数对相关参数进行编码，低32位用于表示深度，高32位用于表示tile的ID。对于光栅化，作者为每个图块tile启动一个线程块block。每个块首先协作地将高斯数据包加载到共享内存中，然后对于给定的像素，通过从前到后遍历列表来累积颜色和 α 值，从而最大化数据加载/共享和处理的并行性增益。当像素中的目标饱和度 α大于阈值时，相应的线程就会停止，不再进行向后累积。需要注意的，GS的前提是需要进行colmap，从而得到相机的内外参数，以及对应的稀疏点云

#### 2D Gaussian Splatting for Geometrically Accurate Radiance Fields

针对3DGS三个问题：3D高斯体使用的是体积辐射表示，但实际的表面通常是薄的，难以准确捕捉细节；3D高斯体本身无法原生支持表面法线；3D高斯体的光栅化过程中，不同的视角会导致不同的二维相交平面，这种多视角不一致性会影响重建效果。

2DGS是将每个点及其周围的区域看成扁平的椭圆盘，中点表示为 $p_k$  ,两个方向的切向量 $t_{u}, t_{v}$ ,尺度向量为 $S = (s_u, s_v)$ 用于控制协方差。首先由两个相互正交的切向量 $t_u, t_v$ 构造法线 $t_w = t_u \times t_v$，进而得到旋转矩阵 $R=[t_u, t_v, t_w]$，并结合尺度矩阵 $S=\text{diag}(s_u, s_v, 0)$ ，构建从局部 $uv$ 平面到三维空间的齐次变换：

$$
P(u, v) = p_k + s_u t_u u + s_v t_v v
 = H(u, v, 1, 1)^\top, \quad\\
H=\begin{bmatrix}
s_u t_u & s_v t_v & 0 & p_k \\
0 & 0 & 0 & 1
\end{bmatrix} = \begin{bmatrix}
RS & p_k \\
0 & 1
\end{bmatrix}
$$

在 $uv$ 平面上，splat 的权重通过二维高斯核建模：

$$
\mathcal{G}(\mathbf{u}) = \exp\left(-\tfrac{u^2+v^2}{2}\right).
$$

将 $P(u,v)$ 投影到屏幕坐标系 $\mathbf{x}=(x,y,z,1)^\top$ 得：

$$
\mathbf{x} = W P(u,v) = W H (u,v,1,1)^\top.
$$

传统方法需计算 $(WH)^{-1}$ 来获得反变换，但在学习过程中数值不稳定。提出显式的射线–splat 相交方法：光线通过图像坐标 $(x,y)$ 定义为两个平面 $\mathbf{h}_x, \mathbf{h}_y$，其在 $uv$ 坐标中的约束为：

$$
\mathbf{h}_u=(WH)^\top \mathbf{h}_x, \quad
 \mathbf{h}_v=(WH)^\top \mathbf{h}_y,
$$

并联立解得 $u(x), v(x)$ 的闭式形式。

考虑到 splat 在某些视角下会退化为直线，引入低通滤波器修正：
$$
\hat{\mathcal{G}}(\mathbf{x}) =
 \max\{\mathcal{G}(\mathbf{u}(\mathbf{x})),
 \mathcal{G}\bigl((\mathbf{x}-\mathbf{c})/\sigma\bigr)\}
$$

颜色合成仍采用前向 $\alpha$-混合：

$$
\mathbf{c}(\mathbf{x}) =
 \sum_i c_i \alpha_i \hat{\mathcal{G}}_i(\mathbf{u}(\mathbf{x}))
 \prod_{j<i} (1-\alpha_j \hat{\mathcal{G}}_j(\mathbf{u}(\mathbf{x}))).
$$

#### 2DGH: 2D Gaussian-Hermite Splatting for High-quality Rendering and Better Geometry Reconstruction

在2DGS的基础上利用Gaussian-Hermite取代了原先的高斯函数，在笔者看来其实该函数就是一种透明度的球谐系数函数；通过引入高阶埃尔米特项，基元函数函数获得了**不对称性和更复杂的轮廓**。

#### Mip-Splatting: Alias-free 3D Gaussian Splatting

核心为在进行高斯核采样的时候限制其频率，使其满足奈奎斯特定理。

结合图像分辨率、相机焦距（ $ f $ ）与场景深度（ $ d $ ），世界空间采样间隔为 $ \hat{f} = \frac{d}{f} $，采样频率为其倒数。

以 3D 高斯 \(p_k\) 中心近似深度（忽略遮挡），结合多相机视野可见性（指示函数 $ 1_n(p_k) $ ），确定 3D 高斯 $ k $ 的最大采样率：

$$
v_k = \max \left( \{ 1_n(p_k) \cdot \frac{f_n}{d_n} \}_{n=1}^{N} \right)
$$

（ $ N$ 为图像总数，$ f_n $、$ d_n $ 为第 $ n $ 个相机参数）。

基于此频率，对每个3D高斯进行低通滤波，进行3D平滑，过滤高频3D高斯表示。两个高斯卷积等于协方差之和：

$$
\mathcal{G}_k(x)_{\text{reg}} = \sqrt{\frac{|\Sigma_k|}{\left|\Sigma_k + \frac{s}{v_k} I\right|}} \ e^{-\frac{1}{2}(x - \mu_k)^T \left(\Sigma_k + \frac{s}{v_k} I\right)^{-1}(x - \mu_k)}
$$

（ $ s $  为尺度超参，$ \Sigma_k $ 为 3D 高斯协方差矩阵，$ \mu_k $ 为高斯中心位置）。

颜色的计算为抗混叠采用区域积分，通过 2D 高斯核实现：

$$
\mathcal{G}_k^{2D}(x)_{\text{min}} = \sqrt{\frac{|\Sigma_k^{2D}|}{\left|\Sigma_k^{2D} + sI\right|}} \ e^{-\frac{1}{2}(x - \mu_k^{2D})^T \left(\Sigma_k^{2D} + sI\right)^{-1}(x - \mu_k^{2D})}
$$

（ $ \Sigma_k^{2D} $  为投影后 2D 协方差矩阵，$ \mu_k^{2D} $ 为投影后的 2D 中心位置）

#### Sort-free Gaussian Splatting via Weighted Sum Rendering

在渲染过程中通过使用加权的方法，去替换对GS进行排序

#### Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering

3D-GS为适配所有训练视图，会过度扩展高斯球，忽略场景底层几何结构，造成大量高斯冗余；图相关效果被固化到单个高斯参数中，插值能力弱，对大幅视图变化和光照效果的鲁棒性不足。

Scaffold-GS 提出了一种基于锚点的分层、区域感知 3D 场景表示方法，从SfM得到的点云构建稀疏体素网格，将体素中心作为锚点，每个锚点配备局部上下文特征、缩放因子和可学习偏移量，并通过多分辨率特征融合与视图相关权重，生成集成锚点特征，提升视图适应性。在视图锥体内，每个可见锚点生成 k 个神经高斯，其位置由锚点位置与可学习偏移量、缩放因子计算得出；不透明度、颜色、旋转四元数、缩放等属性，则通过特定 MLP，基于集成锚点特征、相机与锚点的相对距离和视图方向实时预测，通过不透明度阈值过滤冗余高斯，保证渲染效率；生长操作通过将空间量化为多分辨率体素，对梯度超过阈值的体素添加新锚点，填补纹理缺失或观测不足区域；修剪操作则累计的不透明度，移除无法生成足够不透明度高斯的锚点。

#### MoE-GS: Mixture of Experts for Dynamic Gaussian Splatting

将Mixture-of-Experts（MoE）结构引入Gaussian Splatting（GS）旨在提升动态场景的渲染能力，本质上是一种将MoE与Splatting相结合的尝试。现有的Pixel Router仅在像素空间进行路由，而Volume Router则考虑了高斯的体积属性。为兼顾二者的优势，提出了Volume-aware Pixel Router，通过为每个高斯引入可学习的体积权重以编码时间和视角变化，并在光栅化后经轻量网络优化得到门控权重，用于融合不同专家的输出。该设计有效提升了渲染保真度，但带来额外的计算成本。为此，进一步提出单次多专家渲染策略以消除多次光栅化的冗余，并通过门控感知的高斯剪枝机制根据梯度贡献度去除冗余高斯，从而获得更紧凑的表示。训练阶段则采用基于蒸馏的专家优化，以MoE渲染结果作为伪真值并以门控权重为置信度，引导各专家独立学习，实现更高效且一致的动态场景建模。

#### 3D Gaussian Splatting as Markov Chain Monte Carlo

本文将高斯的放置与优化过程统一解释为一种采样机制，通过引入基于随机梯度朗之万动力学（SGLD）的更新方式，优化过程被视为从一个关于渲染质量的隐式分布中进行采样。与普通的随机梯度下降不同，SGLD 在梯度更新中加入了受控噪声，用于在参数空间中进行探索。为了避免无效扰动，噪声仅施加在高斯的位置上，并根据协方差、不透明度和学习率自适应调整，从而在保持稳定训练的同时提升对初始化噪声和几何缺失的鲁棒性。

在此框架下，原本的 densify 与 prune 策略被重新解释为马尔可夫链中的确定性状态跃迁：每个高斯从旧状态转移到新状态时保持相同的概率密度，即视为在相同概率的样本间跳转。低不透明度的“死亡”高斯被传送至“活跃”高斯所在区域，通过概率采样确定目标位置。训练中还结合对不透明度与协方差的正则化，以促使冗余高斯自然衰减、有效高斯自发密集，实现了资源的自适应分配。

### Large Scene

#### A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets

大范围场景下，3D高斯重建，使用分块加树形结构分层的思想进行处理。

层次化高斯建模：将大场景划分为多个块，并在每个块内为所有高斯基元构建一棵层次化树（如BVH）。树中的叶节点是原始优化的高斯，中间节点则通过自底向上的方式对其子节点进行合并而来。合并过程依据父节点对整张图像的色彩贡献应等于其所有子节点贡献之和。。

自适应渲染与平滑过渡：渲染时，根据视图和目标粒度，在树中自动选择一个最优的节点集合（称为一个 **Cut**）进行渲染。当视角变化导致所需细节层次改变时，系统通过在父节点与子节点属性（位置、颜色、旋转、缩放等）之间进行平滑插值

联合优化与结构压缩：整个层次结构（包括中间节点）是可优化的。训练时，通过随机采样不同的目标粒度，迫使网络联合优化所有层次的表示，确保从粗糙到精细的每一层都能产生高质量的渲染结果。训练后，对树结构进*剪枝压缩，移除那些在多种粒度下都无需被选中的冗余中间节点。

#### CityGaussian: Real-time High-quality Large-Scale Scene Rendering with Gaussians

大规模场景重建，具体的做法：将无边界的点集转换成有边界的点集；确定对应的相机姿态下这个模块是否要参与训练；利用Level-of-Detail加速渲染，参照lightgaussian，不同的高度使用不同的压缩率

注意LoD块的选择需要结合MAD来排除floater带来的影响

#### CityGaussianV2: Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes

相较于第一代的工作，这一代是基于2DGS进行改进，引入depth anything v2进行额外的loss监督；在空旷或弱纹理区域，2D高斯倾向于坍缩成一个点，导致其协方差矩阵被用于抗锯齿的低通滤波器所取代。使得这些高斯无法通过正常路径学习几何，梯度持续累积引发高斯数量的失控性增长直至显存耗尽。在执行分裂操作前，检查高斯的长宽比。当其低于阈值时（表明高斯已过度拉长或坍缩），禁止对其进行分裂。

训练初期重建效果差，传统的光度损失对场景结构不敏感，难以提供可靠的优化指导。引入了以结构相似性损失的梯度为主导的密度自适应标准：

$$
\nabla_{densify}=\max\left(\omega\times\frac{|\nabla\mathcal{L}|_{avg}}{|\nabla\mathcal{L}_{\mathrm{SSIM}}|_{avg}},1\right)\times\nabla\mathcal{L}_{\mathrm{SSIM}}
$$

通过比较平均梯度幅度，动态放大SSIM损失的作用，使其在密度自适应中占据主导地位。

除此之外，由于不像第一代有后处理等剪枝操作以及LOD，所以作者在每个block并行的时候计算每个Gaussian的贡献程度。贡献度低于阈值时会被自动排除。

### Sparse View

#### CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis

稀疏视图下，COLMAP提供的初始点云过于稀疏，尤其在单视图区域几乎为空。对于多视图区域：利用密集对应关系（如MASt3R预测）填补现有点云的空隙。对于单视图区域：利用单目深度估计生成点云，并学习一个各向异性缩放变换，将这些点精准对齐到由多视图几何定义的全局坐标系中。

稀疏视图下，不同区域的不确定性差异巨大，传统均一监督方式会导致高不确定性区域（单视图）重建失败。生成协同可视性映射：该映射记录了每个像素在训练视图中被看到的次数，直接量化了区域的不确定性。

近邻损失：训练一个MLP分类器来区分“场景内”和“场景外”的3D点。将此损失与协同可视性映射结合，实现自适应加权监督：在低共视区域，施加更强的几何约束，迫使高斯贴合到估计的表面上。在高共视区域，弱化此约束，信任多视图的光度一致性。视锥体外监督：对于整体共视度高的场景，将监督信号扩展到当前视锥体之外，提供了场景级别的几何约束，进一步稳定训练。

#### NexusGS: Sparse View Synthesis with Epipolar Depth Priors in 3D Gaussian Splatting

稀疏视图重建类的工作：通过点云致密化策略，将极线深度先验融入3DGS，提升深度准确性和渲染质量。

在点云致密化的过程中，需要借助光流和相机参数，在几线几何框架下建立极线与深度的联系，以获取准确深度。

#### SparseGS-W: Sparse-View 3D Gaussian Splatting in the Wild with Generative Priors

利用几何先验和受限扩散先验，从稀疏的野外图像中重建复杂室外场景

### Mesh Extract

#### SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering

从gaussian splatting中准确地提取出mesh网格，为此作者进行以下改进：1、设计了一个关键的正则化项，最小化3D空间点到由高斯分布定义的最近表面的距离（此时3D高斯近似于一个薄片）；2、区别于传统的Marching Cubes方法，本文使用基于泊松重建的高效网格提取流程。该方法利用前述正则化训练得到的高斯模型，在深度图视线上的3σ区域内进行采样。3、为了对提取的网格进行精细化后处理，建立网格与高斯薄片之间的绑定关系。

#### GaussianCube: A Structured and Explicit Radiance Representation for 3D Generative Modeling

将3D高斯表示（3DGS）转换为结构化体素的方法，便于输入到3D UNet中进行扩散模型训练。该方法主要包括两个核心步骤：高斯数量控制与结构化表示。为了固定高斯数量，采用“Densification-constrained fitting”策略。通过梯度阈值检测需要增密的高斯，从中选取梯度最大的部分进行克隆或拆分，同时交替剔除不透明度低的高斯，防止高斯的数量超过阈值。最终通过填充零alpha的高斯，将总数固定为32768，并省略视角相关的球谐系数，以控制表示复杂度。利用最优传输（OT）方法将高斯基元结构化为体素网格，保持其相对空间关系。高斯中心与对应体素之间的偏移量被用作位置编码。在条件注入方面，类别标签通过AdaGN注入，图像条件借鉴了大词汇3D扩散模型中Transformer的处理方式，文本条件则沿用Stable Diffusion的做法。

扩散模型基于ADM-UNet架构，将二维操作扩展为三维，使用AdamW优化器和指数移动平均（EMA）进行训练。损失函数结合了简单的L2重建损失和基于VGG特征的多分辨率图像感知损失，以提升生成质量。通过边界框归一化方法将高斯位置映射到颜色空间，完成从3D高斯到规整体素表示的转换。

### Accelerate

#### LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS

减小gaussian splatting的存储消耗，增加渲染的速度。计算重要度是根据该高斯在整幅图中被命中的次数，之后利用得到的得分按照比例删除相对应的高斯，剔除完成之后，进行微调且不在增加高斯点。球谐函数的系数，占总参数的81.3%，将它的阶数降低并进行蒸馏；在蒸馏的过程中还通过高斯函数引入额外的视角来扩充学生gs的学习；进一步对球谐系数进行了量化，得到维度为K的代码本：利用预计算的显著性分数，旨在平衡渲染质量损失和压缩率。选择性地对球谐系数中最不重要的元素应用向量量化（VQ）。

## Feed Forward

#### DepthSplat: Connecting Gaussian Splatting and Depth

将深度估计与 3D 高斯喷射 (GS) 结合，通过预训练单目深度特征构建多视图深度模型。多视角特征匹配：视图经 ResNet34 初步提取特征，再经 SwinTransformer 进一步提取，结合相机位姿与最近两视图进行 cross attention 交互，得到特征图后按 MVSPlat 方式生成 cost volume。

单目特征提取：用 Depth Anything v2 提取深度特征（参与训练），经 resize 得到对应尺度特征。

融合单目深度特征与 cost volume，经 2D UNet 处理后，沿深度维度归一化并 softmax，与对应深度加权和得到实际深度输出。对输出进行两倍上采样，结合相关信息进行 Depth Anything v2 深度预测，再通过 UNet 预测高斯参数。

#### GRM: Large Gaussian Reconstruction Model for Efficient 3D Reconstruction and Generation

feed-forward稀疏视图输入的GS工作；核心是 “像素对齐高斯” 设计：高斯位置由相机中心、射线方向和模型预测值共同确定，再逆映射转换为世界坐标系下的 3D 坐标。通过 sigmoid函数结合插值方式，将模型输出的尺度值约束在设定的最小和最大尺度范围内。

4 张输入图先与 Plücker 嵌入拼接，经步长为 16 的卷积压缩分辨率，再加入可学习的位置嵌入，得到固定形状的初始特征。把多视图的特征拉成一个序列，用自注意力机制实现不同视图间的信息交互。为解决分辨率压缩导致的高频信息丢失，先通过线性层扩大特征维度，再用 PixelShuffle 提升分辨率，最后结合窗口注意力与移位操作，对局部特征做进一步处理。损失函数为常规的L2损失作用在图像于不透明度mask上。

#### GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting

稀疏图片场景GS重建，核心思想和meshlrm类似，以单像素对应单高斯替代 triplane 策略。先将图片像素与射线 6 维 Plücker 坐标拼接并 patchify 处理成 token，再经 Transformer 网络处理，最终解码预测 RGB、scale 等高斯参数，并计算 scale、opacity 等。损失函数采用 MSE 结合感知损失，分 256 和 512 分辨率两阶段训练。物体级重建归一化物体至 [-1, 1]，场景级以平均相机姿态定世界坐标系并缩放相机位置。渲染用延迟反向传播 ARF 节省内存，但存在显存占用高、依赖相机位姿参数的缺陷。

#### Instant3D: Fast Text-to-3D with Sparse-View Generation and Large Reconstruction Model

feedforward nerf工作（结合生成模型多视角）

1、微调 SDXL 生成 2x2 多视角图像，推理时结合 SDEdit 思想，用含白色背景和中心黑色高斯斑点的灰度图（公式 I (x,y)）引导生成干净背景

2、以预训练 DINO 为图像编码器，通过 AdaLN 注入相机信息，输出姿态感知图像 token 并拼接；image-to-triplane 解码器通过交叉注意力连接 triplane 与图像 token，经处理生成 triplane 表示，解码器解码颜色和密度信息；损失函数为 MSE+LPIPS

#### LRM: Large Reconstruction Model for Single Image to 3D

单张图生成feed-forward以nerf的方式生成3D模型：图像编码器用DINO，输出全部特征 $\{h_i\}_{i=1}^n$ 而非仅[cls] token； 图像到三平面解码器：相机特征$c$由16维外参、焦距和主点组成，经相似变换归一化后，通过MLP映射为高维嵌入 $\tilde{c}$ ；解码器中通过 $\mathrm{ModLN}_{\mathrm{c}}(f_{j})=\mathrm{LN}(\boldsymbol{f}_j)\cdot(1+\gamma)+\beta$ （ $\gamma,\beta$ 来自 $\tilde{c}$ 的MLP输出）调制特征 ；三平面 $T$ 含 $T_{XY}$ 、$T_{YZ}$ 、 $T_{XZ}$ ，从可学习初始嵌入经交叉注意力、自注意力和MLP处理，最终上采样至64×64分辨率 ；3D点投影到三平面获取特征，经 $MLP^{nerf}$ 解码为颜色和密度；训练时每物体选 $V-1$ 个侧视图监督，loss为 $V$ 个视图的MSE与LPIPS损失均值。

#### InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models

feedforward类单张图片生成3D mesh的方法（结合生成模型多视角）；用zero123++生成多视角图，微调确保背景为白色；数据集基于Objaverse，筛选27万高质量物体，每物体取6视图输入、4视图监督；核心采用两阶段训练（继承Instant3D）：用LRM预权重初始化，在triplane-NeRF上训练，ViT编码器加AdaLN相机调制，移除解码器中的源相机调制层，loss含图像MSE、LPIPS和掩模损失；切换为mesh表示，集成FlexiCubes从triplane提取网格，用密度MLP预测SDF，新增MLP预测变形和权重，loss在阶段一基础上增加深度L1、法线余弦及正则化损失用mesh的。

   优势：光栅化高效，支持全分辨率监督，输出平滑且便于后续处理。为增强鲁棒性，对相机姿态随机旋转缩放，并在输入ViT前添加噪声。

#### LaRa: Efficient Large-Baseline Radiance Fields

该feedforward 2D高斯splatting方法仅需4张图即可重建360度有边界场景

DINO提取图片特征，经Plücker射线方向的AdaLN注入，反投影得3D特征 $\mathbf{V}_{\mathrm{f}}$ ；利用可学习嵌入向量 $\mathbf{V}_\mathrm{e}$ 学习先验  

基于前两者预测含K个2D高斯基元的体素的属性（如不透明度、切向量、缩放、球谐系数）被预测，其位置被约束在所属体素单元的局部邻域内。

采用分组交叉注意力机制对嵌入特征体素 $V_e$ 进行处理，并通过3D转置卷积进行上采样，最终输出 $\mathrm{V}_{\mathcal{G}}$ 。

将高斯基元投影回由粗解码得到的RGB、深度等特征图中，通过交叉注意力和MLP，**计算球谐系数的残差**，并与粗解码结果相加，从而恢复丰富的纹理细节。此过程还引入了**位移特征**以实现遮挡感知。

MSE+SSIM+正则化损失（含深度和法线路项）

#### Gaussian Masked Autoencoders

将mae与splatting结合

#### latentSplat: Autoencoding Variational Gaussians for Fast Generalizable 3D Reconstruction

针对稀疏视图（仅需2张）的3D重建问题，从 回归式与生成式两种方法的固有优缺点出发：

回归式方法 在输入视图信息充分（无遮挡、常见物体）时表现良好，但面对严重遮挡或罕见物体时，由于缺乏先验而容易失败；生成式方法 则能利用强大的学习先验，通过概率“想象”出缺失或不确定的内容。

受**VAE**启发，提出了 “变分3D高斯分布”，不再直接预测确定的场景特征，而是为每个3D高斯预测一个语义特征的分布（均值和方差），从而显式地建模了三维空间中的不确定性。

使用DINO提取图像特征，并通过Epipolar Transformer进行视图间特征交互。预测头不仅输出3D高斯的标准属性（位置、颜色等），还额外输出每个高斯的语义特征均值 $h_μ$和方差$ h_σ$，随后通过采样得到具体特征(也就是于)。

将渲染出的特征图输入到预训练的LDM解码器中，利用其强大的生成先验来补全因视图稀疏而缺失的细节。为了处理渲染图像中的空白区域（透明度高的地方），将随机噪声与未覆盖区域的不透明度相乘后添加到特征中，即 $F=F^{ren}+\sqrt{1−O}⊙ϵ$，这为生成器提供了需要“填补”区域的明确信号。

#### LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation

feed forward 类型gaussian splatting重建工作，输入四张图，可以进行物体重建。

提出非对称UNet架构：以图像和对应的Plücker射线为输入且解码器的输出分辨率低于输入分辨率。

针对使用ImageDream/MVDream等模型生成的多视角图像存在视角不一致和相机姿态误差的问题，引入数据增强方法：对其他输入视角图像进行随机扭曲，模拟生成模型的不一致性；绕场景中心随机旋转相机姿态，使模型对位姿误差更具鲁棒性。

对于mesh提取，生成的高斯基元并不像dreamgaussian那样密集，所以使用instantngp重新将得到的模型额外训练，然后使用nerf2mesh的思想进行mesh提取。

#### MeshFormer: High-Quality Mesh Generation with 3D-Guided Reconstruction Model

能够从稀疏的多视角RGB图像及其对应的法线图，通过单阶段训练高效地生成高质量的3D网格（Mesh）。采用更直接的3D体素作为场景表示而非Tri-plane。

由粗到细的两步策略：1、VoxelFormer：一个3D U-Net，生成一个低分辨率的Occupancy Volume，用于粗略定位表面区域。SparseVoxelFormer：仅对靠近表面的稀疏高分辨率体素进行处理，结合投影感知交叉注意力预测其详细特征，极大地提升了计算效率。

投影感知的交叉注意力：使用DINOv2处理多视角RGB和法线图像，生成多视角图像块特征。将每个3D体素投影到m个视角，插值得到m个RGB和法线特征，并与投影的RGB和法线值拼接，形成m个2D特征，利用注意力机制进行特征融合。

通过结合表面渲染和显式3D SDF监督，MeshFormer可以在一个统一的单阶段训练过程中进行训练。使用了三个小型MLP，从3D体素特征中插值3D特征作为输入，以学习SDF、3D颜色纹理和3D法线纹理。通过dual Marching Cubes算法从SDF体积中提取网格，并使用NVDiffRast进行可微表面渲染。

#### MeshLRM: Large Reconstruction Model for High-Quality Mesh

feedforward类nerf重建工作，整合了可微的mesh提取和渲染，能够端到端的进行mesh生成。

将相机的姿态转变到普吕克坐标系中，并与对应的图片进行拼接得到了九通道的输入，接着将输入patch处理并和triplane token一起送入transformer网络中得到新的triplane token特征，之后再使用两个轻量的MLP去分别预测密度与颜色信息。这一阶段在较低分辨率（256x256）下预训练基础的NeRF模型，使用L2和LPIPS损失。

之后引入可微的Marching Cubes（DiffMC）和可微光栅化器（Nvdiffrast）。这使得整个系统能够：从NeRF的密度场中可微地提取Mesh。通过查询Triplane特征和颜色MLP来渲染该Mesh的新视角图像。用渲染图像与真实图像的损失直接反向传播优化整个模型（包括Triplane生成器），从而实现真正的端到端网格训练。这一阶段在较高分辨率（512x512）下进行Mesh微调。此阶段联合优化NeRF表示和提取的Mesh，损失函数结合了渲染损失（L2 + LPIPS）、法线损失和提出的光线不透明度损失，以同时保证视觉保真度和几何质量。

#### MuRF: Multi-Baseline Radiance Fields

feed-forward类nerf稀疏视图3D重建工作，作者出发点在于MVS与NVS是有区别的。MVS只是利用reference img进行深度估计，而NVS是需要在新的视角下进行重建。本文认为应该通过从ref img中进行采样并且直接在目标视角下进行重建。

特征提取：输入多张参考图像，通过6层CNN进行8倍下采样；使用Unimatch Transformer进一步提取特征，得到多视图特征 $\{F_k\}_{k=1}^K$ 。

在目标视角建立体素（Volume），并将体素点映射回参考视图。为避免下采样信息丢失，以映射点为中心进行9×9窗口的颜色采样 $\{\tilde{c}_k^w\}$ 。进行特征匹配，计算多视图特征间的余弦相似度 $\hat{s} = w_{ij} \sum \cos(f_i, f_j)$ ，并通过可学习权重 $w_{ij}$ 加权。使用MLP学习权重，对颜色 $\hat{c}$ 和特征 $\hat{f}$ 进行加权融合，MLP输入包括采样特征及视角方向差异。拼接相似度、颜色和特征，通过线性层生成目标体素 $\boldsymbol{z}$ 。使用分解的3D CNN（2D+1D卷积）解码体素，输出密度与颜色（4通道）。采用Coarse-Fine分层采样提升重建效率与质量。

#### MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images

这篇工作是一个feedforward）3D高斯散射（3DGS），能够通过少量输入视图重建场景

输入多视图图像和对应相机参数，直接输出3D高斯元参数（均值、不透明度、协方差、颜色）。多视角特征提取与交互：使用浅层ResNet提取4倍下采样的图像特征；通过Swin Transformer进行自注意力和跨视角注意力交互

构建每个视图的**cost volume**：使用逆深度采样D个深度假设，将特征投影到其他视图并计算匹配代价：

$$
\mathbf{C}_{d_m}^i = \frac{\mathbf{F}^i \cdot \mathbf{F}_{d_m}^{j \rightarrow i}}{\sqrt{C}}
$$

得到初始cost volume：\(\mathbf{C}^i \in \mathbb{R}^{\frac{H}{4} \times \frac{W}{4} \times D}\)，结合feat使用U-Net（含跨注意力）预测残差，得到优化后的cost volume

高斯参数预测：均值，通过逆投影将深度图映射回3D空间；不透明度：由cost volume的匹配；协方差与颜色：将图像、cost volume和特征拼接后，通过两层卷积预测 

#### MVSplat360: Feed-Forward 360 Scene Synthesis from Sparse Views

（这篇工作接续MVSSplat）feed-forward类重建工作，使用极其稀疏的观察（例如少于 5 张图像）渲染宽范围甚至 360° 视图；针对稀疏输入下大场景重建易产生噪声伪影的问题，引入生成模块进行去噪，

稀疏多视角重建：使用跨视角Transformer编码器提取并融合多视图特征 $\mathcal{F}=\{\boldsymbol{F}^i\}_{i=1}^N$ ；将深度均匀划分为 $L$ 段 $\mathcal{D}=\{D_m\}_{m=1}^L$ ，通过相机位姿映射特征： $\boldsymbol{F}_{D_m}^{j\to i}=\mathcal{W}(\boldsymbol{F}^j,\boldsymbol{P}^i,\boldsymbol{P}^j,D_m)$ ；构建cost volume： $\boldsymbol{C}_{D_m}^{i}=\frac{\boldsymbol{F}_{D_m}^{j\to i}\cdot\boldsymbol{F}^{i}}{\sqrt{C}}$ ，沿深度维度聚合后通过softmax得到深度估计 $ d $。

预测高斯参数：均值  $\mu=\mathrm{K}^{-1}\boldsymbol{u}d+\Delta$ （含深度和偏移）、不透明度、协方差和颜色，并额外输出特征用于条件生成。

生成去噪模块 ：采用SVD（Stable Video Diffusion）模型，以稀疏视图的CLIP全局平均特征为条件，对高斯特征进行增强。生成图像后通过直方图匹配调整颜色饱和度。

#### PixelGaussian: Generalizable 3D Gaussian Reconstruction from Arbitrary Views

提出了一种feed forward动态高斯建模方法，用于解决传统方法中高斯分布冗余重叠的问题。传统方法在输入图像上均匀预测高斯分布，导致大量重叠的高斯，尤其在多视角输入时性能下降。

利用MVSplat的思想，使用CNN和Swin Transformer提取多视角特征，通过cost volume估计深度，结合相机位姿得到初始高斯位置 \(\mu\)，其余参数随机初始化。

Cascade Gaussian Adaptor (CGA)：计算score map $\mathcal{R} = \Psi(\mathcal{F})$ ，用于评估区域重要性。引入超参网络 $\mathcal{H}$ ，输入每一阶段高斯相关的信息输出阈值 $\tau_{high}^{(k)}, \tau_{low}^{(k)}$ ，指导高斯的分裂与合并：高分区域分裂高斯（通过SplitNet生成新高斯）；低分区域合并高斯（缩放透明度与尺寸）。

Iterative Gaussian Refinement (IGR)  ：对高斯进行微调，通过与多视角特征交互更新参数：

$$
\mathcal{Q}_b = \Phi_{ref}\left(\sum_{i=1}^{N} \alpha_i \cdot \text{DA}(\mathcal{Q}_{b-1}, F_i, P(\mu^{(b)}, C_i))\right)
$$

最终通过MLP输出优化后的高斯参数 \(\mathcal{G}_f\)。

#### pixelNeRF: Neural Radiance Fields from One or Few Images

nerf的feed forward类工作，最少只需要一张图片就能够生成新的场景

使用在ImageNet上预训练的ResNet-34；提取四个不同尺度的图像特征图，将其统一缩放到2倍下采样尺寸后拼接；该编码器在训练过程中被冻结，参数不更新。剩下的思路和nerf一致，只不过mlp中需要多融合对应的featmap特征。

#### pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction

feedforwar GS重建模型，能够实时且memory-efficient渲染场景，只需要两张图片就可以渲染出高质量场景。最少需要两张图片进行重建，因为需要三角量测法帮助模型估计出相片中的实际深度。

每个视图被编码成特征体积 $F$ 和 $\tilde{F}$，对于图像 $I$ 中的像素坐标 $u$，$ℓ$为穿过该坐标的一条射线。沿着 $ℓ$ 方向采样像素坐标 ${\tilde{u}_i}∼\tilde{I}$。对于每个极线样本 $\tilde{u}_l$，通过三角测量计算其到图像 $I$ 的相机原点的距离 $\tilde{d}_{\tilde{u}_l}$。

在该框架下，高斯的中心如果直接当成距离去预测的话容易陷入局部最优的情况，以转而预测位置的概率密度，但是由于位置的确定过程涉及到采样操作，该操作本身是不可导的，作者随后将不透明度与概率密度预测耦合了起来，即将不透明度的梯度传给概率预测

#### SparseNeuS: Fast Generalizable Neural Surface Reconstruction from Sparse Views

NeuS的feedforward类，稀疏图片重建方法。

多层级几何推理：输入N张图像及相机参数。通过相机参数生成3D边界框并体素化，得到3D坐标；使用特征提取网络获取图像特征图；将3D坐标投影到特征图并插值得到特征，计算特征方差构建Cost Volume；使用稀疏3D CNN聚合特征；此外构建两阶段encoding volumes平衡网络来平衡网络的效率；MLP输入位置编码后的3D坐标和几何编码 \( M(q) \)，输出符号距离函数 \( s(q) \)。

多尺度颜色融合 (Multi-scale Color Blending)；将3D点 \( q \) 投影到所有输入视图，融合颜色；MLP根据特征、全局光照一致性（均值/方差）、几何编码和视角方向预测混合权重 \( w_i^q \)。像素级渲染：沿射线采样，通过NeuS将SDF转换为密度，渲染像素颜色。块级渲染：基于局部表面平面假设，通过单应性变换高效提取颜色块，共享混合权重以保持上下文一致性。

一致性感知微调：固定几何编码体积和SDF网络。替换CNN混合网络为MLP，输入坐标、法线、视角、SDF和几何编码。输出混合混合权重。

#### Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs

feedforward类GS 3D重建工作并且是pose-free类型的工作

无需已知或估计相机位姿，仅用两张任意图片即可进行3D重建和新视角合成。两阶段训练：优化3D点云的几何损失；进行新视角合成训练。使用损失掩码策略通过视锥体剔除和共视性测试，生成一个损失掩码。该掩码只对在输入图像中可见且可合理重建的像素计算损失。

该方法建立在特征匹配模型 MASt3R 之上（其预训练权重被冻结）。输入两张图，输出每个像素的3D坐标和置信度，天然适配本工作对无位姿的需求。在 MASt3R 的基础上，添加了预测高斯属性的额外输出头，并引入了预测颜色残差（偏移项） 的机制以更好地捕捉高频细节。

#### Splatter Image: Ultra-Fast Single-View 3D Reconstruction

feedforward类GS 3D重建工作，渲染速度快，并且能够以单张图片输入进行重建

模型接收一张输入图像、一个目标图像以及两者之间的相机相对位姿作为训练数据。它首先从输入图像预测出3D高斯场景，然后利用给定的位姿将该场景渲染到目标视角，最后通过比较渲染结果与真实目标图像来计算损失；在颜色表示方面，采用低阶球谐函数。当高斯点随相机旋转时，其球谐系数的相应变换规则，确保了颜色在不同视角下的一致性。

网络架构设计上，模型直接输出一组定义在相机坐标系下的高斯参数，包括位置、协方差、不透明度和颜色系数。相机位姿被编码为条件信号，通过FiLM模块注入到U-Net中，使网络能够感知视角变化。

#### Triplane Meets Gaussian Splatting: Fast and Generalizable Single-View 3D Reconstruction with Transformers

feedforward类重建工作将显式点云与隐式三平面特征相结合，既保留了显式表示的渲染效率，又具备了隐式表示的高质量重建能力。

首先使用DINOv2作为图像编码器提取特征，并使用adaLN将相机参数融入处理过程。随后通过点云解码器生成包含2048个点的初始点云。考虑到高斯泼墨的渲染质量严重依赖于高斯分布的数量，使用Snowflake反卷积模块，将点云上采样八倍至16384个点。

为了在重建过程中充分利用输入图像的指导信息，引入了投影感知条件机制。从输入图像中提取包括颜色、语义特征、掩码和距离变换在内的多种局部特征，并将其整合到上采样过程中，确保生成的几何细节与输入图像保持一致。

在表示层面，将点云投影到三个正交平面上形成三平面特征。当需要查询任意三维点的特征时，只需从这三个平面插值并拼接特征即可。这些特征随后被输入到一个多层感知机中，解码出每个splatting所需的位置偏移、透明度、尺度、旋转和球谐系数等属性。

## LLM

#### LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias

纯数据驱动的大模型重建任务，不依赖3D表示及渲染方程，提出编码器-解码器与仅解码器（LVSM）两种结构，后者在质量、可扩展性和泛化性上更优。

编码器-解码器框架中，多视角图像与相机位姿经patchify处理后，与可学习token一同输入编码器，压缩后的token作为条件送入解码器，与目标位姿交互映射。

仅解码器架构则直接让图像输入token与目标位姿交互更新。两者输出均通过unpatchify恢复原始分辨率图像，损失函数采用MSE与感知损失结合的方式。

#### SpatialLM: Training Large Language Models for Structured Indoor Modeling

面向结构化室内建模的大型语言模型（LLM），为处理3D点云数据，生成包含墙、门、窗等建筑元素及带语义类别的定向物体框的结构化3D场景结果遵循标准多模态LLM架构，基于开源LLM直接微调，SPATIALLM将结构化场景描述转化为Python脚本，以文本形式预测场景信息，兼顾可解释性、类别扩展性与LLM编码能力利用。为支撑训练，构建大规模合成数据集，含12,328个室内场景（54,778个房间）的点云与3D标注，筛选出59个物体类别（412,932个标注实例）。 模型实现以Qwen2.5-0.5B为基础模型、Sonata为点云编码器，最精细层级分辨率设为2.5cm。

## Diffusion

### shape

#### 3DShape2VecSet: A 3D Shape Representation for Neural Fields and Generative Diffusion Models

提出将扩散模型（SD）思想融入3D形状表示的框架，核心流程围绕点云压缩、隐向量生成、形状解码及扩散网络构建展开。   首先，对3D形状表面采样得到点云数据，两种压缩表示方案：一是采用可学习查询集，二是利用点云本身通过最远点采样（FPS）降维。

VAE：随后通过KL正则化块生成隐向量，该模块在引入生成模型时为必需组件，通过线性投影层计算均值和方差，结合随机噪声生成隐向量并施加正则化约束。   形状解码阶段，在编码与解码间添加latent网络，通过自注意力机制增强特征表达能力。给定查询时，利用注意力机制对隐变量进行特征交互，进而通过全连接层预测占据值（occupy value）。表面重建则基于高分辨率网格采样，通过Marching Cubes算法生成最终表面。   扩散网络部分采用EDM结构，网络层包含自注意力（SelfAttn）和交叉注意力（CrossAttn），后者用于注入图像、文本或部分点云等条件信息（分别通过ResNet、BERT或形状编码器提取特征）。

#### 3DTopia-XL: Scaling High-quality 3D Asset Generation via Primitive Diffusion

提出名为 **PrimX** 的3D场景表示方法，旨在高效支持基于物理的渲染（PBR），并融入了类似 Stable Diffusion 的生成式建模思想。PrimX 使用一组分布在物体表面的体积基元（volumetric primitives） 来近似表达3D网格的形状、颜色和材质信息。每个基元包含其3D位置、缩放因子以及一个存储了局部SDF（有符号距离场）、RGB颜色和材质（金属度、粗糙度）信息的小体素块。通过加权组合这些基元，可以得到一个平滑的、连续的神经体积函数，该函数能够表征整个3D物体。一旦得到 PrimX 表示，可以通过使用 Marching Cubes 算法从SDF函数的零水平集提取3D网格；对网格进行UV展开，然后查询 PrimX 表示中的颜色和材质函数，生成对应的反照率贴图、金属度贴图和粗糙度贴图。最后将这些打包成GLB文件，供图形引擎使用。
给定一个带纹理的3D网格，通过在网格表面采样点来确定体积基元的位置和初始大小。然后，通过查询原始网格的SDF、颜色和材质信息来初始化每个基元体素块的内容。通过损失函数对基元的位置、尺度和体素负载进行微调，使 PrimX 表示能尽可能准确地拟合原始网格。

引入了类似 Stable Diffusion 的两阶段流程：训练一个3D VAE，将每个体积基元的体素块压缩成一个低维的潜在表示。这一步将高维的体素数据转换为紧凑的潜在标记（latent tokens）。在潜在空间训练一个扩散模型。该模型采用类似 DiT 的 Transformer 架构.。

#### CAT3D: Create Anything in 3D with Multi-View Diffusion Models

多视角图像重建的核心是学习在给定若干条件视图及其相机参数的情况下，生成多个目标视角图像的联合分布。模型基于预训练的二维潜在扩散模型构建，通过引入相机姿态嵌入和三维自注意力机制，将多个输入图像的潜在表示相连接（既有观察图像，又有目标图像噪声）。训练过程中，仅对目标图像的潜在表示添加噪声，并加入二进制掩码来区分条件图像和目标图像。模型可处理最多8个条件视图和目标视图的组合，训练时随机选择1或3个条件视图。

在生成新视图时，针对不同类型场景设计了多种相机轨迹，以确保充分覆盖场景且避免不合理视角。由于模型一次只能处理有限数量的视图，生成大量视图时需将目标视点分组，并通过自回归采样策略逐步生成。对于单图像条件，首先生成少量锚视图，再基于已有视图并行生成其他视图组。

#### CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets

这篇论文的核心是将Stable Diffusion（SD）的思想融入3D生成，并进一步整合了材质生成与多条件控制。整体架构基于3DShape2VecSet，通过点云编码将3D几何表示为潜在向量，并利用扩散模型进行去噪生成。在生成过程中，采用**多分辨率**点云采样以丰富几何细节，并通过Marching Cubes算法将生成的占用场转换为三角网格。为了提高生成资产的质量，后续还进行了网格优化（如三角转四边形）和高质量PBR纹理合成。纹理生成部分借鉴了MVDream，并引入多分支UNet和超分技术，实现了2K级高分辨率、多属性的材质贴图。为了支持多样化的控制生成，为每种输入条件（如图像、体素、边界框、稀疏点云等）单独训练了交叉注意力模块。这些条件信息通过特征提取和位置编码被融合到扩散过程中，使得模型能够根据图像、体素、边界框或点云等输入，生成相应的高质量3D内容。还对训练数据进行了格式统一与自动化标注，利用GPT-4V生成细粒度文本描述，增强了模型对几何与风格的理解能力。

#### CraftsMan: High-fidelity Mesh Generation with 3D Native Generation and Interactive Geometry Refiner

采用stable diffusion的思路，通过3DVAE模型将3D资产编码到隐空间。编码器处理点云和法线信息，解码器将隐向量重建为occupacy Field，使用 marching cube 进行网格采样，并通过remeshing工具优化网格质量。（VAE部分参照 3DShape2VecSet以及Michelangelo）

在隐空间上训练条件3D扩散模型，利用多视角生成作为中间条件。多视角图像经过特征提取后，通过基于相机参数的调制方法融入扩散过程，指导3D形状生成。

最后进行基于法线的几何优化：将粗网格渲染为法线图，使用法线扩散模型增强细节，然后通过可微分渲染优化网格顶点，最小化渲染法线与增强法线之间的差异。整个过程还引入了拉普拉斯平滑来稳定优化，并支持交互式局部细化，用户可通过绘画刷选择区域进行针对性修复。

#### Direct3D: Scalable Image-to-3D Generation via 3D Latent Diffusion Transformer

在 3D 生成中，于 VAE 阶段将点云数据编码为 tri-plane。通过在 3D 物体表面均匀采样，结合 perciever 架构和可学习 token 得到 tri-plane 特征，再经卷积上采样。训练时采用半连续点云采样方法，通过 MLP 预测查询点 occupancy，优化了传统离散占据表示在物体表面梯度突变问题。对 DiT 的改进借鉴 PixArt 方法，利用时间嵌入预测全局参数，并添加可训练嵌入进行模块调整。

#### Tencent Hunyuan3D-1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation

先由单张图片生成多视角图片，再通过多视角图片生成 3D 模型。在多视角扩散生成模型中，结合 Zero-1-to-3++ reference attention ，将模型替换为 stable diffusion，目标是生成 0 仰角、6 个均匀方位角、以 3x2 排列的图像。训练时重新设计了 CFG 权重。在稀疏视图下 3D 重建，将上阶段生成的图像及对应相机参数 embedding 输入后续网络，为完善物体上方与底面细节，还会加入非 0 仰角生成的图片（其 camera embedding 设为 0）。同时借鉴 Meshlrm、Gs-lrm 对 triplane 超采样减少伪影，最后借鉴 NeuS 从 3D 表示中提取 mesh。

#### Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation

Hunyuan3D 提出了两个核心模型：形状生成与纹理生成。

形状生成：包含两个模块：ShapeVAE（整体架构继承自3DShape2VecSet） 和 DiT；ShapeVAE：输入为从 3D 形状表面采样的点云坐标与法向量，编码器输出隐变量，解码器预测 SDF。除了均匀采样，还引入重要性采样以在边缘和角落处增强细节。编码器使用最远点采样进行下采样，解码器通过自注意力和交叉注意力生成 3D 辐射场，最终输出 SDF。损失函数结合 SDF 重建误差和 KL 散度，并采用多分辨率训练策略；DiT：基于 FLUX 架构，将图像条件与隐变量序列输入扩散模型。使用 DINOv2 提取图像特征，并去除背景以提升效果。损失函数采用流匹配损失。

纹理生成 ：分为图像预处理与多视角纹理生成。预处理：对输入图像进行去光处理，并基于启发式方法选取 8–12 个视角，确保覆盖 UV 空间。Paint 模型：基于 SD2.1，引入参考注意力和多视角注意力，增强多视角间的一致性。同时加入几何条件（法线贴图与坐标贴图）和可学习相机嵌入，以提供视角信息。训练时采用视图丢弃策略，提升泛化性；推理时可输出任意指定视角的图像。最后使用超分模型提升纹理质量，并通过投影与插值填补 UV 贴图中的空缺区域。

#### Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material

在数据处理和网络结构两方面展开：

从ShapeNet等公共数据集及自定义数据集中收集超10万个带纹理和不带纹理的3D数据，还从Objaverse-XL筛选出超7万个经人工标注的高质量数据用于纹理合成。归一化处理：计算3D物体轴对齐 bounding box，将物体均匀缩放适配到以原点为中心的单位立方体中，点云数据则通过减去质心中心化后，按到中心的最大欧氏距离缩放，以统一尺度并保留几何关系。watertight处理：利用IGL库构建符号距离场生成封闭表面，通过移动立方体算法在零水平集提取无边界不连续性的拓扑闭合表面。SDF采样：采用靠近形状表面和在整个$[-1,1]^{3}$空间均匀分布两种策略随机选择查询点，计算SDF值。表面采样：结合均匀采样和特征感知采样，各占最终点集的50%，以捕捉完整几何信息。条件渲染：使用Hammersley序列算法在球面上均匀分布采样150个相机，生成包含随机视场角和调整相机半径的增强数据集。

纹理合成预处理：从Objaverse和Objaverse-XL筛选数据，从四个仰角及每个仰角24个均匀分布的方位角渲染数据，生成相应贴图及图像，并概率性渲染参考图像。

Hunyuan3D-ShapeVAE：将多边形网格表示的3D资产形状压缩为潜在空间中的连续令牌序列，条件编码器采用DINOv2 Giant。        

Hunyuan3D-DiT：在ShapeVAE潜在空间上训练，用于从用户提供的图像预测物体令牌序列，通过VAE解码器解码回多边形网格，其扩散Transformer块堆叠21个Transformer层，包含维度拼接、交叉注意力层和混合专家层。使用流匹配目标训练模型预测速度场，推理时随机采样起点，用一阶欧拉常微分方程求解器计算结果。   

 Hunyuan3D-Paint：引入基于物理渲染的材质纹理合成框架，遵循BRDF模型，输出反照率等贴图，引入3D感知旋转位置编码提高跨视角一致性。在混元3D 2.0多视角纹理生成架构基础上，引入新的材质生成框架，保留ReferenceNet特征注入机制，拼接几何渲染贴图与潜在噪声。空间对齐多注意力模块：采用预训练VAE压缩多通道材质图像，实现并行双分支UNet架构，为反照率和金属度-粗糙度贴图设计包含多种注意力的并行多注意力模块，并传播反照率参考注意力模块计算输出到MR分支。利用同一物体不同光照条件下渲染的参考图像训练样本计算一致性损失，以生成无光照和阴影的反照率贴图及准确的MR贴图。

#### Hunyuan3D 2.5: Towards High-Fidelity 3D Assets Generation with Ultimate Details

形状生成模型通过扩大模型参数和训练数据量实现，模型结构继承自 2.0 和 2.1 版本；纹理生成模型在 Hunyuan3D 2.1 的多视图 PBR 纹理生成架构基础上扩展而来，以 3D 网格渲染的法向图和 CCM 作为几何条件，参考图像作为引导，借鉴 3D 感知 RoPE 增强跨视图一致性，实现无缝纹理贴图生成。

多通道材质生成中，为反照率、MR、法向三种材质贴图引入可学习嵌入，初始化三个独立嵌入向量，通过交叉注意力层注入相应通道，嵌入和注意力模块可训练。提出双通道注意力机制确保反照率与 MR 的空间对齐，在多通道间共享注意力掩码，利用基色通道计算的注意力掩码指导其他分支的参考注意力，使生成的反照率和 MR 特征保持空间一致性并受参考图像引导，同时训练中引入光照不变性一致性损失，实现材质属性与光照成分解耦。

几何对齐方面采用双阶段分辨率增强策略，第一阶段用传统多视图训练方法，使用 6 视图 512×512 图像；第二阶段实施 “缩放训练” 策略，训练中随机缩放参考图像和多视图生成图像，使模型学习细粒度纹理细节，规避直接高分辨率多视图训练的内存限制。推理阶段利用高达 768×768 分辨率的多视图图像，通过 UniPC 采样器加速，实现高效高质量生成。

#### LN3Diff: Scalable Latent Neural Fields Diffusion for Speedy 3D Generation

现有 3D 生成模型存在可扩展性差、效率低、泛化性不足的问题。融入SD思想进行 3D 生成，编码器输入系列图片及相机plvcker坐标，输出 8 倍下采样的 3 维空间特征；解码器采用 transformer，用self-plane和cross-plane attention机制，将 ViT 换成 DiT 结合 adaLN 效果更佳，decoder upsampler 用卷积上采样处理 triplane。训练时，在新视角无监督情况下，引入输入与 novel 视角辨别器，训练损失综合渲染、几何、KL 散度和对抗损失。diffusion 采用 SDE，借鉴 LSGM 思想并改进为 v-prediction 和重要性采样，文本和图像条件均通过 CLIP 编码，双条件时经 rescale 后相加。

#### Locally Attentional SDF Diffusion for Controllable 3D Shape Generation

LAS-Diffusion，该方法采用两阶段生成策略，以解决SDF表示法计算开销大的问题。第一阶段在低分辨率下生成粗略的占用网格；第二阶段则对结果进行细分，并使用SDF表示进行精细建模。两个阶段均采用自条件（self-condition）策略进行训练。将3D数据集归一化至[-0.8, 0.8]³范围，为后续SDF计算做准备。在第一阶段，使用一个5层尺度的UNet，通过设定特定阈值从数据集中提取占用网格。网络在推理时仅保留大于0.5的结果作为输出。

草图与相机位姿作为关键条件信息被引入。利用Vision Transformer（ViT）提取草图的图像块特征。将体素信息反向映射回草图平面以获取对应坐标，并将该坐标邻近区域的图像块特征与网络中的特征令牌进行交叉注意力计算，从而实现条件引导。在第二阶段，将第一阶段的粗略占用网格进行细分，然后输入到一个更轻量的4层尺度UNet中，进行精细几何与SDF值的预测，最终完成高精度3D网格的生成。

#### MAR-3D: Progressive Masked Auto-regressor for High-Resolution 3D Generation

为了解决3D重建中VAE忽略几何细节、Transformer计算量高以及缺乏渐进式分辨率提升策略的问题，本文提出了MAR-3D模型。该模型融合了金字塔VAE和级联掩码自回归Transformer，能够在连续空间中渐进式地提升潜在分辨率。训练时采用随机掩码，推理时采用随机自回归去噪，以适应3D数据的特性，并结合条件增强的级联训练策略以加速收敛。

金字塔VAE：编码器对输入点云进行下采样，得到多个分辨率的点云，并将其与表面法线拼接成多分辨率点嵌入。通过跨注意力机制，可学习查询与各层级的点嵌入交互，粗层级捕捉结构特征，细层级提取几何细节，再经过自注意力得到潜在表示。解码器则通过自注意力处理编码特征，并采样查询点，利用交叉注意力预测占用值。

级联掩码自回归模型：该模型由MAR-LR和MAR-HR组成，架构相同但输入不同。借鉴MaskGIT的随机并行解码策略，训练时采用掩码自动编码器（MAE）随机掩码潜在表示，并使用扩散损失进行优化。推理时从图像表示开始迭代生成。

MAR-LR为潜在表示补充可学习的位置信息，并利用CLIP和DINOv2的特征处理条件图像，训练时随机归零条件特征以实施无分类器引导。MAE编码器和解码器对图像和潜在表示进行随机掩码，采用双向注意力机制。扩散过程中，解码器生成条件向量，通过轻量级MLP从高斯噪声重建真实表示。

MAR-HR采用由粗到细的生成策略，为了减少训练与推理的差异，对低分辨率表示添加高斯噪声。推理时，首先生成随机的表示生成顺序，从输入图像提取图像表示，输入MAE编码器-解码器。根据预定义的生成顺序选择条件向量，并行进行DDIM去噪。每个自回归步骤中处理的表示数量遵循余弦调度，初始步骤处理较少表示，后续步骤逐步增加。

在扩散模型中使用无分类器引导（CFG），其系数采用线性策略，逐步调整有条件与无条件输出的权重，以平衡生成质量与多样性。

#### Michelangelo: Conditional 3D Shape Generation based on Shape-Image-Text Aligned Latent Representation

与Stable Diffusion思想结合的方法，核心在于将3D形状的隐空间与图像、文本的隐空间对齐。设计了Shape-Image-Text Aligned Variational Auto-Encoder（SITA-VAE）模型。

SITA-VAE：利用预训练并冻结的CLIP图像编码器和文本编码器，分别处理图像和文本输入，生成对应的嵌入表示。可训练的3D形状编码器处理从3D表面采样的点云数据，通过带有傅里叶位置编码和线性投影后，使用基于perceiver的编码器提取形状的潜在嵌入。该嵌入包含一个全局语义token和多个局部几何token，共同构成形状的隐表示。

为了在数据有限的3D模态与丰富的图像-文本模态之间建立联系，模型通过对比学习进行训练。使用对比损失函数来拉近3D形状嵌入与对应的图像嵌入、文本嵌入在隐空间中的距离，从而利用大规模图像-文本数据增强3D表示的对齐能力。在解码阶段，一个基于Transformer的神经场解码器利用形状的潜在嵌入，通过交叉注意力机制查询3D空间点的占用情况，逐步重建出3D形状。

还引入了对齐形状潜在扩散模型（Aligned Shape Latent Diffusion Model）来提升生成质量。该方法在对齐的隐空间上进行扩散过程，其训练目标为预测噪声，并使用CFG策略在推理时调节生成过程。

#### Stable Virtual Camera: Generative View Synthesis with Diffusion Models

该方法旨在通过生成模型实现高质量的新视图合成（NVS），重点关注生成结果的平滑性、模型泛化能力以及在稀疏输入条件下的重建质量。

该框架基于Stable Diffusion 2.1构建，包含自动编码器和潜在去噪U-Net。为了增强时序建模能力，将U-Net中低分辨率残差块的2D自注意力扩展为3D自注意力，并在其后通过跳跃连接引入1D自注意力，使参数量增至13亿。还可选择性地加入3D卷积，将模型扩展为视频模型，参数量达到15亿，在推理时若帧序列有序，启用时间路径可进一步提升输出平滑度。

为将基础模型适配于多视图任务，引入了相机姿态作为Plücker嵌入条件，并进行了场景尺度归一化。输入帧的潜在表示与相机嵌入及视图类型掩码相连，目标帧则进行加噪处理。同时引入CLIP图像嵌入以提供语义信息，并对新增层的权重进行零初始化。训练采用两阶段课程学习策略：首先以较短上下文长度和大批量进行训练，随后增加序列长度并减小批次继续训练。训练时随机选取输入与目标帧，并采用小子采样步长以保证时间连续性。在可选阶段中，还使用视频数据专门微调时间相关权重。训练过程中随帧数增加动态调整信噪比，所有训练均基于方形高分辨率图像。

在生成新视图时，对于较小规模任务，采用单阶段采样，通过重复输入帧将序列填充至模型容量。当目标视图数量超出单次前向处理能力时，启用两阶段采样：首先生成一组锚帧，再以其为基础分块生成最终目标帧。对于集合NVS，采用基于最近邻的分块策略；对于轨迹NVS，则通过插值方式在相机路径上均匀生成锚帧并补全中间帧，从而优化时间平滑性。在输入视图较多时，模型能较好地进行插值；而在稀疏输入下则避免扩展以防性能下降。生成长序列时，引入锚帧记忆机制，通过检索空间最近邻的锚帧自回归地生成后续内容，有效维持长程3D一致性，优于传统时序最近邻方法。

#### SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion

利用生成模型提升多视角生成一致性，运用多种技巧。在一致性视图生成部分，采用SV3D并对原始SVD模型改进，如移除无关条件、进行条件图像处理、使用CLIP嵌入和相机轨迹嵌入。相机轨迹有静态和动态两种，静态轨道相机以固定仰角绕物体均匀分布的方位角旋转，动态轨道则在静态轨道基础上添加随机噪声和多频正弦波随机加权组合。训练时CFG从前视图1线性增加到后视图2.5再减回1，微调先让SVD无条件生成静态轨迹，再在相机位姿条件下生成动态轨迹。 

生成3D模型的pipeline采用由粗到细训练方案，粗阶段用Instant-NGP的NeRF表示法在较低分辨率重建SV3D生成的图像，细阶段从训练好的NeRF提取网格并采用DMTet表示法在全分辨率下微调，后处理进行UV展开并导出网格。生成时提出解耦光照模型，用24个球面高斯建模光照，考虑朗伯反射，还引入损失减少嵌入光照影响。 此外，针对单纯使用SDS辅助物体重建出现纹理过饱和等问题，提出maskSDS loss，通过从随机相机轨道渲染对象、计算参考相机视图方向等流程应用损失。同时使用深度平滑损失函数、双边法向平滑损失和深度损失等几何损失函数。

#### TEXTure: Text-Guided Texturing of 3D Shapes

利用Stable Diffusion，通过深度图与修复模型，结合UV纹理图集，从多视角逐步为网格着色。

在纹理生成中，系统从初始视点出发，依据深度图生成初始彩色图像并投影至纹理。后续选取固定视点，渲染网格得到深度图与当前纹理渲染图，并生成新图像以更新纹理。为控制生成过程，将渲染视图划分为“保留”、“优化”和“生成”三个区域。“生成”区是首次可见需绘制的区域；“优化”与“保留”区则根据三角形法线方向及历史绘制质量进行区分，旨在减少三角面片在倾斜视角导致的纹理失真。

在采样去噪过程中，通过掩码控制，固定“保留”区域内容不变，并将当前渲染图的噪声注入到“优化”与“生成”区域，以保持纹理连贯性。为解决新生成区域与已有纹理不一致的问题，交错使用深度条件模型和修复模型的策略。对于“优化”区域，在采样前期使用棋盘格状掩码引导噪声与现有纹理对齐，后期则放开生成。将图像投影回图集时，通过基于梯度的优化实现，并在区域边界应用软掩码（结合高斯模糊）以实现不同视图间纹理的平滑过渡。

在纹理转移方面，该方法能够将已绘制网格或少量输入图像的纹理迁移到新的目标几何形状上。为增强鲁棒性，对源网格施加了基于拉普拉斯谱的随机低频几何变形以进行数据增强。训练使用结合视图方向和纹理标记的特定提示语。微调后的模型即可用于为目标形状生成纹理。若纹理源仅为少量图像，则先通过显著性分割提取主体，并进行数据增强后，再在深度条件模型上进行训练。

#### ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis

结合视频扩散模型与点云先验的方法，用于从单张或稀疏图像合成高质量的新视图。现有方法受限特定类别、室内场景或单一物体，视频扩散模型虽具备强先验但缺乏三维理解，点云重建方法如DUSt3R虽能捕捉几何信息但重建质量有限，因此将二者优势结合以实现高保真视图合成。

该方法基于动态图像到视频（I2V）扩散模型DynamiCrafter，并引入DUSt3R从参考图像重建点云并估计相机参数。对于单张输入，通过复制图像构建图像对，利用点云重建获取相机姿态与内参；对于多张输入，则进行全局点云对齐。在推理时，沿指定相机轨迹渲染点云序列，但由于点云本身存在遮挡、缺失和视觉质量低的问题，直接渲染难以得到理想结果。训练以点云渲染和参考图像为条件的视频扩散模型，学习生成高质量新视图的条件分布。模型基于潜在扩散架构，包含VAE编解码器、视频去噪U-Net和CLIP图像编码器。训练时，使用点云渲染与其实图像配对数据，冻结VAE部分，在潜空间优化噪声预测目标。推理时，将参考图像与点云序列编码至潜空间，通过迭代去噪生成新视图序列。

为生成长序列且稳定的视频，提出迭代视图合成策略：从初始点云出发，基于“最佳下一视角”规划相机轨迹，每次生成新视图后反向投影更新点云，逐步扩展视角并完善几何表示。轨迹规划中，通过效用函数评估候选视角对遮挡和缺失区域的覆盖程度，选择最优视角进行插值与生成。

#### XCube: Large-Scale 3D Generative Modeling using Sparse Voxel Hierarchies

一种基于前馈网络、融合扩散模型思想、以由粗到细方式生成高分辨率三维体素的方法

方法采用分层体素网格结构，包含从粗到细的L层体素网格及其对应属性（如语义和法线），每层之间具有严格的包含关系。核心组成部分包括稀疏结构变分自编码器（VAE）和分层体素潜在扩散模型。

稀疏结构VAE负责对每一层体素进行编码和解码。编码器利用稀疏卷积和池化操作将输入体素及其属性降采样至潜在表示的分辨率；解码器则采用神经核曲面重建方法，从潜在表示出发，通过逐步修剪、细分和上采样，重建出当前层级的体素与属性。在分层体素潜在扩散部分，模型构建了一个联合概率分布，将每一层的生成过程分解为基于上一层级结果的条件扩散过程。扩散模型采用v-prediction方式，并将上一层的属性特征与网络输入直接拼接作为条件注入。时间步信息通过AdaGN模块融入，文本提示则通过CLIP编码后借助交叉注意力机制进行交互。

训练过程中，VAE部分采用包含重建损失和KL散度的标准损失函数，属性损失（进一步整合了法线误差、语义分类误差和基于TSDF的表面误差）。扩散部分则采用参考速度目标进行监督。在采样生成时，过程从最粗糙层级开始，通过扩散模型采样得到潜在变量，经VAE解码器生成该层体素及属性，并可选择通过一个细化网络进一步优化输出。随后，将生成的体素与属性连同全局条件（如文本）一起作为下一层扩散模型的条件，重复此过程直至生成最精细层级的体素网格，其中最高层属性包含可用于表面重建的TSDF信息。

为提高生成质量，在稀疏卷积网络中实施“早期膨胀”以保留更多局部上下文；引入“细化网络”来修正层级传递累积的误差；在VAE解码前对其后验分布添加噪声以增强鲁棒性。条件信息（如上层属性、文本、类别标签或单次扫描点云）通过拼接、交叉注意力或AdaGN等不同方式注入模型。此外，还设计了条件注入方案来处理扫描数据中体素稀疏性问题。

#### RomanTex: Decoupling 3D-aware Rotary Positional Embedded Multi-Attention Network for Texture Synthesis

该方法对Stable Diffusion进行扩展，构建了一个图像引导的多视图生成模型，旨在提升3D重建与新视图合成中的多视图一致性和几何准确性，最终进行纹理生成。

核心改进在于引入了多视图注意力机制与3D几何条件。模型在原有自注意力块基础上，增加了参考注意力块和多视图注意力块，通过并行结构整合多种条件。为了注入3D信息，将世界坐标系中的规范坐标图和法向图与去噪潜变量进行连接。通过3D感知的旋转位置嵌入技术，将3D位置信息融入注意力计算。该方法建立多分辨率2D-3D对应关系，构建分层体素金字塔，使UNet的多尺度特征图能够与不同分辨率的3D几何信息对齐。这种双向映射机制不仅提高了几何条件的准确性，还显著改善了跨视图一致性。解耦的多注意力模块采用并行结构，同时处理自注意力、多视图注意力和参考注意力。为避免多视图注意力和参考注意力之间的功能纠缠，训练时引入了dropout策略，通过将多视图分支的权重设为零来实现，这有效提升了在参考相机姿态之外生成视图的稳定性。

针对几何条件与参考图像条件可能存在的冲突，提出了几何相关的无分类器引导技术，在几何特征明显时优先遵循几何信息，在几何简单时则更多依赖参考图像细节，从而平衡几何一致性与纹理细节的生成质量。

### Material/Texture

#### MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion

本文聚焦于多视图PBR材质生成，核心贡献在于引入一致性正则化训练策略与双通道材质生成框架。一致性正则化训练通过参考图像对训练，强制不同视角和光照条件下生成的扩散噪声接近，以此提升模型对视角扰动的鲁棒性，解决视图敏感和光照纠缠问题，实现光照不变且精确的PBR贴图生成。

双通道材质生成框架包含多通道对齐注意力与可学习材质嵌入，前者对反照率和金属粗糙度通道独立优化，反照率通道用参考引导交叉注意力保证生成符合参考图像，金属粗糙度通道通过残差连接从反照率特征继承空间连贯性，后者为两通道引入可训练嵌入，捕捉纹理独特特征，从而生成高质量、连贯且无伪影的纹理。多视图扩散基于潜在扩散模型，通过多视图注意力实现几何一致的多视图输出同步去噪。PBR材质采用迪士尼BRDF框架，以反照率、金属度和粗糙度参数定义表面，存储于反照率RGB纹理和金属粗糙度组合图。

训练时从包含不同相机姿态和光照条件图像的候选集中选择参考图像对，输入后网络处理图像并通过一致性正则化强制输出等价，以此完成模型优化。

### Shape + Texture

#### Structured 3D Latents for Scalable and Versatile 3D Generation

提出SLAT/trillis框架，构建统一通用的3D潜在空间，同时表达几何与外观信息。其结构化潜在表示将3D资产定义为活跃体素位置索引与局部潜在变量的集合。通过从球体采样图像并利用DINOv2提取视觉特征，经稀疏VAE编码为潜在表示，再解码为多种3D形式：解码为3D高斯时，每个潜在变量生成多个高斯，位置由体素位置加偏移量确定，损失包括重建、体积和不透明度正则化项；解码为辐射场时，采用CP分解表示局部辐射体积；解码为网格时，借助FlexiCubes参数和有符号距离值，通过上采样提升分辨率，损失包含几何、颜色和正则化项。生成结构化潜在变量分两阶段。

第一阶段用VAE和Transformer生成稀疏结构，第二阶段基于用Transformer生成潜在变量。该框架可实现细节变化（保持结构改变潜在变量）和区域编辑（调整目标区域体素与潜变量）。

#### Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation

DiffusionGS 是一种结合 3D 高斯与生成模型的方法，能够同时建模物体和场景。它以一个干净条件视图和若干噪声视图为输入，并结合视点信息，通过前向扩散向噪声视图添加噪声，再由去噪网络预测每像素的 3D 高斯原语。每个高斯包含位置、形状、透明度和颜色信息，通过像素对齐的射线参数化深度和协方差。去噪器使用 Transformer 主干处理图像与视点信息，并通过高斯解码器生成每像素高斯图，再合并成高斯点云。利用可微分 2D 渲染将预测的多视图高斯渲染回图像，通过 L2 损失和感知损失监督去噪训练。在测试阶段，模型从标准正态分布采样噪声，逐步迭代去噪以生成完整的 3D 高斯表示。为了兼顾场景和物体级训练数据，DiffusionGS 设计了混合训练策略，通过控制视点角度和方向来选择条件视图、噪声视图和新视图，并使用 Plücker 坐标及其改进表示像素对齐射线。深度范围差异通过双高斯解码器处理。在训练中，除常规去噪和新视图损失外，还设计了点分布损失来引导物体中心的高斯点云更加集中，从而提升几何和纹理质量。

### Multiview

#### Zero-1-to-3: Zero-shot One Image to 3D Object

模型基于Stable Diffusion进行改进，旨在有效融合条件信息。训练数据需包含四元组 \( (x, \hat{x}_{(R,T)}, R, T) \)，即源图像、目标视角真实图像及其相对位姿。其训练目标是最小化噪声预测误差，具体而言，是让模型预测的噪声 \( \epsilon_{\theta} \) 与真实噪声 \( \epsilon \) 在潜在空间中的L2距离最小。

条件信息的融合通过双分支架构实现：高阶语义分支：将输入图像 \( x \) 的CLIP嵌入向量与相对相机位姿 \( R, T \) 进行拼接，随后通过交叉注意力机制融入到扩散模型中，以提供全局的语义和视角约束。低阶视觉分支：直接将输入图像 \( x \) 与噪声潜在变量在通道维度上进行拼接，为模型提供底层的像素级视觉信息。

该工作可与SDS方法结合，用于辅助三维重建任务。

#### Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model

Zero-1-to-3的升级版本，为了保证不同视角之间的一致性，选择一口气生成不同视角的图像，从原始的scale linear替换成普通的linear，该方法可以让模型在高分辨率的条件下更加注重全局信息，提高训练的效率；使用了reference attention以及global condition的技巧

#### One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization（multiview）

结合Zero123模型从单张图像生成360度三维网格，借鉴并扩展了SparseNeuS框架。

通过2D特征网络提取输入图像的多个2D特征图。随后构建3D cost volume，将每个3D体素投影到各2D特征平面，并计算投影点特征的方差。该代价体经稀疏3D CNN处理后生成几何体，用于编码形状的几何信息。

两个MLP网络分别用于预测3D点的有符号距离函数（SDF）值和颜色，基于SDF的可微分渲染技术用于生成RGB图像和深度图。

为将SparseNeuS从前向重建扩展至全360度物体建模，使用3D物体数据集并冻结Zero123模型。通过球面相机模型将每个形状渲染为n个均匀分布的GT RGB与深度图像。对每个视图，利用Zero123预测其四个邻近视图。训练时输入这4×n个预测图像及对应GT，并随机选取n个GT RGB视图作为目标视图，使模型能够学习处理Zero123的多样化预测并重建一致的全角度网格。

利用Zero123预测输入图像的四个邻近视图；采用由粗到细策略枚举所有可能的elevation；对每个角度计算相机位姿，并通过重投影误差评估图像与位姿的一致性；最终选取重投影误差最小的角度，确定全部4×n个源视图的相机位姿。

#### One-2-3-45++: Fast Single Image to 3D Objects with Consistent Multi-View Generation and 3D Diffusion

从单张图像生成3D网格，在多视角生成一致性方面，方法沿用了Zero-1-2-3++的策略。具体通过引入局部条件与全局条件来增强一致性：局部条件利用参考注意力机制，全局条件则结合CLIP特征与可学习参数。噪声调度采用线性Schedule和v-prediction方式。在微调过程中，首先训练LoRA参数以及Self-Attention和Cross-Attention层，随后对整个网络模型进行端到端的微调。

在3D表示方面，使用SDF Volume和颜色 Volume。SDF体积可根据预设阈值转换为占用网格，从而定义物体的几何形状。3D生成过程借鉴了LAS-Diffusion的两阶段由粗到细扩散策略：第一阶段（粗粒度）：主要预测物体的占用网格，确立大致几何结构；第二阶段（细粒度）：在粗粒度几何基础上，进一步细化预测SDF和颜色体积，以获取更精细的几何与外观。

在扩散过程中，多视角图像的局部信息作为关键条件被引入。使用预训练的DINOv2模型从每个输入的多视角图像中提取2D局部特征。根据已知的相机位姿，将这些2D特征反向投影至3D空间，形成3D特征体。对于每个3D体素，其汇聚了来自多个视角的2D特征，通过一个共享权重的MLP进行处理，并经过最大池化操作来聚合多视角信息，最终得到该体素的融合特征表示。

在颜色渲染方面，该方法采用了TensoRF的渲染方式来完成最终的颜色合成。

#### Free3D: Consistent Novel View Synthesis without 3D Representation

该工作延续 Zero123 思路，针对两大问题进行改进：指定相机位姿时无法准确生成新视角图片，多视角图片存在不统一问题。模型架构沿用 Stable Diffusion，将 Plücker 射线与 LN 结合（类似 DiT 注入方式），通过 Plücker 射线表示法显式指定每个像素对应的位姿，且该表示符合光线传播特性。为保证图像一致性，采用多视角 attention 机制。损失函数包含基于噪声预测的 L2 损失，还额外添加基于 LPIP 的损失以强化一致性，其中通过视角矫正方法处理图像并利用特征距离衡量差异。

#### FlashWorld: High-quality 3D Scene Generation within Seconds

FlashWorld，一种可在数秒内生成高质量3D场景的生成模型。其核心在于结合多视图生成的视觉质量优势与3D生成的几何一致性优势。方法上，首先通过双模式预训练构建一个同时支持两种生成路径的扩散模型；随后，采用跨模式蒸馏策略，以多视图模式为教师、3D高斯溅射（3DGS）模式为学生，通过分布匹配蒸馏（DMD）和跨模式一致性损失，实现高质量与3D一致性的统一。此外，引入单视图图像与文本驱动的分布外协同训练，增强模型泛化能力。实验表明，FlashWorld 在生成质量、一致性和速度（约9秒/场景）上均优于现有方法。

### SDS

#### DreamFusion: Text-to-3D using 2D Diffusion

DreamFusion 的核心在于使用了基于概率密度蒸馏的损失函数，使得 2D 扩散模型可以作为先验来优化参数化图像生成器。随机初始化一个NeRF模型，在每次迭代中，会随机采样一个相机视角和光照条件，将当前的3D NeRF模型渲染成一张2D图像。随后，这张渲染图像被送入一个参数被冻结的2D扩散模型（如Imagen），结合文本prompt，输出生成图像。SDS损失的关键在于，它并不直接使用扩散模型去噪后的图像，而是通过计算其引导下的梯度来更新NeRF的参数。

#### Make-It-3D: High-Fidelity 3D Creation from A Single Image with Diffusion Prior

可视为 DreamGaussian 前身，利用 2D 生成模型先验生成 3D 模型，采用 Instant NGP 作为3D生成模型，分两阶段优化：Coarse 阶段：通过参考图像颜色监督、SD 的 SDS 损失训练，针对生成图与图像对齐问题引入 CLIP 损失（仅在小步长开启），还使用负皮尔森系数及深度正则化。Refine 阶段：针对第一阶段几何合理但纹理粗糙的问题，将 NeRF 转换为点云表示。为解决多视角投影导致的点云噪声（视角冲突），提出迭代策略：从参考视图构建初始点云，投影新视角时用掩码标记已覆盖区域，仅提升未观察点并集成到稠密点云。采用延迟点云渲染优化纹理质量，为每个点优化 19 维描述符，引入扩散先验约束新视角渲染；通过多尺度延迟渲染方案，将点云多次光栅化生成不同尺度特征图，拼接后经 U-Net 渲染器得到最终图像，损失函数与 Coarse 阶段相同。

#### DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation

将 Gaussian splatting 与 SDS 结合的方法，可大幅降低 3D 模型生成时间，同时提出高效 mesh 提取及 UV 空间微调策略以提升生成质量。模型框架中，Gaussian splatting 仅保留物体颜色等必要参数，初始化时用球体内随机位置采样的 3D 高斯，周期性 densify 且从较少高斯开始。训练时采样随机相机位姿渲染图像，线性减少时间步以添加随机噪声，借助二维扩散先验通过 SDS 优化三维高斯分布。图生 3D 时输入图片与前景 mask，结合 SDS 损失和参考图像损失；文生 3D 则利用 CLIP embedding。

针对生成高斯模糊问题，微调阶段提出 local density query 分块查询法，先将 3D 空间分块，在小块中整合透明度，再结合 marching cube 与阈值提取 mesh，用 nvdiffrast 找 uv 坐标后，从多个视角渲染图像反投影到纹理图。微调 uv 空间时借鉴 SDEdit 思路，给 uv 纹理添加噪声后用多步扩散还原，再通过生成图片优化初始纹理。

#### DreamTime: An Improved Optimization Strategy for Diffusion-Guided 3D Generation

作者认为SDS中时间步的选择应随训练阶段变化，不应采用均匀采样方式，为此提出“Weighted non-increasing t-sampling”方法。该方法通过特定方式控制时间t的下降速度，W越大下降越平缓，反之则越快。其中W由两部分组成，第一项反映扩散模型的训练过程，第二项反映时间的侧重点，两者经处理后构成最终的W值。

#### LucidDreamer: Towards High-Fidelity Text-to-3D Generation via Interval Score Matching（SDS）

为解决 SDS 生成结果过饱和的问题，作者提出 ISM 方法并结合 DDIM 反演。SDS 存在输入波动显著影响伪真实值、噪声和相机姿态随机性导致波动、优化中伪真实值不一致造成特征平均化、单步预测生成的伪真实值缺乏细节或模糊等问题。引入 DDIM 反演通过迭代预测可逆噪声潜在轨迹，产生与$x_0$对齐的更一致伪 GT。ISM 将潜在单步预测改为多步预测以提升伪真值图片质量，得到相应的损失表达式，虽迭代方法提升质量但耗时，经变量代换并舍去特定项后得到最终表达式，形成整体算法流程与模型架构。

#### Magic3D: High-Resolution Text-to-3D Content Creation

为解决SDS中渲染精度和图片分辨率较低的问题，采用由粗到细的两阶段优化策略。

第一阶段用eDiff-I在64分辨率下渲染，结合Instant NGP而非计算量大的Mip-Nerf 360，通过两个MLP分别预测反射率/密度和法线，利用基于密度的体素裁剪与八叉树射线采样跳过空白空间，初始化并定期更新占用网格生成八叉树，用MLP预测法线避免计算开销，同时用环境映射MLP表示背景并控制学习率以聚焦神经场优化。

第二阶段为提高高分辨率渲染效率，使用带纹理的3D网格作为场景表示，借助第一阶段预训练权重回避拓扑变换难题，具体用可变形四面体网格表示3D形状，顶点包含符号距离场值和变形量，通过可微算法从符号距离场提取表面网格，以神经颜色场作为体积纹理。从神经场初始化的密度场转换为符号距离场得到初始值，体积纹理字段用粗略阶段优化的颜色场初始化，优化中通过可微分光栅化器渲染高分辨率图像并反向传播优化顶点参数，追踪像素对应3D坐标查询纹理颜色联合优化，增加焦距放大细节，保持预训练环境图并合成前景背景，对网格相邻面角度差异正则化以确保表面光滑和几何稳定。

#### ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation

VDS 是基于 SDS 提出的方法，从变分和粒子流角度改进，解决过饱和、过渡平滑及细节不突出问题。SDS 优化目标复杂，直接求解困难，VDS 参照扩散过程重新定义变分推断过程，引入受粒子变分推断启发的变分分数蒸馏更新规则，通过模拟常微分方程近似采样。真实图像噪声得分用预训练扩散模型估计，渲染图像噪声得分由在渲染图像上训练的噪声预测网络估计（这里是核心），该网络参数化方式有小型 U-Net 和预训练模型 LoRA，实验发现 LoRA 效果好。VDS 与 SDS 不同，SDS 是 VSD 特例，VSD 可用多套模型参数，单参数时也学习参数化得分函数，且通过 LoRA 能利用文本提示信息，而 SDS 中高斯噪声无法利用该信息。

#### ReconFusion: 3D Reconstruction with Diffusion Priors

通过扩散先验提升基于NeRF的 3D 重建效果，实现仅用少量照片就能重建高质量 3D 场景。整体框架分为新视图合成和 3D 重建两部分。新视图合成时，利用扩散模型学习给定输入图像及对应相机参数、目标新视图相机参数下的新视图图像条件分布，借助 PixelNeRF 模型生成特征图，并与噪声潜在表示结合送入生成模型。3D 重建阶段，使用loss使渲染图像与观察图像尽可能相似，同时将扩散模型的先验融入 3D 模型，让新视图渲染更贴近真实场景。

#### Score Distillation via Reparametrized DDIM

与VDS目标一致，旨在解决SDS生成的3D模型过饱和和过于卡通化的问题，但切入点不同。本文认为问题的根源在于SDS训练过程中方差过大，并指出SDS本质上是方差较大的一种DDIM。考虑直接用DDIM替代SDS，但面临一个关键问题：去噪网络期望输入图像的噪声水平与时间步t相关，而3D渲染图像是从模糊逐渐变清晰的，导致噪声水平不匹配。

为此，引入了一个新变量 \( x_0(t) \)，表示通过一步去噪预测得到的图像，其演变过程与3D渲染从模糊到清晰的过程相似。基于此，作者将DDIM公式重新参数化为关于 \( x_0(t) \) 的形式，并推导出一个新的更新公式。新公式依赖于 \( x(t) \)，而如何从 \( x_0(t) \) 一致地反推 \( x(t) \) 并不明确。作者通过定义方程 \( \kappa_y^t(x_0(t)) = \epsilon \) 来解决这个问题，从而将更新公式转化为仅依赖于 \( x_0(t) \) 和预测噪声的形式。作者提出通过DDIM反转来近似求解 \( \kappa_y^t \)，即在文本条件y的指导下，通过反向积分ODE从图像反推噪声。该方法在算法中引入了一个小的随机扰动H(t)以提升效果。SDS可被视为该框架下假设噪声为高斯分布的一个特例。

### Worl model

#### HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D Worlds from Words or Pixels

分阶段生成框架，先以扩散模型生成全景图初始化世界，再进行分层与重建。生成全景图时，文本条件下用LLM翻译并增强中文输入以匹配训练数据风格，图像条件下通过相机内参估计和ERP投影生成完整360°图，基于Panorama-DiT模型生成全景图，还引入高程感知增强和环形去噪策略缓解几何失真与边界不连续问题。数据处理时对全景图进行质量评估和人工筛选，通过三阶段流程生成训练caption，图像条件下用场景感知提示生成策略避免物体重复。为实现自由探索，将场景分解为天空层、背景层和物体层，通过实例识别、环形填充预处理后的层分解及自回归“洋葱剥层”实现层补全。分层重建包括层对齐深度估计和分层3D生成，前景物体可直接投影或用图像到3D模型生成，背景层经深度压缩后平面扭曲，天空层支持HDRI贴图和3D高斯溅射表示。长距离扩展借助Voyager模型，通过世界一致性视频扩散和自回归场景扩展合成连贯RGB-D视频。系统优化上，离线用多阶段流程压缩网格，在线采用Draco压缩，推理时利用TensorRT框架结合缓存和多GPU并行加速。

### 4D

#### PartRM: Modeling Part-Level Dynamics with Large Cross-State Reconstruction Model

现有方法如Puppet-Master基于2D视频表示微调预训练视频扩散模型预测物体后续状态，存在输出单视图视频、处理慢等问题。作者提出PartRM方法，从3D重建角度出发，并构建数据集PartDrag-4D。 PartDrag-4D数据集旨在解决现有数据集3D数据缺失或不符合运动学动力学的问题。它基于PartNet-Mobility构建，选取738个网格，对每个网格的关节部件进行动画处理，生成20,548个状态，还为每个3D网格状态生成12个视图，并处理拖动采样点。 PartRM方法输入单视图观测和输入拖动，目标是生成3D高斯表示。在处理中，利用Zero123++生成并微调多视图图像，通过拖动传播模块利用关节运动学先验增强输入拖动条件，基于预训练LGM构建PartRM并进行drag embedding 。训练采用两阶段流程，第一阶段用知识蒸馏法专注学习动作，利用溅射图像优势，通过对2D图像对应像素的高斯参数应用L2损失，让学生网络学习预训练网络从目标观测视图生成的高斯；第二阶段用目标观测图片替换监督信号，联合优化外观、几何和部件级运动，通过渲染新视图计算多种损失函数实现。 

## Geometry

### Point

#### PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation

核心在于通过多层感知机（MLP）提取每个点的特征，同时设计对称函数（如最大池化）聚合全局特征以应对点云无序性。为解决点云变换不变性问题，引入两个关键模块：一是空间变换网络（STN），通过学习仿射变换矩阵对齐输入点云坐标；二是特征变换网络，对提取的点特征进行正交变换以增强特征稳定性。网络整体采用编码器 - 解码器结构，编码器通过 MLP 和最大池化生成全局特征用于分类，解码器则结合全局与局部特征实现逐点分割。

#### DUSt3R: Geometric 3D Vision Made Easy

该 3D 重建方法在 MVS 中引入点云表示，无需先验信息，可松耦合提取经典 SFM 和 MVS 流程的中间输出。Pointmap 与 RGB 图像一一对应，可通过相机内参和深度图在相机坐标系中转换。网络架构受 CroCo 启发，包含两个相同分支，每个分支有图像编码器、带交叉注意力的 Transformer 解码器和回归头，输入两张图像后输出点图和置信图。训练目标包括通过计算平均距离解决点图尺度模糊的 3D 回归损失，以及结合置信度和回归损失的置信度感知损失。下游应用涵盖点匹配（通过 3D 点图最近邻搜索建立对应关系）、恢复内参（优化估计焦距）、相对姿态估计（Procrustes 对齐或 RANSAC 结合 PnP）、绝对姿态估计（结合内参和对应关系运行 PnP-RANSAC）和全局对齐（构建连接图后优化点图、姿态和缩放实现场景全局对齐）。

#### MUSt3R: Multi-view Network for Stereo 3D Reconstruction

DUST3r在处理多视图重建时因全局对齐计算量大存在局限，作者提出的MUSt3R通过对称架构扩展实现多视角3D结构重建，并引入多层内存机制降低计算复杂度以适配大型图像集。其多视图架构采用共享权重的Siamese解码器和头，确定参考图像后添加可学习嵌入并去除交叉注意力中的旋转位置嵌入，通过改变解码器块交叉注意力计算方式，使每个图像令牌与其他所有图像令牌交互。为简化相机内参估计，预测头额外输出点图，通过Procrustes分析估计变换。面对大量图像计算难题，采用迭代内存更新机制，新图像与保存的令牌交叉注意力计算后更新内存；引入全局3D反馈，用终层信息增强早期层内存令牌以传播全局3D知识。内存管理分在线和离线场景：在线场景基于视频流中帧的发现率（通过KD树计算像素到当前场景距离的分位数）决定是否保留预测并更新3D场景；离线场景用ASMK检索和最远点采样选择关键帧，按相似度排序后构建场景潜在表示并渲染。

#### Point3R: Streaming 3D Reconstruction with Explicit Spatial Pointer Memory

该方法提出一种流式三维重建建模方法，显式表征memory特征位置，基于内存的流程为$X_t = \mathcal{F}( \mathcal{M}_{t-1}, I_t )$，其中$\mathcal{M}$是显式空间指针内存，存储三维指针及关联的空间特征。图像编码器用ViT将输入图像编码为图像令牌$F_t$。交互解码器中，空间指针内存由三维位置$P$和空间特征$M$组成，用两个交织解码器实现图像令牌与内存交互，得到更新后的图像token和姿态token$z_t'$，并预测点图、置信度图和相机姿态。内存编码器处理当前帧后，利用当前特征和预测点图得到新指针，新指针通过融合机制整合到现有内存中，计算新指针与现有指针欧氏距离，按距离融合或添加。将旋转位置嵌入扩展到3D层级位置嵌入，用不同频率基生成层级旋转矩阵，注入相对位置信息，图像令牌3D位置由前一帧全局点图确定。训练目标对姿态用L2范数损失，对点图用置信度感知损失，姿态参数化为四元数和平移向量，点图损失结合置信度和欧氏距离。

#### Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors

该模型基于DUSt3R框架改进，可灵活融合额外模态增强性能。编码器采用共享ViT以Siamese方式编码输入图像及其辅助信息（相机内参、深度图等），解码器有两个，通过交叉注意力机制通信并在模块中融入相对姿态等辅助信息，头部回归出点图和置信图。损失函数包括3D回归损失和置信感知损失，最终损失为各部分加权和。通过专用MLP嵌入相机内参、深度图和相对姿态等模态信息，嵌入策略采用“inject - 1”。下游任务中，深度图提取可一次完成且能高分辨率处理，还可恢复焦距、估计相对姿态及进行全局对齐。训练时输入带随机辅助信息子集的注释图像，随机选择模态、裁剪数据，使用多个数据集在GPU上训练。

#### 3D Reconstruction with Spatial Memory

Spann3R本质上是Dust3R与memory based的方法融合，使得模型能够更高效的在全局坐标系中预测点云信息；memory的机制同样借鉴了XMem的思路：working memory选取5张frame 先进先出原则，并且确保相似程度小于0.95，旧的frame会被存放到long-term memory；long-term memory 存储到一定阈值会被执行稀疏化操作

#### Continuous 3D Perception Model with Persistent State

CUT3R是流式3D重建模型，其以图像流为输入且无需相机信息，每张图像同时执行状态更新和读取，输出相机参数和世界坐标系点图并累积为密集重建。 状态-输入交互机制中，当前图像经ViT编码器编码为token，状态也表示为token并初始化为可学习token。图像token与状态通过两个相互连接的Transformer解码器双向交互，图像token前有可学习“姿态token”，交互后从富集状态信息的图像token和姿态token输出中提取显式3D表示，预测自身坐标系和世界坐标系的点图及置信度图，还有自身运动。 用未见过的视图查询状态时，虚拟相机内参和外参用射线图表示，经单独Transformer编码器编码后与当前状态交互读取信息，不更新状态，用头部网络解析为3D表示和颜色信息。 训练目标包括3D回归损失、姿态损失和RGB损失，分别对不同输出进行监督。

### Framework

#### TorchSparse: Efficient Point Cloud Inference Engine

### depth

#### Affine-Invariant Depth

#### Learning to Recover 3D Scene Shape from a Single Image

该方法采用两阶段流程，由深度预测模块（DPM）和点云模块（PCM）组成。DPM从单张RGB图像预测深度图，但其结果存在未知的尺度和偏移；PCM以此生成的点云为输入，通过学习恢复深度偏移与焦距，从而修正3D几何结构。点云由针孔相机模型反投影获得，焦距影响点云的横向尺度，深度偏移则导致非均匀形变。为估计这两个参数，采用PVCNN对扰动点云进行学习，通过合成具有随机偏移与焦距扰动的训练样本进行监督。推理时，PCM利用DPM预测的深度生成初始点云，并输出修正后的深度和焦距比例因子，实现自适应几何校正。

在深度预测阶段，提出图像级归一化回归（ILNR）损失以缓解不同数据集间的尺度差异，并引入成对法线回归（PWN）损失以增强局部几何一致性。ILNR结合tanh归一化与修剪的Z-score，使网络在仿射不变空间中学习深度；PWN通过对成对点法线关系的约束，提升平面与边缘区域的结构保真度。同时结合多尺度梯度损失（MSG）改善细节还原。综合损失由ILNR、PWN与MSG组成，用于稳定且几何一致的深度学习。

#### Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data

主要聚焦于如何利用数据集： 第一阶段：训练有标签数据集阶段；为了能够让多个数据集在统一的尺度下进行训练affine-invariant loss， 训练过程中使用DINOv2的权重进行初始化 。第二阶段：训练收集到的无标签数据，如果用训练好的教师模型进行预测伪标签并且使用重新初始化的权重将所有数据集一起放进去训练，效果并不理想。在训练期间向未标记的图像注入强烈的扰动（color distorting Gaussian blur 以及cut mix）。 第三阶段语义信息的协助： 首先使用RAM+GroundingDINO+HQ-SAM对大量数据集进行语义标注（得到了4k类信息），利用DINOv2强大的语义表征能力，在特征图上利用余弦相似度进行监督。 
$$
\mathcal{L}_{feat} = 1 - \frac{1}{HW} \sum_{i=1}^{HW} \cos(f_i, f'_i),
$$
当相似度相差太大的情况下便不再强迫模型继续约束。

#### Depth Anything V2

目前现有数据集的噪声很严重，使用生成数据分布上会有偏移，并且所覆盖的场景非常单一，转而利用生成数据集上训练出来的模型在无标签数据集上生成伪标签，进行训练。之后可以进一步对教师网络进行蒸馏（其实就是用伪标签进行训练）。除此之外还是介绍了构造新的数据集的流程。

#### Pixel-Perfect Depth with Semantics-Prompted Diffusion Transformers

Diffusion-based 深度估计方法通常依赖 VAE 进行特征压缩，但这会导致边缘与细节区域出现噪声。为此，本文提出在像素空间中直接进行扩散建模，并引入语义引导模块与级联式 DiT架构以缓解收敛性和细节退化问题。

模型基于 Semantics-Prompted Diffusion Transformers (DiT)构建，完全摒弃卷积结构。给定噪声样本与条件图像后，模型在像素空间执行扩散预测。为了提升全局语义一致性与细节质量，引入来自预训练视觉基础模型的高层语义特征，并通过多层感知机与双线性插值进行融合，使扩散过程在语义提示下更具结构感与细节辨识度。此外，提出的 Cascade DiT 设计利用块大小分级策略：早期层以较大 patch 关注全局结构，后期层以更小 patch 强化局部细节，从而兼顾计算效率与细粒度恢复。最终，通过分层 MLP 重构至像素级深度图。深度预测结果进一步采用对数尺度与分位归一化处理，以在室内外场景中保持数值分布稳定与范围均衡。

#### **Metric Depth**:

#### Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image

度量模糊性问题源于相机焦距的未知。当使用不同焦距或不同相机拍摄时，即使图像外观相似，实际拍摄位置也可能差异很大，因此若焦距未知，单张图像的度量深度估计将是不适定问题。为此，许多方法假设训练和测试图像的焦距已知，或转而学习仿射不变深度以避免度量模糊性。焦距对度量深度恢复起决定性作用，而传感器和像素尺寸的影响较小。

为解决不同相机参数带来的尺度不一致问题，提出了标准相机变换。其核心思想是将所有训练样本转换到一个统一的标准相机空间（焦距固定为$f^c$）。一种方法是对深度标签进行缩放，使其与标准相机对应；另一种方法是对输入图像进行缩放并调整主点位置，从而将原始相机模型映射为标准模型。两种方式均可在训练和推理阶段通过相应的逆变换恢复度量信息。随机裁剪操作仅改变视场角和光心位置，不会引入新的度量模糊性。

在损失设计上，提出了RPNL，以增强局部深度对比度并缓解全局归一化造成的细节压缩问题。其做法是从预测深度与真值深度中随机裁剪多个图像块，并对每个局部区域进行中值绝对偏差归一化，以保持局部结构一致性。除此之外，还结合了尺度不变对数损失（SILog）、像素对法线回归损失（PWN）和虚拟法线损失（VNL）等，以进一步提升深度估计的精度与稳定性。

#### Metric3Dv2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation

在Metric3D的基础上引入了法向量预测分支，实现了深度与法线的端到端联合优化。整体框架采用循环块结构，通过ConvGRU模块与两个预测头对低分辨率深度和未归一化法向量进行迭代更新，并在每步中同时优化隐藏特征，实现深度与法线信息的交互式融合。经过多步循环后，得到优化后的深度与法线预测结果，再通过上采样和后处理生成最终输出。训练阶段使用联合监督损失，深度部分包括尺度不变对数损失、虚拟法向量损失、成对法向量回归损失及基于局部统计归一化的RPNL损失，以提升结构一致性与局部对比度；法线部分在具备真实标签时采用不确定性监督损失，在无标签场景下通过深度-法向一致性损失进行间接约束。最终的总损失以加权形式联合深度、法线及一致性三类约束，从而实现深度与法线的协同优化与几何一致性增强。

#### UniDepth: Universal Monocular Metric Depth Estimation

UniDepth 旨在构建一个无需相机内参即可进行单目度量深度估计的通用模型，其核心思想是在缺乏外部标定信息的情况下实现对3D空间的准确恢复。传统方法若直接从图像预测3D点，会忽略透视几何等先验，导致模型需从数据中重新学习投影规律。UniDepth 引入伪球面 3D 表示，将每个像素的三维输出定义在方位角、仰角和对数深度空间中，从而将相机方向与深度解耦。相机方向由方位角与仰角张量构成，并进一步通过球面谐波编码（SHE）嵌入，实现可微的密集相机表示。

模型由编码器、相机模块和深度模块组成。编码器可采用 CNN 或 ViT 结构，用于提取多尺度特征；相机模块根据这些特征预测焦距与主点等相机参数，并生成密集的角度表示。该表示经 SHE 处理后作为提示传递给深度模块，后者通过交叉注意力层实现条件约束，使深度预测在几何上与相机结构保持一致。为防止相机分支梯度主导整体优化，特征输入时会进行梯度分离操作。在训练中，UniDepth 引入几何不变性约束，确保同一场景在不同相机视角或几何增强下的空间特征保持一致。具体做法是在随机缩放和平移增强下对同一场景预测的深度特征进行配准，并最小化它们之间的差异，从而提升模型的跨相机泛化能力。最终输出在 3D 空间中定义为角度与深度的拼接形式，通过指数映射得到线性深度。优化时采用结合方差与均值项的加权误差损失，对角度与深度分量分别施加尺度与平移不变约束，并与几何一致性损失联合训练。

#### UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler

改进版 UniDepth 在原始框架基础上优化了网络结构与损失设计，引入边缘引导归一化损失（Edge-Guided Normalized Loss）以提升深度预测的局部精度。该损失聚焦于RGB梯度前5%的高对比度区域，通过局部归一化对比预测与真值深度，强调形状一致性而非绝对值，从而增强对真实几何边缘的感知。网络采用ViT编码器生成多尺度特征，并利用相机模块提取针孔相机参数，经正弦编码形成相机嵌入向量，以此调制FPN式解码器输出对数深度。优化部分在重构的MSE损失基础上，引入深度一致性约束、边缘归一化项与不确定性监督。

#### UniK3D: Universal Camera Monocular 3D Estimation

对 UniDepth 的泛化扩展，旨在构建可适配任意相机模型的通用三维场景估计框架。其核心思想是将相机参数与场景几何结构解耦，避免依赖针孔或等矩形等特定相机假设。UniK3D 采用基于球坐标的光线表示，将相机方向编码为极角与方位角张量 $C = \theta | \phi$，并通过球谐函数逆变换从系数张量 $H$ 重建角度场，从而以紧凑形式捕捉光线的空间分布。

针对数据集偏向小视场角的问题，模型提出非对称角度损失（Asymmetric Angular Loss），利用分位数回归机制在训练中强调大角度区域的学习，从而提升对广角场景的表示能力。

网络结构由编码器（Encoder）、角度模块（Angular Module）与径向模块（Radial Module）组成：编码器提取密集特征与类别令牌；角度模块从令牌中预测球谐系数及视场定义域，重建球面光线场；径向模块以此为条件预测半径与置信度，并通过球坐标到笛卡尔坐标变换生成三维点云。UniK3D 通过静态相机光线编码与课程学习策略强化外部约束，使模型在训练初期依赖真实相机参数、后期逐步过渡到预测参数，并在梯度层面隔离相机分支，防止特征混淆。

### Multi Output

#### VGGT: Visual Geometry Grounded Transformer

直接从一个、几个或数百个视角推断场景的所有关键3D属性，实现3D视觉任务的统一处理，可得到摄像机参数、点图、深度图和3D点轨迹等。其输入是观察同一3D场景的N个RGB图像序列，通过Transformer映射到对应的3D属性。相机参数包含内参和外参，采用特定参数化方法，点图在第一个相机坐标系中定义，具有视点不变性。轨迹不由Transformer直接输出，而是输出相关特征，由单独模块处理，且两个网络联合端到端训练。 输入序列中第一个图像作为参考帧，网络对除第一帧外的所有帧具有排列等变性。训练时让模型显式预测深度图能提升性能，推理时结合独立估计的深度图和相机参数可得到更准确的3D点。 每个输入图像先通过DINO补丁化为标记，所有帧的图像标记组合后经主要网络结构处理，交替使用帧级和全局自注意力层，默认使用24层。对于每个输入图像，用额外标记增强图像标记，传递到AA Transformer产生输出标记，第一帧和其他帧的相关标记设置不同，以区分并在第一台相机坐标系中表示3D预测。 相机参数由输出相机标记通过额外自注意力层和线性层预测。输出图像标记经处理后用于预测密集输出，包括深度图、点图、跟踪特征及相关不确定性。跟踪采用CoTracker2架构，以密集跟踪特征为输入，可应用于任何输入图像集。 训练使用多任务损失，包括相机、深度、点图和跟踪损失，其中跟踪损失权重较低。相机损失用Huber损失比较预测与真实相机参数，深度和点图损失结合不确定性进行权衡，跟踪损失计算预测与真实对应点的距离，并应用可见性损失。 通过规范化数据消除歧义，将所有量表示在第一台相机坐标系中，用点图中3D点到原点的平均距离规范化相关量，且让Transformer从训练数据中学习规范化。模型存在局限性，不支持鱼眼或全景图像，在极端输入旋转和大量非刚性变形情况下性能会下降。

#### VGGT-Long: Chunk it, Loop it, Align it -- Pushing VGGT's Limits on Kilometer-scale Long RGB Sequences

针对基于 Transformer 的 3D 视觉基础模型在长序列 RGB 重建中存在的内存瓶颈与漂移累积问题，提出“分块处理—块间对齐—回环优化”三阶段框架，实现无相机标定、千米级户外场景的单目三维重建。将输入长序列 $I={I_1,\ldots,I_N}$ 按重叠方式划分为 $K$ 个块，每块包含 $L$ 帧、重叠 $O$ 帧： $C_k:\ [(k-1)(L-O),\ (k-1)(L-O)+L]$。 每个块独立输入 VGGT，输出带置信度的 3D 点云 $P_k\in\mathbb{R}^{H\times W\times3}$ 与局部相机位姿，其中置信度反映静态与动态区域的重建可靠性。

针对相邻块 $C_k$ 与 $C_{k+1}$，在重叠区域内筛选高置信度对应点 $(p_k^i, p_{k+1}^i)$，通过迭代加权最小二乘（IRLS）与 Huber 损失求解相对变换：

$$
S_{k,k+1}^*=\arg\min_{S\in\text{Sim}(3)}\sum_i\rho(||p_k^i-Sp_{k+1}^i||_2)
$$

 权重定义为 $w_i^{(t)}=c_i\cdot\frac{\rho'(r_i^{(t)})}{r_i^{(t)}}$，低置信点通过过滤与衰减降低影响。

基于 DINOv2 的视觉位置识别模型提取全局图像描述子，通过相似度检索与非极大值抑制获得高置信度回环候选 $(I_i,I_j)$。随后构建“回环中心块”生成稳健的桥接点云，并计算回环变换：

$$
S_{ji}=S_{j,\text{loop}}\circ S_{i,\text{loop}}^{-1}
$$

 最终，构建全局 Sim(3) 李群优化问题，联合最小化相邻块约束与回环约束。

#### FastVGGT: Training-Free Acceleration of Visual Geometry Transformer

在FastVGGT中，为了加速推理与训练并减少冗余计算，引入了基于ToMeSD的Token融合策略。方法将Token分为目标Token、源Token与显著Token三类：目标Token作为代表性锚点参与全局注意力计算，源Token则根据特征相似度融合到最相似的目标Token中以减少计算量，而显著Token用于保持跨视图几何一致性，直接参与注意力计算以保证重建稳定性。具体流程中，第一帧的所有Token被指定为目标Token以建立全局参考；随后在各帧中固定比例地选取显著Token保留精细细节，其余Token则通过基于区域的随机采样在目标与源Token间均匀分配。融合阶段，源Token按余弦相似度与目标Token匹配并进行平均融合，以压缩注意力输入；在计算完成后，通过复制融合特征恢复至原始序列长度，实现“解融合”，保证解码阶段的密集预测精度。该机制在显著降低计算复杂度的同时，兼顾了几何一致性与重建质量。

#### Streaming 4D Visual Geometry Transformer

以 VGGT 为基础，将其中全局注意力替换为因果注意力，从而实现仅依赖历史帧与当前帧的时序建模。然而，这种因果约束可能导致误差在时间上逐步累积。为缓解这一问题，引入了 知识蒸馏策略，利用预训练的非因果教师模型输出作为伪真值，指导学生模型的训练，使其在仅访问有限帧信息的情况下仍能保持全局一致性与准确性。

#### VGGT-X: When VGGT Meets Dense Novel View Synthesis

为了解决当前3D基础模型在密集图像场景中难以建模的问题（主要表现为显存占用过高和输出噪声较大），本文基于VGGT进行了结构性改进，并结合3DGS-MCMC提升了噪声鲁棒性与优化稳定性。3DGS-MCMC通过将高斯渲染的优化过程重构为随机梯度朗之万动力学，使模型在优化中引入受控随机扰动，从而在渲染保真度与对噪声初始化的鲁棒性之间取得平衡。为提高大规模图像输入下的可扩展性，本文提出了内存高效的VGGT实现方案。其网络由基于DINO的帧级特征提取器、交替全局与逐帧注意力的Transformer结构以及多任务解码器组成。通过舍弃中间层输出、采用BF16精度计算及分块异步处理策略，显著降低了VRAM占用并提升推理吞吐量。

针对相机参数估计不稳定的问题，引入了全局对齐（GA）模块，通过极线几何约束优化多视角间的姿态一致性。考虑到匹配噪声对优化的影响，进一步提出自适应加权策略，根据极线距离分布自适应调整匹配点权重，并结合基于中值误差的学习率调节，实现从粗到精的姿态优化。在3D重建阶段，结合3DGS-MCMC的鲁棒优化特性，采用联合优化机制对高斯参数与残差相机姿态进行共同更新，从而在存在姿态误差的条件下仍能获得稳定且高保真的重建结果。

#### MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision

MoGe 的核心在于通过仿射不变点图表示和优化监督策略，解决单目几何估计中焦距—深度模糊和局部几何缺失的问题，并借助大规模混合数据提升泛化能力。模型从输入图像直接预测3D点图，输出坐标与图像XY轴对齐，预测结果在尺度和平移上保持仿射不变，以保证在焦距不确定时的几何一致性。通过优化投影误差，可以从点图中快速恢复相机的焦距和Z轴偏移，实现尺度不变的几何重建。训练时，除了对全局点图进行对齐监督外，还引入多尺度局部几何损失，以提升局部结构的精度。同时加入法向量约束以增强表面质量，并使用掩码损失处理天空等无几何区域，从而实现鲁棒的单目三维几何估计。

#### MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details

MoGe-2 在 MoGe 的基础上，进一步解决了单目几何估计中缺乏 metric 尺度和细节丢失的问题。它将尺度预测与相对几何解耦：保留仿射不变点云分支以维持相对几何精度，同时利用 DINOv2 的 CLS 令牌通过 MLP 单独预测尺度因子，避免对几何学习的干扰，从而解决焦距模糊带来的尺度不确定性。

针对真实数据存在噪声与缺失的问题，MoGe-2 设计了精修流水线。用在合成数据上训练的模型对真实图像进行推理，通过局部对齐方式检测并剔除深度异常值，避免因全局偏差导致的过滤错误。在保留真实有效深度的基础上，利用预测的细节结构，对缺失区域采用对数空间泊松补全，实现局部细节与整体尺度的兼顾。

相比 MoGe，MoGe-2 通过解耦尺度学习和真实数据精修，既保持了相对几何的精度，又补上了 metric 尺度与细粒度结构的短板，实现更高质量的单目 3D 几何重建。

### Pose

#### PoseDiffusion: Solving Pose Estimation via Diffusion-aided Bundle Adjustment

PoseDiffusion通过扩散模型实现扩散辅助光束平差（Diffusion-aided Bundle Adjustment），在给定图像集$I$时，直接对相机参数的条件分布$p(x|I)$进行建模。首先在包含真实图像及其相机参数的大规模训练集上训练一个条件扩散去噪网络$D_θ$，使其能够在去噪过程中逐步恢复潜在的真实相机位姿。去噪器采用基于Transformer的结构，以噪声相机参数、扩散时间步及图像特征（由预训练DINO ViT提取）为输入，预测去噪后的相机参数。训练通过最小化去噪误差实现，使模型学习到在不同扩散阶段下的条件分布。在推理阶段，通过对$p_θ(x|I)$的采样获得相机参数估计，相当于完成光束平差任务。由于该分布在最优位姿附近近似为狄拉克分布，从中采样的结果即可作为有效的相机解。

为进一步提高几何精度，PoseDiffusion引入几何引导采样（Geometry-guided Sampling）。通过两视图几何约束（如极线约束）对采样过程进行引导，使估计的相机参数满足图像间的几何一致性。具体地，模型利用图像对之间的2D匹配点，通过Sampson极线误差评估相机兼容性，并在每个采样迭代中通过梯度引导调整去噪器输出，使Sampson误差最小化，从而实现几何一致的位姿估计。

相机参数包括焦距、旋转和平移，旋转以四元数表示，焦距采用指数参数化以保证正值。为避免因SfM重建导致的场景特定坐标歧义，PoseDiffusion在输入阶段对训练数据进行**坐标规范化**：将所有相机位姿转换为相对于随机枢轴相机的相对位姿，并在特征中显式标识该枢轴相机，同时对平移尺度进行归一化。这种规范化策略使模型学习到与具体场景无关的相对几何关系，从而实现稳健的跨场景相机估计。

## Tracker

#### SpatialTrackerV2: 3D Point Tracking Made Easy

该方法将3D点跟踪任务分解为视频深度估计、相机运动估计和目标运动估计三大模块，通过显式分离相机自运动与场景中目标的独立运动，提高动态场景下的跟踪鲁棒性。前端利用改进的DepthAnythingV2模型进行视频深度估计，引入交替注意力时间编码器以兼顾帧内细节与帧间一致性，并通过可学习的Pose/Scale令牌实现深度与位姿的尺度对齐；随后通过可微位姿头预测相机参数，生成由相机运动诱导的初始3D轨迹。后端采用SyncFormer与可微Bundle Adjustment联合优化3D轨迹与相机位姿。SyncFormer为双分支结构，分别在2D与3D空间中建模点的时空关联，通过跨注意力交互保持轨迹一致性，同时估计点的可见性与动态概率。每次迭代后利用加权Procrustes和DirectBA优化相机位姿，形成闭环更新。训练阶段采用异质数据联合训练策略，结合含完整标注的RGB-D数据、仅深度数据及无标注视频，实现前端与后端的分阶段预训练与端到端联合优化，从而在跨场景视频中获得高精度、尺度一致的像素级3D点轨迹。

## Matching

### Geometric

#### GIM: Learning Generalizable Image Matcher From Internet Videos

一种从海量互联网视频中自监督学习图像匹配器的方法，旨在提升匹配器在跨场景和无监督条件下的泛化能力。利用视频的时间一致性自动生成匹配监督信号，无需人工标注或结构先验，通过光流估计、特征提取和一致性筛选获得高质量伪匹配对，并设计了特征匹配网络在大规模多样化视频上进行训练。相比基于SfM或合成数据的方法，GIM在多种匹配与重建任务上显著提升了跨域泛化性能，能在未知场景中稳定输出高质量匹配结果。

#### Grounding Image Matching in 3D with MASt3R

该工作将DUST3r应用于图像特征匹配，旨在解决现有基于关键点匹配在重复模式或低纹理区域易出错、密集匹配视为2D问题的局限。DUSt3R框架通过Transformer网络基于两输入图像预测局部3D重建结果，以两个密集3D点云形式呈现：两图像经Siamese方式由ViT编码，通过交叉注意力交换信息，两预测头从编码器与解码器输出的连接表示中回归最终点图和置信度图。损失函数采用完全监督的回归损失，考虑到如无地图视觉定位等应用需求，修改回归损失以忽略对预测点图的归一化，最终的置信度感知回归损失结合了回归损失与置信度相关项。针对DUSt3R点图在极端视角变化下匹配鲁棒但对应关系不够精确的问题，MASt3R添加匹配预测头，该头由2层MLP和GELU交错组成并接归一化，输出两个密集特征图，利用InfoNCE损失鼓励局部描述符至多只匹配一个，最终训练目标结合回归损失和匹配损失。在匹配算法上，传统互匹配计算复杂度高，快速匹配算法从初始稀疏像素集开始迭代映射收集互匹配对，为处理高分辨率图像，采用粗到细匹配策略，先对下采样图像匹配得到粗对应集，再在全分辨率图像生成重叠窗口裁剪，枚举窗口对并选择覆盖大部分粗对应集的子集，对每个选定窗口对独立匹配后将对应关系映射回原始坐标并连接，实现密集的全分辨率匹配。

### Semantic

#### Dense Semantic Matching with VGGT Prior

语义匹配问题，即在不同实例之间寻找像素级对应关系，现有方法在处理几何模糊性和对称结构时容易出错，且逐像素匹配忽略跨图像不可见性和流形约束，难以保证泛化性。为此，基于VGGT骨干提出改进方案：利用前几层特征块保留几何先验，微调后续层以适应语义匹配任务，并新增匹配头预测双向采样网格和逐像素置信度图。训练中结合循环一致性约束，通过匹配-重建闭环优化像素对应，同时学习匹配不确定性，确保预测置信度与误差相关。为支持密集监督，设计了合成数据生成流程，通过3D资产渲染和多条件图像合成构建带密集标签的配对数据。训练采用渐进式方案：先在合成数据上进行密集监督预训练，再在真实数据上进行稀疏关键点监督，最后加入循环一致性匹配损失和不确定性学习。针对混叠伪影问题，引入平滑损失约束相邻像素采样坐标连贯性，从而提升匹配结果的空间一致性和鲁棒性。

#### MATCHA:Towards Matching Anything

MATCHA 方法融合了 DIFT、DINOv2 与稳定扩散模型的特征优势，用于同时提升几何、语义与时间匹配性能。DIFT 能为不同匹配任务提供有效特征，但需手动选择描述符；DINOv2 具备强语义表达能力，适用于物体级匹配，但缺乏空间细节。MATCHA 通过动态融合两者特征，获得兼具几何精度与语义泛化的统一特征描述符。输入图像 $I$ 经 DIFT 与 DINOv2 提取语义与几何特征 $F_h$、$F_l$、$F_d$，并通过多层自注意力与交叉注意力模块进行动态融合，得到增强后的语义特征 $F_s$ 与几何特征 $F_g$。随后，将两者与 DINOv2 特征拼接形成最终的匹配特征 $F_m$。在训练阶段，MATCHA 采用有限监督优化策略：语义分支结合 CLIP 对比损失与密集语义流损失，以提升语义匹配一致性；几何分支则利用双 softmax 损失在双向相似度矩阵上进行几何约束，从而强化关键点级别的精确匹配。整体上，该框架在不同任务间实现了语义与几何表征的互补融合，显著提升了跨场景匹配的鲁棒性与泛化能力。

## Pretrain

#### CroCo: Self-Supervised Pre-training for 3D Vision Tasks by Cross-View Completion

给定同一场景的两张不同视角图像 $x_1$ 与 $x_2$，将 $x_1$ 划分为 $ N $ 个非重叠补丁，并随机掩蔽其中的补丁。模型需利用剩下的集合与另一张图视图的补丁集合重建被掩蔽部分。

#### CroCo v2: Improved Cross-view Completion Pre-training for Stereo Matching and Optical Flow

主要解决自监督预训练方法在在立体匹配或光流等密集几何视觉任务中表现不好。基于croco进行改进，探索了三种改进途径：大规模收集合适的真实世界图像对；相对位置嵌入，提升视觉 Transformer 性能；扩展基于视觉 Transformer 的架构。


## Other 

#### Convolutional Occupancy Networks

理想的3D几何表示需具备编码复杂几何、适应大场景、整合信息及计算存储高效等特性，而现有体素、点云、网格和隐式表示等方法各有局限，比如体素内存需求大、点云无法表示拓扑关系等。结合CNN与隐式表示的优势，提出更灵活的隐式表示用于3D重建，整体架构可根据特定任务调整。 编码器负责从输入数据（噪声稀疏点云或粗占用网格）提取特征以构建平面/体积特征表示，针对不同输入类型采用不同预处理方式：体素化输入用一层3D CNN，3D点云用带局部池化的浅PointNet网络。其中，平面编码器将输入点投影到规范平面并离散为像素单元，通过平均池化聚合特征得到平面特征，有地面平面投影（计算高效）和三个规范平面投影（恢复z维度丰富几何结构）两种变体；体积编码器能更好表示3D信息但分辨率较低，同样通过平均池化生成特征体积。 解码器利用2D和3D U-Net网络处理编码器输出的特征，且因卷积操作具备平移等变性，模型输出特征也有此特性以支持结构化推理。U-Net网络具体分为处理地面平面特征的单平面解码器（2D U-Net）、用共享权重2D U-Net分别处理各特征平面的多平面解码器，以及用3D U-Net处理特征体积的体积解码器。 占用预测基于聚合后的特征图估计3D空间任意点的占用概率，不同解码器对应不同特征查询方式：单平面解码器将点正投影到平面后双线性插值查询特征，多平面解码器对三个规范平面特征求和聚合信息，体积解码器通过三线性插值获取特征，再借助小型全连接占用网络完成占用预测。 训练阶段，在感兴趣体积内均匀采样查询点，用二元交叉熵损失衡量预测占用值与真实值的差异；推理时运用多分辨率等值面提取（MISE）算法提取网格，且因模型是全卷积的，可通过“滑动窗口”方式处理以实现大场景重建。

#### Dora: Sampling and Benchmarking for 3D Shape Variational Auto-Encoders

在3D VAE学习任务中引入重要性采样，设计锐边采样（SES）算法及双交叉注意力架构，并开发基于锐法线误差（SNE）指标的评估基准。3DShape2VecSet通过泊松盘采样和FPS处理点云，经交叉注意力与自注意力完成特征编码及占用值预测。本文方法则利用SES策略增强均匀采样点云，通过识别三角网格显著边缘并生成显著点集来补充重要点，形成密集点云；编码时对均匀点云和显著点云分别下采样合并后，通过双重交叉注意力计算并相加注意力结果生成潜在代码，训练采用基于占用场监督的均方差损失函数，以此提升点云细节编码与几何重建精度。

#### Flexible Isosurface Extraction for Gradient-Based Mesh Optimization

Flexicube是一种用于可微分网格优化的mesh可微表示方法，其核心基于网格的符号距离函数（SDF），通过Dual Marching Cubes提取三角形网格。它引入三组额外参数与标量函数共同通过自动微分优化以适应目标，还支持提取四面体网格和自适应分辨率的分层网格。 先看其前身方法。Marching Cubes沿网格边缘提取mesh顶点，通过线性插值标量函数改变符号的位置来确定顶点；Dual Contouring在每个3D网格单元中生成顶点再连接成mesh，能捕捉锐利边缘，但可能出现不连续、自相交等问题；Dual Marching Cubes可在一个网格单元中生成多个顶点，使网格更平滑连续，不过若用类似双轮廓法定位顶点，也会有DC方法的问题。 再看Flexicube方法本身。它引入的插值权重$α$和$β$，分别用于调整沿每条边的交叉点位置和每个面内dual vertex的位置，且公式设计为凸组合，使顶点位置在网格单元顶点的凸包内，避免mesh自交；分割权重 $γ$ 控制四边形分割为三角形的方式，优化时插入中点顶点分割，最终提取时沿$γ$值乘积较大的对角线分割；变形向量$δ$允许底层网格顶点位移变形，且限制变形范围不超过网格间距一半，保证网格拓扑稳定。 

在拓展应用方面，拓展到四面体网格时，顶点集合包括网格顶点、在网格单元中提取的mesh顶点等，生成四面体时根据网格边连接的顶点符号是否相同有不同处理方式，同时要处理双行进立方体连接性中单元含多个提取网格顶点时形成四面体需选正确顶点的复杂性；扩展自适应分辨率时，利用自适应分层网格，支持八叉树网格，借鉴Dual Contouring方法适配到FlexiCubes的Dual Marching Cubes扩展中，通过让细化后的八叉树网格顶点与较粗单元相邻时从粗单元面顶点插值获取值来保证符号一致性，处理八叉树中相邻dual vertex跨越不同层次的连接问题。 最后是loss损失函数，包括$\mathcal{L}_{\mathrm{dev}}$和$\mathcal{L}_{\mathrm{sign}}$。$\mathcal{L}_{\mathrm{dev}}$通过惩罚dual vertex与边交叉点之间距离的偏差，确保双顶点位置不偏离边交叉点太远；$\mathcal{L}_{\mathrm{sign}}$通过惩罚隐式函数在网格边上的符号变化，防止在无监督信号区域生成不合理几何形状，保证生成网格拓扑结构合理。

#### Floating No More: Object-Ground Reconstruction from a Single Image

视角 3D 重建的文章，核心旨在解决物体置于平面上重建时出现的 “浮动” 问题，当前 3D 重建模型同样无法准确捕捉物体与地面的真实位置关系。此前相关工作多将相机简化为正交模型，且未考虑物体与地面的位置关联，本文提出从单张图像中同步建模物体、相机参数与地平面，并设计了视场指导的高度映射模块，以高效将估计结果转化为深度图与点云。

文章首先介绍了两种关键表示方法：一是像素高度表示，即定义图像中每个点相对地面的高度，这种方式无需依赖相机参数，还能直接建模物体与地面的位置关系；二是视角场表示，用于编码物体表面每个点的仰角与滚转角度。

在模型流程上，先通过分割任务对 PVT 编码器进行预训练，利用预训练好的编码器预测两个关键密集结果：像素高度（Pixel Height）与视场角参数。将这两项估计结果输入至 “视场指导的像素高度投影模块”，完成最终的 3D 重建。

#### MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers

此前生成模型存在面片数多、后处理繁琐、细节表征不足的问题，其大多直接从图片或文本生成 mesh。因模型缺乏先验位置信息，该过程难度大。本文采用点集表示，依据链式法则推导，提出先得形状再生成 mesh 的思路。

在构建从 3D 模型到点云的数据集时，为避免训练与推理的域偏差，作者未直接对 3D mesh 表面采样，而是先基于 3D 模型生成粗糙 mesh，再据此采样。

训练 VQ-VAE 时，以 mesh 顶点坐标为输入，经 encoder 编码提取，由 VQ-VAE 查询后送入解码器解码。针对训练结果的不足，后续微调 decoder，将形状信息输入解码器，并在 token 采样时加噪，即利用 point cloud 特征提取编码器提取特征，经可学习 MLP 映射后，与查询结果拼接。

#### Neural Kernel Surface Reconstruction

一种基于神经核场（NKF）的新型 3D 隐式表面重建方法，针对传统 NKF 使用全局支持核导致的计算不可扩展和对噪声敏感问题，提出了高效、可扩展、对噪声鲁棒的学习框架。该方法从带法向的点云出发，通过分层体素结构与稀疏神经核场相结合，精准预测三维表面，将重建结果编码为 NKF 的零水平集。

首先利用点与法向量的拼接特征，通过点编码器提取局部特征，再经过稀疏 U-Net 编解码结构预测分层体素网格。结构预测模块决定体素的细分、保留或删除，从而自适应地构建层次结构；法向和核预测分支分别输出法向量和核特征，掩码预测分支负责裁剪远离真实表面的伪几何。基于这些特征定义的层次化 NKF，将形状表示为多尺度局部核的加权组合，避免了全局核带来的开销和噪声放大。通过线性系统求解得到最优系数，实现高效隐式场构建。

训练阶段，模型使用带噪点和密集点的成对数据，不依赖特定类别。通过表面、TSDF、法向、外部区域和最小表面积等多种损失，引导隐式场在表面附近精确逼近零、梯度方向与真实法向一致、远离表面无几何，并对层次结构预测和掩码进行监督，确保结构自适应、伪几何剔除。整体方法充分结合 NKF 的泛化能力与分层结构的高效表示，实现了精细、快速且鲁棒的 3D 表面重建。

#### PoseDiffusion: Solving Pose Estimation via Diffusion-aided Bundle Adjustment

PoseDiffusion 通过扩散模型对图像的相机参数分布进行建模，实现高精度的光束平差。首先在大规模数据集上训练一个 Transformer 去噪器，以图像特征为条件对相机参数的噪声样本进行去噪，从而学习条件分布。推理时，通过采样该分布即可估计相机位姿。由于扩散模型直接回归精确几何量存在局限，PoseDiffusion在采样过程中引入两视图几何约束，利用图像间的匹配点和极线几何，通过 Sampson 误差的梯度引导采样迭代，使生成的相机参数满足几何一致性。

相机参数包括旋转（四元数）、平移和焦距，焦距通过指数映射保证正值。训练数据来源于 SfM，因此位姿存在场景特定的相似变换。为提升泛化性，方法在输入时以随机枢轴相机为参考，对外参进行相对化和尺度归一化，并在图像特征中标记枢轴信息。通过扩散建模与几何引导相结合，PoseDiffusion 既能生成精确的相机参数，又避免了直接优化极线约束易陷入次优的问题。

#### Representing 3D Shapes With 64 Latent Vectors for 3D Diffusion Models

COD-VAE 提出了一种将 3D 形状高效压缩为紧凑 1D 潜在向量集的生成框架，在保持重建质量的同时实现了 16 倍压缩和 20.8 倍速度提升。其基础源于 VecSet，但VecSet对点云的压缩效率和解码速度有限。COD-VAE 在此基础上设计了更高效的多阶段 Transformer 编解码结构，并结合不确定性引导的令牌剪枝策略。

在编码阶段，模型先对输入点云进行采样与位置嵌入，通过交叉注意力与自注意力层分阶段将高分辨率点特征逐步聚合到中间补丁，再压缩到紧凑特征向量（类似于class token），最终通过KL正则化获得1D潜在向量。解码时，先对潜在向量进行通道解压，再用 Transformer 从解压特征重建稠密的 triplane 表示。为了降低 Transformer 的计算量，引入不确定性预测模块，仅保留最具信息量的令牌参与后续计算，从而以较少的特征重建完整的三平面表示。随后利用神经场解码器将重建的 triplane 转换为连续占用场，实现高效的 3D 形状重建。

训练采用二元交叉熵与KL散度联合损失，并通过对不确定性预测与重建误差的监督，使令牌剪枝过程可靠。整体采用两阶段训练：先训练不含潜在模块的自编码器，再在冻结前端的基础上优化KL块和潜在解码器，提升压缩与重建能力。该方法在保证重建精度的前提下显著提升了压缩率与生成速度。

#### Visual Geometry Grounded Deep Structure From Motion

VGGSfM 的目标是通过端到端可微的点追踪网络实现结构从运动（SfM），摆脱传统基于几何算法（如特征匹配 + RANSAC + BA）的非端到端流程。核心思想是在深度网络中显式建模点追踪、相机估计与三角测量，使得整个重建过程可被统一地优化。整体函数形式为：

$$
 f_{\theta}(\mathcal{I}) = (\mathcal{P}, X)
$$

 即网络 $f_\theta$ 接收图像集合 $\mathcal{I}$，直接输出相机参数 $\mathcal{P}$ 与场景点云 $X$，并通过最小化重投影与几何监督的损失 $\mathcal{L}$ 学习参数 $\theta$ 。VGGSfM 将传统 SfM 的四个关键阶段整合为全可微模块：

$$
\begin{aligned}
 \mathcal{T} &= \mathbb{T}(\mathcal{I}) \quad &\text{(点追踪)}\\
 \hat{\mathcal{P}} &= \mathfrak{T}*{\mathcal{P}}(\mathcal{I}, \mathcal{T}) \quad &\text{(相机初始化)}\\
 \hat{X} &= \mathfrak{T}*{X}(\mathcal{T}, \hat{\mathcal{P}}) \quad &\text{(三角测量)}\\
 (\mathcal{P}, X) &= \text{BA}(\mathcal{T}, \hat{\mathcal{P}}, \hat{X}) \quad &\text{(光束平差)}
 \end{aligned}
$$

点追踪器 $\mathbb{T}$：VGGSfM 引入前馈式多帧点追踪网络：输入多帧图像，直接输出跨帧一致的2D轨迹集合 $\mathcal{T}$。Cost-volume金字塔：在多尺度上构建点到特征的相关体，展平后形成令牌序列 $V \in \mathbb{R}^{N_T \times N_I \times C}$。Transformer跟踪：令牌经过多层自注意力Transformer，输出每个点在各帧的2D位置 $y_i^j$与可见性$v_i^j$。置信度预测：通过Aleatoric不确定性建模预测方差 $\sigma_i^j$，损失为高斯负对数似然。粗到细估计：先在全图上粗匹配，再在局部补丁中细化，获得亚像素精度。

相机初始化器 $\mathfrak{T}_{\mathcal{P}}$：该模块利用深度Transformer在全局特征与轨迹特征之间执行交叉注意力，联合估计所有相机位姿。输入包括图像特征$\phi(I_i)$ 与轨迹描述符 $d^{\mathcal{P}}(y_i^j)$；将轨迹对输入8点算法估计初步相机作为几何先验，并嵌入Transformer更新多次以细化预测；使用批量8点算法近似RANSAC过滤噪声匹配，确保稳健性。

三角测量器 $\mathfrak{T}_{X}$：给定初始相机 $\hat{\mathcal{P}}$ 与轨迹 $\mathcal{T}$，预测初步点云$\bar{X}$并细化为最终 $\hat{X}$。初步点由闭式DLT三角测量获得；将相机光线与最近点距离、位置嵌入为特征；使用Transformer融合轨迹与几何特征，回归每个3D点的坐标。

光束平差（BA）：采用可微Levenberg–Marquardt优化器，最小化重投影误差，支持对低置信度或几何不一致的点进行过滤（如低可见性、方差过大、Sampson误差超阈值等）。

总体损失由点云误差、相机误差与轨迹似然构成。



### 4D

#### PhysGen3D: Crafting a Miniature Interactive World from a Single Image

该方法将单张图像转化为交互式、以相机为中心的微型世界，通过构建 3D 世界、动力学模拟和物理渲染生成符合物理规律的逼真视频。借助预训练视觉基础模型，用 GPT-4o 和 Grounded-SAM 分割前景物体，InstantMesh 结合 Zero123++ 生成 3D 网格，迭代修复处理遮挡物体；Dust3r 估计深度生成背景点云与碰撞器，LaMA 修复模型填充背景。多阶段对齐策略确定物体在相机坐标系的姿态和尺度，精对齐阶段通过可微渲染优化损失；Mitsuba3 逆渲染和 DiffusionLight 优化材料与光照参数，GPT-4o 查询物理参数并估计尺度因子。基于 MPM 的 Taichi Elements 模拟器将 3D 资产转换为粒子表示，应用尺度因子调整物理参数，按用户输入设置初始速度和特殊效果。模拟后通过运动插值变形网格，利用优化后的材料在 Mitsuba3 中渲染，构建背景阴影捕捉表面，双通阴影映射提取效果，合成前景、阴影与修复背景生成最终视频。

#### 









