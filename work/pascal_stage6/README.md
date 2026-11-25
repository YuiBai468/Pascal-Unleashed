# Pascal Unleashed · P40 低误差实验

输入为 P40 显存中的 FP16 矩阵，输出为 FP32。测速包含实际执行的编码、转置、INT8 GEMM、补偿及最终写回。每个整矩阵精度结果验证所有输出元素，参考为对同一 FP16 输入计算的 FP64 GEMM。标为探测的实验用于筛选。

## 三档精度追加复测

以下每档都用种子 2109、43、101 的高斯 FP16 输入，与 FP64 对照检查每组的 268,435,456 个输出。速度为各组多次端到端计时的中位数。

| 档位 | 矩阵尺寸 `M×N×K` | P40 实测速度 | 整矩阵相对 Frobenius 误差 |
| --- | --- | ---: | ---: |
| 高速 | `16384×16384×8192` | 23.13–23.28 T/s | 0.37289–0.37321% |
| 均衡 | `16384×16384×16384` | 19.26–19.36 T/s | 0.12370–0.12384% |
| 高精度 | `16384×16384×16384` | 18.06–18.19 T/s | 0.093471–0.093476% |

高速档用了 2560 个双字节坐标，物理 K 为 13568。同尺寸、同种子的原 2048 坐标配置是 23.76 T/s、0.38344%，新配置是 23.28 T/s、0.37294%。2816 坐标的配置中有一组的原始值为 22.99775 T/s。 [高速档复测](results/slide-stage10-fast-2560-verified.json) · [同尺寸原配置](results/slide-stage10-fast-big-control.json) · [2816 坐标复测](results/slide-stage10-fast-2816-verified.json)

均衡方案仅裁剪并补偿左操作数，右操作数按峰值缩放。两边使用分别拟合的码表。筛选固定矩阵尺寸和输入种子，排除速度和精度同时较差的配置。将速度损失和误差增量分别用候选前沿的范围归一化，再比较两项等权的平方距离。同尺寸、同种子的整矩阵对照中，3.45 配置为 19.36 T/s、0.12370%，3.55 配置为 19.39 T/s、0.12613%，等权评分略优的是 3.45。原双轮 AR2 为 19.80 T/s、0.15201%。 [训练与码表筛选](results/asymmetric-stage10-3p45.json) · [等权比较](results/stage10-balanced-frontier.json) · [中间档原配置](results/dense08-stage10-balanced-control.json) · [3.45 整矩阵复测](results/clipped-stage10-balanced-verified.json) · [3.55 同输入对照](results/clipped-stage10-balanced-3p55-matched.json)

高精度档用归一化 FP16 存储一个方向的尾项补偿中间结果，写回前恢复尺度，最终输出仍为 FP32。在 `16384³` 上，这个专用中间缓冲区占用 0.5 GiB，减少了两步补偿间的读写字节。这使 `3.02×RMS` 在三组复测中守住了 18 T/s。阈值 2.96 和 2.98 的误差更低，稳态复测都有测速低于 18 T/s。 [3.02 整矩阵复测](results/clipped-stage10-precision-3p02-verified.json) · [2.96 复测](results/clipped-stage10-precision-2p96-verified.json) · [2.98 复测](results/clipped-stage10-precision-2p98-verified.json)

```powershell
$p40 = 'GPU-REPLACE-WITH-LOCAL-UUID'
$env:OPENBLAS_NUM_THREADS = '4'
python -u slide_gpu.py --gpu $p40 --shape 16384 16384 8192 --triples 0 --dual 2560 --pair-penalty .0025 --edge-cap 128 --transpose 128-stream --epilogue stream --inplace --gpu-reference --seeds 2109 43 101 --warmup 6 --samples 9 --output results/slide-stage10-fast-repeat.json
python -u clip_spmm_gpu.py --gpu $p40 --shape 16384 16384 16384 --model results/asymmetric-stage10-3p45.npz --clip-override 3.45 --tail-tile tail_directional128_bytes --tail-packet-bits 8 --left-only --packed-tail --fused-right --tile-panel 1 --gpu-reference --seeds 2109 43 101 --warmup 6 --samples 9 --output clipped-stage10-balanced-repeat.json
python -u clip_spmm_gpu.py --gpu $p40 --shape 16384 16384 16384 --model results/folded-lane-0.npz --clip-override 3.02 --tail-tile tail_directional128_bytes --tail-packet-bits 16 --compact-tail --packed-tail --fused-right --tile-panel 1 --gpu-reference --seeds 2109 43 101 --warmup 6 --samples 9 --output clipped-stage10-precision-repeat.json
```

## 当前三档目标与实测

2026-11-25 调整门槛。极速档守住 23 T/s 后继续压低整矩阵误差。中间档在 18–23 T/s 之间按实测选择速度和精度的折中点。高精度档要求同一配置达到至少 18 T/s 且整矩阵相对 Frobenius 误差严格低于 0.1%。此前 20 T/s 同时低于 0.1% 的实验口径及结果保留在下文。

| 档位 | 已验证配置 | 端到端 | 整矩阵相对误差 |
| --- | --- | ---: | ---: |
| 极速档 | `8192×8192×8192`，2048 个 K 坐标双字节，编码尺度上限 128 | 23.12–23.25 T/s | 0.38345–0.38374% |
| 中间档 | `8192×8192×16384`，按行峰值缩放的双轮 AR2 | 19.51–19.55 T/s | 0.15199–0.15242% |
| 高精度档 | `16384×16384×16384`，训练编码表和按方向分开的 16 位尾项补偿 | 18.33–18.48 T/s | 0.0996874–0.0996952% |

三档使用同一块 P40 和种子 2109、43、101 的高斯 FP16 输入。前两档每组核对 67,108,864 个输出，高精度档每组核对 268,435,456 个输出。每个吞吐量为该组多次端到端计时的中位数，误差的参考矩阵取自同一份 FP16 输入。高精度档的 GPU FP64 参考另用 CPU FP64 抽查了坐标和方向。 [极速档复测](results/slide-stage8-23t-verified.json) · [中间档复测](results/dense08-stage7-mid-verified.json) · [高精度档复测](results/clipped-stage8-18t-verified.json)

极速档延续短块打包、计算乘积、再重构的思路。从 1280 增加到 2048 个双字节坐标后，物理 K 为 13312，仍用一次 INT8 GEMM。2304 个双字节坐标在一个种子上落到 22.85 T/s。尺度上限 128 时，极少数无法沿法向放进 INT8 范围的编码点会裁剪。整矩阵误差低于上一版，最大单点绝对误差在三组中为 4.82–5.10。公开论文在精确算术下研究打包和乘法次数，本项目的分块和定点精度由 P40 实测确定。 [OpenAI 矩阵乘法预印本](https://github.com/openai/math/blob/main/preprints/Matrix-Multiplication-Nine-Fourths-October-2-2026/build/paper.tex) · [Rafael Pass 的构造性打包研究](https://arxiv.org/abs/2610.28627) · [上一版极速档](results/slide-stage7-edge-1280-verified.json)

中间档在三组验证中用约 19.5 T/s 把相对误差压到约 0.152%。同一双轮 AR2 编码在 `8192³` 的一次补测里为 18.90 T/s 和 0.14261%，较短的 K 改善了峰值缩放的误差，也提高了其它步骤在总时间里的占比。 [等 K 补测](results/dense08-stage8-mid-matching-k.json)

高精度档将两个稀疏补偿方向分别计算。每个方向连续复用宽度为 128 的稠密字节面板。四个尾项的 K 索引和系数共用一个分包，系数以两个 signed INT8 数位表示，并用 DP4A 计算尾项点积。相同 `16384³` 尺寸的筛选中，FP32 尾项系数的补偿阶段约为 49 ms，分包后约为 36 ms。最终全量复测的单组端到端中位数为 476.1–480.0 ms，现有代码复跑种子 101 得到 18.47 T/s 和同一整矩阵误差。 [方向补偿筛选](results/clipped-stage8-wide-16384-16384.json) · [尾项分包筛选](results/clipped-stage8-packets16-big.json) · [最终代码复测](results/clipped-stage8-final-source-check.json)

```powershell
$p40 = 'GPU-REPLACE-WITH-LOCAL-UUID'
$env:OPENBLAS_NUM_THREADS = '3'
python -u slide_gpu.py --gpu $p40 --shape 8192 8192 8192 --triples 0 --dual 2048 --pair-penalty .0025 --edge-cap 128 --transpose 128-stream --epilogue stream --inplace --seeds 2109 43 101 --warmup 10 --samples 11 --output results/slide-stage8-fast-repeat.json
python -u dense08_gpu.py --gpu $p40 --shape 8192 8192 16384 --seeds 2109 43 101 --warmup 8 --samples 11 --output dense08-stage7-mid-repeat.json
python -u clip_spmm_gpu.py --gpu $p40 --shape 16384 16384 16384 --model results/folded-lane-0.npz --tail-tile tail_directional128_bytes --tail-packet-bits 16 --packed-tail --fused-right --tile-panel 1 --gpu-reference --seeds 2109 43 101 --warmup 7 --samples 9 --output clipped-stage8-18t-repeat.json
```

### 高精度档的精度优先配置

沿用同一编码表和 16 位尾项补偿，把每行裁剪阈值从 `3.20×RMS` 降到 `3.05×RMS`。核心区间缩小，量化更细，尾项补偿的工作量随之增加。`16384³` 的三组高斯 FP16 输入每组对照 FP64 检查了 268,435,456 个输出，端到端为 **18.08–18.17 T/s**，相对误差为 **0.0944108–0.0944173%**。裁剪阈值 `3.00×RMS` 的单种子筛选测到 18.08 T/s 和 0.09287%，精度仅核对了 192 行。 [3.05 整矩阵复测](results/clipped-stage9-cut-3p05-verified.json) · [3.00 筛选](results/clipped-stage9-cut-3p00-probe.json)

在 CPU 上把两个加权编码字节沿有效输入方向和垂直方向分解，可以单独估算后者产生的附加内积。`112×112×16384` 的三个种子使用 `3.20×RMS`，以每行垂直分量的平均绝对值乘各元素符号来估计这个内积，误差为 0.08677–0.08743%。直接计算该内积的误差为 0.07484–0.07531%。这些结果用于衡量误差来自哪里，补偿的 GPU 开销尚未测量。 [CPU 分解实验](results/clipped-stage9-normal-correction-cpu.json)

```powershell
python -u clip_spmm_gpu.py --gpu $p40 --shape 16384 16384 16384 --model results/folded-lane-0.npz --clip-override 3.05 --tail-tile tail_directional128_bytes --tail-packet-bits 16 --packed-tail --fused-right --tile-panel 1 --gpu-reference --seeds 2109 43 101 --warmup 6 --samples 7 --output clipped-stage9-cut-3p05-repeat.json
```

### 集中在少数 K 列的异常值

启用 `--hot-columns` 后，每个输入抽取 64 行估计列能量。在任一输入中超过平均列能量 16 倍的列进入候选集合。候选至多 16 列时，两边都从 INT8 路径剔除这些 K 坐标，并在写回时用原始 FP16 值计算其 FP32 点积补回。候选超过 16 列时走原有编码。这条路径处理集中在少数列的异常值，当前测过同列和错列两组压力输入。

同列压力测试把两边相同的三个 K 列各放大 32 倍。在 `16384³`、种子 2109 的整矩阵复测中，自动识别出三个坐标，跑出 **18.45 T/s 和 0.0071876%**。同尺寸高斯输入打开检测时未命中列，跑出 **18.40 T/s 和 0.0996874%**。两边错开三列的 `8192×8192×16384` 测试识别出六个坐标，得到 **17.53 T/s 和 0.0852158%**。 [同列异常测试](results/clipped-stage8-hot-co_spike-big.json) · [高斯输入检测](results/clipped-stage8-hot-normal-big.json) · [错列异常测试](results/clipped-stage8-hot-outliers.json)

在同一份 `8192×8192×16384` 的同列异常输入上，启用列补偿后整矩阵误差从 **81.32855%** 降至 **0.0072963%**，端到端从 18.04 T/s 变为 17.60 T/s。 [原始路径](results/clipped-stage8-cospike-check.json) · [列补偿复测](results/clipped-stage8-hot-cospike-matched.json)

```powershell
python -u clip_spmm_gpu.py --gpu $p40 --shape 16384 16384 16384 --model results/folded-lane-0.npz --tail-tile tail_directional128_bytes --tail-packet-bits 16 --packed-tail --fused-right --hot-columns --tile-panel 1 --gpu-reference --dataset co_spike --seeds 2109 --warmup 5 --samples 7 --output clipped-stage8-hot-repeat.json
```

## 20 T/s 与 0.3% 误差

在 `M=N=K=8192` 的高斯 FP16 输入上，K 轴前 5888 个坐标各用两个 INT8 字节，余下 2304 个坐标两两编入三个字节。合并后物理 K 为 15232。两段共用行尺度，用一次 INT8 GEMM 产生结果。计时包含编码、转置和 FP32 写回。三个种子实测为 **20.10–20.27 T/s**，相对误差为 **0.29864–0.29884%**。每个种子对照 FP64 验证了 67,108,864 个输出。结果见 [整矩阵复测](results/slide-20t03-verified.json)，其它比例见 [筛选记录](results/slide-20t03-screen.json)。

```powershell
$p40 = 'GPU-REPLACE-WITH-LOCAL-UUID'
$env:OPENBLAS_NUM_THREADS = '3'
python -u slide_gpu.py --gpu $p40 --shape 8192 8192 8192 --triples 0 --dual 5888 --transpose 128-stream --epilogue stream --inplace --seeds 2109 43 101 --warmup 12 --samples 13 --output results/slide-20t03-recheck.json
```

## 0.1% 误差路线

对于每个裁剪后的输入，编码采用 `q = (b-1)H + G`。在原舍入结果越界时，将 `(H,G)` 沿等值线移动 `(±1,∓(b-1))`，使 `q` 不变并保持两位都是合法的 signed INT8。基数 15 的可用范围由 `±1820` 扩大到 `±1905`。区间内的全体整数已在 CPU 上验证重构恒等式。基准从 `3.02σ` 提高到 `3.18σ` 后，三组实测中每个输入需要补偿的元素比例由约 `0.252%` 降至 `0.147%`。

裁剪后的矩阵是 `Ac`、`Bc`，尾项是 `Ra=A−Ac`、`Rb=B−Bc`。修正项为 `Ra·Bᵀ + Ac·Rbᵀ`。训练查表版在裁剪分布上直接最小化乘积误差。自写补偿核以显存里的 FP16 尾项来源为基准，用 FP32 计算尾项值，并可把稠密乘数压到 INT8。单个 CUDA 线程块对输出块加完两项修正后执行一次 FP32 写回。

## 全矩阵结果

下面各项使用 `M=8192, N=8192, K=16384`。每项使用种子 2109、43、101，验证每个种子的 67,108,864 个输出。

| 路线 | 端到端 | 整矩阵相对误差 |
| --- | ---: | ---: |
| 原区间，cuSPARSE 修正 | 15.06–15.09 T/s | 0.09798–0.09799% |
| 扩展编码区间，cuSPARSE 修正 | 15.86–15.87 T/s | 0.09959–0.09961% |
| 训练编码表，cuSPARSE 修正 | 15.85–15.86 T/s | 0.09823–0.09825% |
| 训练编码表，INT8 稠密乘数修正 | 15.96–15.97 T/s | 0.09967–0.09971% |
| 尾项按列块存储，融合右矩阵编码与转置 | 16.30–16.32 T/s | 0.09967–0.09971% |

每个种子的原始样本与误差在 [初始基线](results/clipped-cusparse-overlap-verified.json)、[扩展区间](results/clipped-fold-overlap-verified.json)、[训练编码表](results/clipped-fold-model-verified.json)、[INT8 修正](results/clipped-fold-bytes-verified.json) 和 [融合右矩阵编码](results/clipped-fold-fused-right-verified.json)。最快一组约为 135 ms。按 20 T/s 换算的总预算约为 110 ms，其中两轮 INT8 GEMM 单独测量已经约为 104–105 ms。

## 复测

在本目录运行，GPU 参数使用当前机器上的 P40 UUID。`PASCAL_CUDA_DLLS` 可以指向 CUDA 12 DLL 目录，也可使用本机 PyTorch 附带的 DLL。

```powershell
$p40 = 'GPU-REPLACE-WITH-LOCAL-UUID'
$env:OPENBLAS_NUM_THREADS = '3'
python -u clip_spmm_gpu.py --gpu $p40 --shape 8192 8192 16384 --base 15 --clip 3.18 --fold --overlap --seeds 2109 43 101 --warmup 6 --samples 9 --output clipped-fold-recheck.json
python -u clip_spmm_gpu.py --gpu $p40 --shape 8192 8192 16384 --model results/folded-lane-0.npz --tail-tile tail_tile_bytes --packed-tail --fused-right --tile-panel 1 --seeds 2109 43 101 --warmup 6 --samples 9 --output clipped-fold-fused-right-recheck.json
```

## 其它路线

- [矩形和长 K 的裸 INT8 筛选](results/raw-rectangular-stage6.json)中，单轮约为 41–42.5 TOPS。使用相同编码试跑的矩形端到端约为 14.7 T/s。
- [预留 Strassen 相加范围](results/strassen-digit-scan.json)的 CPU 筛选，第二位限制到 `±63` 时，即使精确恢复裁剪尾项，误差仍约为 0.110–0.112% 或更高。限制到 `±31` 的误差更大。
- [法向误差单独修正](results/normal-correction-scan.json)的 CPU 筛选中，最好的平面编码在精确扣除全部法向量内积后仍约为 0.175%。
- [两个输入共用 K 轴旋转](results/rotation-tail-16384.json)在 CPU 上确实压低尾项。对 256 个坐标做四次稠密旋转，尾项 RMS 由约 0.0144 降至 0.00017–0.00019。这个实验只衡量尾项，GPU 端到端性能和旋转后整矩阵误差仍待验证。
- [K 轴互逆变换](results/dual-precondition-scan.json)用近尾项约束求出 `Q`，分别变换为 `A·Q` 和 `B·Q⁻ᵀ`，在精确算术下保持原乘积。[CPU 全矩阵测试](results/dual-precondition-gemm.json)使用 `2048×2048×4096`，K 轴每 64 列一组。三组输入每边原先各有约 1.13 万个裁剪尾项，变换后计数均为零，且省去稀疏修正后的误差是 0.09952–0.09957%。[P40 变换单项测速](results/dual-transform-gpu-probe.json)在 `8192×8192×16384` 上使用显存中预先准备好的 FP32 数据和宽度 128 的变换矩阵，两侧应用耗时约 9.32 ms。这个时间只覆盖变换矩阵乘法。
