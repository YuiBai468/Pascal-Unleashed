# Pascal Unleashed · P40 23.6 T/s

本轮用单张 Tesla P40 实测。基准从驻留显存的两张 FP16 输入开始，计入量化、打包、右矩阵布局转换、INT8 矩阵乘法和完整的 FP32 写回。有效吞吐量按原始形状的 `2*M*N*K` 计算。误差为整张输出对 FP64 参考的相对 Frobenius 误差。

## 8192³ 单次矩阵乘法

| 配置 | 高斯输入端到端 | 整矩阵相对误差 |
| --- | ---: | ---: |
| 两个 FP16 编成三个 INT8 | 24.98–25.24 T/s | 0.4891–0.4894% |
| 法向平移，1024 个位置用更细的标量编码 | 23.71–23.76 T/s | 0.4071–0.4075% |

每档五组独立输入。法向平移将少量超出 INT8 的码沿整数法向移回范围内，相应的平面投影保持精确。两种码共享行尺度并放入一次 INT8 GEMM。第一档结果位于 `results/plane-8192-verified.json`。第二档首组位于 `results/slide-8192-frontier.json`，其余四组位于 `results/slide-8192-dual-verified.json`。

## 异常值检测与 FP32 补偿

GPU 扫描两份原始输入的 K 坐标，通过每侧按坐标峰值的中位指数估算正常范围，找出异常的相邻坐标组。主乘法打包时跳过这些组。FP32 写回时补上它们的原始外积。默认容量是 16 组，每组两个 K 坐标。超过容量时记录 `fallback=true`，全量交给基准编码。

8192³ 上，法向平移加 512 个高精度位置的三组种子实测如下，计时包含检测、收集、量化和补乘。

| FP16 测试分布 | 端到端 | 相对误差 |
| --- | ---: | ---: |
| 正态随机 | 24.02–24.11 T/s | 0.4172–0.4174% |
| 两侧在不同 K 坐标存在 32 倍尖峰 | 23.78–23.98 T/s | 0.3146–0.3158% |
| 两侧在相同 K 坐标存在 32 倍尖峰 | 23.89–23.95 T/s | 0.0211–0.0215% |
| 随机 K 坐标和随机行上的 8 个孤立极值 | 23.72–23.74 T/s | 0.4151–0.4155% |

明细见 `results/robust-slide-512-first.json` 和 `results/robust-slide-512-holdout.json`。代码会核对 GPU 与 CPU 检测到的全部坐标、核对整数点积和补乘后的 32×32 子矩阵，以及核对重复执行的整张结果。

从解压目录用新种子 20261126 复跑 32 倍错位尖峰。原始平面编码为 25.23 T/s、16.4694% 误差。打开检测、法向平移和补偿后为 23.99 T/s、0.3149%。两次使用相同随机输入，记录在 `results/extracted-unmasked-check.json` 和 `results/extracted-robust-check.json`。

## 4096³ 原基准

两个或三个 FP16 共用一个额外 INT8 码的混合配置，实际 K=5760。在单次基准上实测 22.86 T/s、0.4981%。16 组独立输入一起计时的流水吞吐是 23.44 T/s，16 组完整矩阵误差为 0.4979–0.4995%。流水结果在 `results/batch-slide-768-16.json`，单次结果在 `results/vector-512.json`。

## 复跑

实验用 Windows、Tesla P40 SM 6.1、驱动 582.78、CUDA 12.6 cuBLAS 和 Python 3.12。需要 NumPy，CUDA DLL 默认取自本机 PyTorch 安装，也可用 `PASCAL_CUDA_DLLS` 指定。通过 `nvidia-smi -L` 查 P40 UUID。

```powershell
$p40Id = 'GPU-<P40的UUID>'
python -u plane_gpu.py --gpu $p40Id --shape 8192 8192 8192 --encoders plane_bytes_8192 --seeds 2109 43 101 20261125 17 --warmup 26 --samples 13
python -u slide_gpu.py --gpu $p40Id --shape 8192 8192 8192 --triples 0 --dual 1024 --seeds 2109 43 101 20261125 17 --warmup 27 --samples 13
python -u robust_gpu.py --gpu $p40Id --shape 8192 8192 8192 --slide --dual 512 --datasets normal outliers co_spike isolated --seeds 2109 43 101 --warmup 25 --samples 11
```

`work/pascal_stage5` 是本轮工作目录。JSON 保存每组的 CUDA event 样本、形状、实际 K、误差及可用的分阶段计时。
