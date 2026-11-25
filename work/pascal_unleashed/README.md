# Pascal Unleashed · Tesla P40 实测

这版保留原始 AR2 CUDA 内核，新增了分组 RGS、双 cuBLAS INT8 路径和 GPU 量化编码。全部计算路径用 FP32 输出，原生 FP16 基线沿用 FP16 输出。

## P40 数据

环境为 Windows、Tesla P40、TCC、驱动 582.78、cuBLAS 12.6.4。4096 方阵使用设备事件计时，预热三次，每批三次 kernel 或计算流水线，报告各批次中位数。计时前数据已驻显存。

| 4096 方阵 | 耗时 | 吞吐 |
|---|---:|---:|
| cuBLAS SGEMM，FP32 驻显存输入 | 13.99–14.76 ms | 9.31–9.83 TFLOPS |
| cuBLAS FP16 输入，FP32 计算与输出 | 16.34 ms | 8.41 TFLOPS |
| 原生 FP16 cuBLAS | 764.46 ms | 0.18 TFLOPS |
| cuBLAS 单轮 INT8 | 3.52 ms | 38.99 INT8 TOPS |
| 原始融合 AR2，K 组 32 | 37.30 ms | 3.68 等效 TFLOPS |
| 自定义融合 RGS，64 个元素一计算 tile，K 组 4096 | 15.74 ms | 8.73 等效 TFLOPS |
| RGS 预打包，K 组 4096 | 7.73 ms | 17.77 等效 TFLOPS |
| RGS FP16 驻显存输入至 FP32 驻显存输出，K 组 4096 | 8.73 ms | 15.74 等效 TFLOPS |

最后一行包括双方的分块最大值、采样中位数、尺度、量化、打包、两次 INT8 GEMM 与重构。B 的连续内存布局按其转置提供。计时从已经在显存里的 FP16 输入开始。需要从主机上传或另外转置 B 的应用应加上相应成本。4096 方阵的 32×32 抽样相对 Frobenius 误差约 0.152%，参考为原始 FP16 输入在 FP64 中相乘。这是合成随机数据的近似计算结果。

8192 方阵的预打包和驻显存端到端分别为 14.67、13.95 等效 TFLOPS。当前最好的已测尺寸为 4096 方阵。INT8 cuBLAS 显式算法 1 与默认算法最佳测量相近。另一条复用 INT32 输出缓冲区的方案需要整数 `beta=15`，本机 cuBLAS 返回参数错误，实测主路径使用两个缓冲区。

## 复现

Python 需要 NumPy。CUDA 运行使用 Windows NVIDIA 驱动以及 CUDA 12 的 NVRTC、cuBLAS DLL。默认找 CUDA 版 PyTorch wheel 的 `torch/lib`。也可以把 `PASCAL_CUDA_DLLS` 设为包含这些 DLL 的目录。

```powershell
nvidia-smi -L
python bench.py --gpu GPU-UUID --shapes 512 1024 2048 4096 256x4096x2048 --samples 7
python pipeline.py --gpu GPU-UUID --shapes 2048 4096 --groups 512 1024 2048 4096 --samples 7
python end_to_end.py --gpu GPU-UUID --shapes 4096 --groups 512 2048 4096 --samples 9
```

在 `bench.py` 或 `end_to_end.py` 加 `--validate-only` 可只做编译与 GPU 正确性测试。命令行需显式选择 GPU。配套 `results` 包含各轮原始采样、CSV、PTX 和 `sm_61` cubin，根目录的 `*.log` 是终端记录。`end-to-end.json` 来自标量打包版本，`end-to-end-aligned.json` 来自当前对齐向量加载版本。

## 实现与约束

- `ar2_sm61.cu` 是原始实现副本。`rgs_sm61.cu` 包含寄存器分块的融合内核和 cuBLAS 重构 kernel。`encode_sm61.cu` 处理动态 FP16 输入。
- CPU 对各行和列做完整中位数统计。GPU 每个尺度块最多均匀取样 256 行或列，实际缩放上界仍按所有元素计算。支持 32 到 8192 的 32 倍数重构组。
- FP16 量化使用基数 16、带符号范围 ±1904、最近偶数舍入。整数部分计算 `15P+Q`，外部用 FP32 乘回 16。尾块以零填充。非有限输入设置检测标志，当前入口拒绝继续返回数值结果。
- GPU 端到端校验包括互逆尺度、非 4 倍数 K、对齐向量读取、零矩阵、FP16 次正规数和极大有限值。基准用已编码数据计算 INT64 参考与带 FMA 的 FP32 重构，另用未量化 FP64 点积报告近似误差。
- 本阶段没有覆盖真实模型权重和激活，也没有测出 20 等效 TFLOPS。偶然的单次峰值不用于报告。
