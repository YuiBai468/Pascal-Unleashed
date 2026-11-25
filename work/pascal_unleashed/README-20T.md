# Pascal Unleashed · 20T 实验

4096×4096×4096，Tesla P40，Windows TCC，cuBLAS 12.6.4。每批三次流水线，预热三次，报告 CUDA Events 计时九批的中位数。等效 TFLOPS 按原矩阵的 `2MNK/t` 计算。测试数字包含 CUDA 显存内 FP16 编码、INT8 GEMM、FP32 重构；预打包路径另行记录。数据上传与 B 的转置布局准备在计时之前完成。

最快的混合路径让前 768 个 K 通道走双轮 AR2，剩下 3328 个 K 通道走单轮 INT8。与全程双轮相比，INT8 乘加工作量降低到 59.375%。实现使用三次 cuBLAS GEMM 和一个统一的重构 kernel。

| 输入 | FP16 驻显存到 FP32 驻显存 | 时间 | 48×48 验证区域相对误差 |
|---|---:|---:|---:|
| 正态随机 FP16 | 22.09–22.83 等效 TFLOPS | 6.02–6.22 ms | 1.070% |
| 单轮段恰好落在 INT8 刻度上 | 22.78–22.81 等效 TFLOPS | 6.03 ms | 0.0418% |
| 单轮段为 INT8 刻度加噪声 | 22.84 等效 TFLOPS | 6.02 ms | 0.0734% |

接近刻度的输入在可精确表示的 INT8 刻度上添加标准差为 `2^-11` 的正态噪声，再转为 FP16。实际低段编码尺度由各行最大值计算。参考输出用 FP64 计算。计时的候选切分为固定的 K 边界。当前没有运行时选择器或真实权重与激活验证。

`try_20t.py` 还试了一种全程双路编码。若以整数坐标 `h,l` 表示 `q=5h+4l`，写 `r=-4h+5l`，则 `q_A q_B + r_A r_B = 41(h_A h_B+l_A l_B)`。选择中心余数并验证 INT8 容量后，可以把两个通道沿 K 拼接成一次 INT8 GEMM。样本误差 0.234%，最好预打包中位数 18.51 等效 TFLOPS。切换 cuBLAS 算法或放大矩阵形状均未在这轮使它突破 20T。

## 复现

需要 Windows NVIDIA 驱动、Python、NumPy 和 CUDA 12 的 NVRTC/cuBLAS DLL。脚本默认从 CUDA 版 PyTorch wheel 的 `torch/lib` 找 DLL。也可用环境变量 `PASCAL_CUDA_DLLS` 指向 DLL 目录。GPU 通过 `nvidia-smi -L` 提供的 UUID 显式选择。

```powershell
python end_to_end.py --gpu GPU-UUID --validate-only
python hybrid_end_to_end.py --gpu GPU-UUID --high-k 768 --samples 9 --near-grid-noise 0.00048828125
python try_20t.py --gpu GPU-UUID --shapes 4096 --samples 7
```

`results` 里的 JSON 包含原始批次计时、验证区域误差，以及有预打包/FP16 动态编码之分的记录。`*.log` 是实测终端输出。已有 AR2/RGS 的 FP16 编码路径在增加输入 stride 后通过 P40 回归校验。
