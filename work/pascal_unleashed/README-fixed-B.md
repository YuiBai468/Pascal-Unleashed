# Pascal Unleashed · 固定权重复用

全程两轮 INT8，仍使用原来的 AR2 重构。右矩阵 B 在循环外按固定的互逆指数编码并上传。每次调用仍从显存里原始 FP16 激活 A 开始，融合求尺度和打包，执行两次 cuBLAS INT8 GEMM，再生成 FP32 输出。

4096 方阵，Tesla P40、cuBLAS 12.6.4，算法号 1。每种合成数据用两份不同 A 重复同一 B，包含正态输入和互逆尺度输入。

| 口径 | 预热复测中位吞吐 |
|---|---:|
| A、B 每次都从显存里的 FP16 编码 | 15.71–15.72 等效 TFLOPS |
| A 每次编码，B 预编码复用 | 17.11–17.26 等效 TFLOPS |
| A、B 都已编码 | 17.77–17.79 等效 TFLOPS |

缓存 B 时，FP64 参考的抽样相对 Frobenius 误差约 0.158%–0.159%，与同批数据上的双动态版同量级。测试覆盖 CUDA 输出与同一编码的 CPU 参考逐项匹配。权重用全长 K 组预编码，一次性编码和上传时间不计入每次调用。没有测真实模型权重。

Windows 上需要 NumPy、CUDA 12 的 NVRTC/cuBLAS DLL 和支持 SM 6.1 的 NVIDIA 驱动。默认从 CUDA 版 PyTorch 的 `torch/lib` 获取 DLL。可用环境变量 `PASCAL_CUDA_DLLS` 覆盖路径。

```powershell
nvidia-smi -L
python cache_weight.py --gpu GPU-UUID --samples 9
```

`cache-weight.json` 保留最初较短预热的数据。`cache-weight-warm.json` 是增加 GPU 预热后复测的数据。
