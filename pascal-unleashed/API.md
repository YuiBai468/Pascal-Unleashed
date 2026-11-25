# 开发者接口

## C 和 C++

`include/pascal_unleashed.h` 是发布 ABI。链接 `PascalUnleashed.lib`，部署同目录的 `PascalUnleashed.dll` 与 `pascal_sm61.ptx`。`tools/sdk_example.c` 可独立编译，它通过 CUDA 计算一个已知的 2×2 矩阵结果。

1. 用 `pu_devices` 和 `pu_device` 枚举。`pu_open` 打开选定 CUDA 设备，可传入 cuBLAS 目录。
2. 传现有的 64 位 CUDA 显存地址，或用 `pu_alloc`、`pu_upload` 建立显存参数。
3. `pu_gemm` 计算 `C = α·A·B + β·C`。描述符约定 A 逻辑形状为 M×K，B 描述的是转置视图 N×K，C 为 M×N。行、列、批的步长以元素计。
4. 读 `PuResult` 的实际计算档位、回退原因、CUDA 事件时间及吞吐。读回或继续使用 CUDA 输出，并在结束后关闭上下文。

| 内存布局 | A 的行／列步长 | B 的行／列步长 | C 的行／列步长 |
| --- | --- | --- | --- |
| 常规行主序 A(M,K)、B(K,N)、C(M,N) | K / 1 | 1 / N | N / 1 |
| 存储 B 的转置 Bᵀ(N,K) | K / 1 | K / 1 | N / 1 |
| 列主序 A(M,K)、B(K,N)、C(M,N) | 1 / M | K / 1 | 1 / M |

输入支持 FP16 和 FP32，输出支持 FP16 和 FP32。用 `PU_CAST_FP32` 显式允许将 FP32 输入转换成 FP16 后加速。零批和零 M/N 为无操作，零 K 只执行 β 缩放。输入批步长为 0 可以广播。SDK 输出支持行主序、列主序和带行列间隔的稠密布局。调用者负责保证描述符可访问且输出不与输入重叠。

传 `CUstream` 地址可与现有 CUDA 流互操作。SDK 在所选流完成后返回，同一上下文调用串行执行。`pu_open` 会复用同设备当前 CUDA 上下文，或使用主上下文。已有自定义上下文必须在整个 SDK 上下文生命周期内有效。`PU_CACHE_WEIGHTS` 可复用 B 编码。缓存依赖调用方提交非零 `weight_version`，当 B 内容变更、释放或地址被复用时必须更新版本或清空缓存。

设置 `PU_FALLBACK` 可根据 `PuResult.reason` 看到原生回退。自定义 `tolerance > 0` 选择原生 FP32 累加。当前版本没有对任意输入的严格误差认证。对 cuBLAS 应用可按上表把现有缓冲区和流传入 SDK；原库依赖与链接保持原样。

## Python

```python
import numpy as np
from pascal_unleashed import Context

a = np.ones((64, 128), dtype=np.float16)
b = np.ones((128, 64), dtype=np.float16)
with Context(device=0) as ctx:
    output, run = ctx.numpy_gemm(a, b, mode='fast')
    print(output[0, 0], run.as_dict())
```

`numpy_gemm` 测得的 GPU 时间不含主机传输，结果始终为 FP32。`Context.gemm` 直接传显存参数。`Context.benchmark` 支持指定尺寸、种子、档位和参考比较。

## PyTorch

```python
import torch
from pascal_unleashed.torch import Engine, patch_linear

engine = Engine('balanced', fallback=True, guard=True)
try:
    with torch.inference_mode(), patch_linear(engine):
        layer = torch.nn.Linear(512, 768, device='cuda', dtype=torch.float16)
        result = layer(torch.randn(128, 512, device='cuda', dtype=torch.float16))
    print(engine.metrics)
finally:
    engine.close()
```

也可直接用 `engine.matmul(a, b)`，支持批维广播和非默认 CUDA Stream。梯度跟踪、其他设备和类型使用原调用。`patch_linear` 修改当前进程中的函数入口，适合受控推理范围。工作线程同时切换补丁的用法未认证。加载应用时建议由面板启动独立进程。
