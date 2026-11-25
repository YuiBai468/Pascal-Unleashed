# 构建及交接

## 环境

Windows x64、Python 3.12、NumPy、CUDA 12.6 NVRTC 及 cuBLAS、Zig 0.15.2、Windows 自带的 .NET Framework 4.x C# 编译器。运行时 C ABI 不依赖 Python、Zig 和编译器。CUDA PTX 在发行前编译，运行时由驱动加载。

从源码目录运行以下命令。

```powershell
.\pascal-unleashed\build.ps1 -Python python -Zig 'C:\Tools\zig\zig.exe' -Cuda12Directory 'C:\CUDA\v12.6\bin'
```

`build_kernels.py` 读取源码树中 `work/pascal_stage6` 和 `work/anima_p40` 的已有 CUDA 源文件与码表，对运行时 CUDA 辅助内核编译 PTX。`build.ps1` 编译核心 DLL、导入库、C 示例、控制面板和安装程序，再生成便携包和 Windows Python wheel。第三方工具和 NVIDIA 库通过构建与用户环境引入。

## 验收步骤

1. 解压便携包，配置 `PASCAL_CUDA_DLLS`，运行 `PascalSdkExample.exe`。
2. 安装 wheel 后执行 `python pascal-unleashed/tools/verify.py --output acceptance.json`，检查大小、形状、特殊值、异常值保护、CUDA 流、回退和显存错误恢复。
3. 启动面板或用 `PascalUnleashedPanel.exe --benchmark output.json 0 2048 2048 2048 2` 执行同一后端的测试。
4. 使用安装器的 `--dir`、`--quiet`、`--no-register` 在隔离目录做安装与卸载检查。
5. 对于已有本地 Anima/ComfyUI FP32 安装，运行 `tools/replay_comfy.py --help`，根据本地模型安装路径回放。模型权重由应用提供。项目原始流程位于同仓库的 `work/anima_p40`。

## 工程目录

- `native` 保存 DLL 实现和新增的 CUDA 转换、布局、保护辅助操作。
- `include` 保存 C ABI，`tools` 保存 SDK 示例和实测脚本。
- `python` 保存无框架 ctypes 客户端、可选 PyTorch 接入与进程启动脚本。
- `panel` 保存控制面板和安装器。
- `work` 保存经过筛查且仍可复现的数学和 CUDA 研究代码。
- `release` 在构建机生成。原始研究快照留在本地，公开版保留经脱敏的数据和报告。
