# 安装和卸载

## 安装版

双击 `Pascal-Unleashed-v1.0-win-x64-setup.exe`。默认安装至当前用户的 `%LOCALAPPDATA%\Programs\PascalUnleashed`，创建开始菜单快捷方式与 Windows 应用卸载条目。选择自定义目录时可以从命令行运行 `setup.exe --dir "D:\Apps\PascalUnleashed"`。静默安装加 `--quiet`，隔离安装测试可以再加 `--no-register`。

## 便携版

解压到任意可写目录，打开 `PascalUnleashedPanel.exe`。面板、DLL 和 `pascal_sm61.ptx` 放在一起。面板默认把应用配置放到 `%LOCALAPPDATA%\PascalUnleashed\State`，可在启动前用 `PU_STATE_DIR` 指定独立目录。

## CUDA 依赖

需要工作正常的 NVIDIA 驱动和 CUDA 12 版 cuBLAS。面板支持选择已有的 CUDA `bin` 或 CUDA 12 PyTorch 的 `torch\lib`。命令行也可以在当前进程设置 `PASCAL_CUDA_DLLS`。DLL 会检查调用者传入的目录、进程中已加载的 CUDA 12 cuBLAS、CUDA 环境变量和 Windows 安全库搜索路径。

若系统缺少库，从 [NVIDIA CUDA Toolkit 下载页](https://developer.nvidia.com/cuda-12-6-3-download-archive) 安装兼容的 CUDA 12 组件，或复用已有的 CUDA 12 PyTorch 环境。实际验收使用 CUDA 12.6 的 cuBLAS。安装程序和便携包都未打包 NVIDIA 组件。 [CUDA 许可及可再分发附录](https://docs.nvidia.com/cuda/eula/index.html)

## Python 包

```powershell
python -m pip install .\pascal_unleashed-1.0.0-py3-none-win_amd64.whl
python -c "import pascal_unleashed as p; print(p.devices())"
```

使用 NumPy 接口需安装 NumPy。使用 PyTorch 接口需在目标应用环境已有兼容的 CUDA PyTorch。C SDK 不依赖 Python、NumPy、PyTorch 或 .NET。

## 恢复和卸载

面板内使用 **按原环境启动**，或直接通过原本的入口启动应用。加速环境变量只写入选中的子进程。安装版通过 Windows 设置中的已安装应用卸载，也可以执行安装目录内的 `Uninstall.exe --uninstall --dir "安装目录"`。卸载按随包清单清理安装文件，用户的本地配置保留在 `State`，可单独备份或移除。便携包直接移除解压目录。Python `python -m pip uninstall pascal-unleashed`。

发行文件可用 SHA-256 清单核对。社区构建的安装器目前未做 Authenticode 签名。
