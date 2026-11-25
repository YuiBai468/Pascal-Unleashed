AR2 数值审查与 SM 6.1 CUDA 原型

文件
  ar2_audit.py         NumPy 消融实验、抵消反例、舍入区间试验
  ar2_sm61.cu         32×32 输出 tile，K 分块 32，融合两组 DP4A 与 FP32 累加
  ar2_sm61_check.py   Windows 上编译并验证 CUDA 原型

运行
  python ar2_audit.py
  python ar2_sm61_check.py
  nvidia-smi -L
  python ar2_sm61_check.py --gpu GPU-UUID

CPU 脚本只依赖 NumPy。编译与 GPU 验证脚本使用 Windows NVIDIA 驱动和
CUDA 12 版 PyTorch wheel 携带的 NVRTC DLL，需要能编译 compute_61 的工具链。
省略 --gpu 时只编译。指定 --gpu 时只使用对应 UUID，且校验计算能力为 6.1。

实验口径
  CPU 实验采用 NumPy default_rng(42)，每类 8 组，FP16 原始输入，FP64 参考。
  A 每行与 B 每列独立缩放。候选基数由范数误差上界选出，再算两轮主乘积。
  CPU 脚本里的参考乘法、误差恒等式检查和残差试验用于验证方法。
  有限 FP16 输出的舍入区间检查包含一项针对这批合成数据的 FP64 计算余量。

  CUDA 原型接收已经打包的 H 与 H+R、基数和缩放系数，输出 FP32。
  GPU 对照检查使用预定的基数和二进制缩放，验证同一 AR2 算法的输出。
  MX250 已通过 4 组形状，包括 M、N、K 尾块。P40 吞吐待测。
  CPU 的均值、符号和 Hadamard 变体尚未融合进 CUDA 原型。
