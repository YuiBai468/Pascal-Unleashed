"""Two dense INT8 products and FP32 outlier corrections, without a residue GEMM."""

import research
from research import Spec


research.SPECS = (
    Spec('AR2-16', 16),
    Spec('OP16-2.8-two', 16, clip_a=2.8, clip_b=2.8),
    Spec('OP16-2.95-two', 16, clip_a=2.95, clip_b=2.95),
    Spec('OP16-3.1-two', 16, clip_a=3.1, clip_b=3.1),
    Spec('OP18-2.95-two', 18, clip_a=2.95, clip_b=2.95),
    Spec('OP14-2.95-two', 14, clip_a=2.95, clip_b=2.95),
)


if __name__ == '__main__':
    research.main()
