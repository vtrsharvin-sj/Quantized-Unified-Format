"""Optional fast kernels. Importing this package never imports Triton."""

import os

# Triton 3.9 emits a one-time UserWarning when TRITON_DEFAULT_FP_FUSION is unset. "1" is
# Triton's own default (FP fusion on), so the generated code does not change. Every Triton
# import in quf happens in a module of this package, so this runs first. A value the user
# set is kept. Kernels that need exact fp16 rounding still pass enable_fp_fusion=False.
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

from quf.runtime.kernels.triton_kernels import (  # noqa: E402, F401
    Int4Layout,
    dequantize_layout,
    permute_activation,
    repack_int4,
    repack_quantized_weight,
    triton_available,
)
from quf.runtime.kernels.dispatch import (  # noqa: E402, F401
    CachedFp16Weight,
    Int8KernelWeight,
    kernel_linear,
    prepare_for_kernel,
)
from quf.runtime.kernels.fusion import (  # noqa: E402, F401
    FusedKernelWeight,
    fuse_projections,
    fused_linear,
    split_output,
)
