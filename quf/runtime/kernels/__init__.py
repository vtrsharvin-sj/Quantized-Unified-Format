"""Optional fast kernels. Importing this package never imports Triton."""

from quf.runtime.kernels.triton_kernels import (  # noqa: F401
    Int4Layout,
    dequantize_layout,
    permute_activation,
    repack_int4,
    repack_quantized_weight,
    triton_available,
)
from quf.runtime.kernels.dispatch import (  # noqa: F401
    CachedFp16Weight,
    Int8KernelWeight,
    kernel_linear,
    prepare_for_kernel,
)
from quf.runtime.kernels.fusion import (  # noqa: F401
    FusedKernelWeight,
    fuse_projections,
    fused_linear,
    split_output,
)
