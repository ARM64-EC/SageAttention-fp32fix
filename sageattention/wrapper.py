import torch
import torch.nn.functional as F

from .core import sageattn


# Keep a handle to the original SDPA implementation so we can delegate to it
# when SageAttention's dtype requirements are not met. Prefer the private
# ``_scaled_dot_product_attention`` if available so we still fall back to the
# real PyTorch implementation even if ``scaled_dot_product_attention`` has
# already been monkey-patched elsewhere.
_torch_sdpa = getattr(F, "_scaled_dot_product_attention", F.scaled_dot_product_attention)


def sageattn_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *args,
    **kwargs,
):
    """Fallback-safe wrapper for :func:`sageattn`.

    SageAttention currently supports only ``torch.float16`` and
    ``torch.bfloat16`` inputs. Some models may invoke
    ``scaled_dot_product_attention`` with ``float32`` tensors, which would
    trigger SageAttention's dtype assertion. This wrapper detects unsupported
    dtypes and delegates to PyTorch's native SDPA implementation instead,
    avoiding crashes while preserving SageAttention for supported inputs.
    """

    allowed_dtypes = (torch.float16, torch.bfloat16)
    if (
        q.dtype not in allowed_dtypes
        or k.dtype not in allowed_dtypes
        or v.dtype not in allowed_dtypes
    ):
        return _torch_sdpa(q, k, v, *args, **kwargs)

    return sageattn(q, k, v, *args, **kwargs)
