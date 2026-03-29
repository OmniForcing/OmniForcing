"""
Schedulers package for LTX-2 DMD distillation.

Note: Custom schedulers have been removed. DMD and ODE training now use:
- Simple linear sigma conversion for fixed denoising step lists
- LTX-2's native scheduler (from ltx-core) for ODE trajectory generation
"""

__all__ = []
