# comfyui-turntable-render
# Headless 3D turntable renderer for ComfyUI.
# Renders a GLB/OBJ at N azimuth angles and outputs a stitched sprite sheet.
#
# Requires EGL (standard on Linux CUDA servers).
# If EGL is unavailable, set PYOPENGL_PLATFORM=osmesa in your environment.

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
