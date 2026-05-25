# comfyui-turntable-render
# Headless 3D turntable renderer for ComfyUI.
# Renders a GLB/OBJ at N azimuth angles and outputs a stitched sprite sheet.
#
# Tries EGL first (standard on Linux CUDA servers).
# Falls back to osmesa if EGL is unavailable.

import os

# Must be set before any OpenGL / pyrender import.
# Try EGL; if it later fails at runtime, the node will retry with osmesa.
if not os.environ.get("PYOPENGL_PLATFORM"):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
