# comfyui-turntable-render
# Headless 3D turntable renderer for ComfyUI.
# Renders a GLB/OBJ at N azimuth angles and outputs a stitched sprite sheet.
#
# Tries EGL first (standard on Linux CUDA servers).
# Falls back to osmesa if EGL is unavailable.

import os
import pathlib
import subprocess
import sys

# Must be set before any OpenGL / pyrender import.
# Try EGL; if it later fails at runtime, the node will retry with osmesa.
if not os.environ.get("PYOPENGL_PLATFORM"):
    os.environ["PYOPENGL_PLATFORM"] = "egl"


def _try_build_custom_rasterizer():
    """
    Opportunistically compile kijai's custom_rasterizer CUDA extension if it
    isn't already importable.  This extension is required by Hy3DRenderMultiView
    but is not compiled by Graydient's standard custom-node install flow.

    Runs silently; any failure is printed but never raises — ComfyUI continues
    without the texture-painting nodes, which simply won't work until this
    succeeds on a subsequent run after the venv is cached.
    """
    try:
        import custom_rasterizer  # noqa — already compiled and on sys.path
        return
    except ImportError:
        pass

    # Locate kijai's differentiable_renderer directory relative to this node
    rast_dir = (
        pathlib.Path(__file__).parent.parent
        / "ComfyUI-Hunyuan3DWrapper"
        / "hy3dgen"
        / "texgen"
        / "differentiable_renderer"
    )
    if not rast_dir.is_dir():
        return  # kijai wrapper not installed alongside us — skip silently

    setup_py = rast_dir / "setup.py"
    if not setup_py.exists():
        print("[turntable-render] custom_rasterizer setup.py not found — texture painting unavailable")
        return

    print("[turntable-render] compiling custom_rasterizer (first-time setup, ~60 s)…")
    try:
        result = subprocess.run(
            [sys.executable, str(setup_py), "build_ext", "--inplace"],
            cwd=str(rast_dir),
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode == 0:
            # Make the freshly compiled .so importable
            sys.path.insert(0, str(rast_dir))
            print("[turntable-render] custom_rasterizer compiled OK — texture painting now available")
        else:
            print(
                "[turntable-render] custom_rasterizer build failed "
                f"(exit {result.returncode}):\n{result.stderr[-600:]}"
            )
    except subprocess.TimeoutExpired:
        print("[turntable-render] custom_rasterizer build timed out — texture painting unavailable this run")
    except Exception as exc:
        print(f"[turntable-render] custom_rasterizer build exception: {exc}")


_try_build_custom_rasterizer()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
