"""
nodes.py — TurntableRenderNode for ComfyUI

Takes a 3D model file path (GLB or OBJ), renders it at N evenly-spaced
azimuth angles using pyrender headless (EGL), and returns one stitched
horizontal IMAGE tensor — one column per view.

Column order (left → right) for num_views=8:
  0° front | 45° | 90° right | 135° | 180° back | 225° | 270° left | 315°

The 0° front view has the camera on the +Z axis looking toward the origin.
This matches glTF/GLB convention where the front face of the model faces +Z.
"""

import math
import os

import numpy as np
import torch
import trimesh
from PIL import Image as PILImage

# pyrender is intentionally NOT imported here at module level.
# Importing it at ComfyUI startup triggers EGL/CUDA-GL context initialisation
# which pre-allocates GPU VRAM before any workflow node has run, competing with
# the Hunyuan3D geometry and paint models.  Instead, each function that needs
# pyrender imports it lazily so the EGL context is only created when the
# TurntableRenderNode actually executes (last in the graph).


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_scene_and_bounds(model_path: str):
    """
    Load a GLB/OBJ into a pyrender Scene and compute bounding box.

    Returns
    -------
    scene  : pyrender.Scene  (contains mesh nodes only — no camera/lights yet)
    center : np.ndarray [3]  (centroid of bounding box)
    radius : float           (half of the longest bounding-box dimension)
    """
    import pyrender  # lazy: keeps EGL off the GPU until this node runs
    loaded = trimesh.load(model_path)

    if isinstance(loaded, trimesh.Scene):
        bounds = loaded.bounds          # (2, 3) array — [min, max]
        geometries = list(loaded.geometry.values())
    elif isinstance(loaded, trimesh.Trimesh):
        bounds = loaded.bounds
        geometries = [loaded]
    else:
        raise ValueError(f"[turntable] unsupported mesh type: {type(loaded)}")

    if bounds is None or len(geometries) == 0:
        raise ValueError(f"[turntable] no geometry found in {model_path}")

    center  = (bounds[0] + bounds[1]) / 2.0
    extents = bounds[1] - bounds[0]
    radius  = float(max(extents)) / 2.0
    if radius < 1e-6:
        radius = 1.0   # degenerate mesh fallback

    scene = pyrender.Scene(
        ambient_light=[0.5, 0.5, 0.5, 1.0],
        bg_color=[0.0, 0.0, 0.0, 0.0],     # transparent background
    )
    for geom in geometries:
        py_mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
        scene.add(py_mesh)

    return scene, center, radius


def _look_at_pose(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Build a 4×4 camera pose matrix for pyrender (OpenGL convention).

    Camera local axes:
        +X = right
        +Y = up
        -Z = forward (camera looks along -Z)
    """
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        forward = np.array([0.0, 0.0, -1.0])
    else:
        forward /= norm

    world_up = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # Camera pointing nearly straight up/down — pick alternate up
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
    else:
        right /= right_norm

    up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] =  right
    pose[:3, 1] =  up
    pose[:3, 2] = -forward   # camera -Z points toward target
    pose[:3, 3] =  eye
    return pose


def _render_one(scene,
                center: np.ndarray,
                radius: float,
                az_deg: float,
                el_deg: float,
                W: int,
                H: int,
                camera_dist: float = 3.2,
                fov_rad: float = 0.42,
                key_intensity: float = 3.0,
                fill_intensity: float = 1.2) -> np.ndarray:
    """
    Render one frame.  Camera and lights are added then removed so the
    scene object can be reused across frames.

    Azimuth convention (glTF / ComfyUI):
        0°   → camera at (0, 0, +dist)  — front face of model visible
        90°  → camera at (+dist, 0, 0)  — right side
        180° → camera at (0, 0, -dist)  — back
        270° → camera at (-dist, 0, 0)  — left side

    Returns
    -------
    RGBA uint8 array of shape (H, W, 4)
    """
    import pyrender  # lazy: keeps EGL off the GPU until this node runs
    dist = radius * camera_dist
    az   = math.radians(az_deg)
    el   = math.radians(el_deg)

    cx, cy, cz = center
    eye = np.array([
        cx + dist * math.cos(el) * math.sin(az),
        cy + dist * math.sin(el),
        cz + dist * math.cos(el) * math.cos(az),   # az=0 → +Z
    ])

    pose = _look_at_pose(eye, center)

    cam    = pyrender.PerspectiveCamera(yfov=fov_rad, znear=0.01, zfar=1000.0)
    dl_key = pyrender.DirectionalLight(color=np.ones(3), intensity=key_intensity)

    fill_eye  = center + np.array([radius * 2.0, radius * 3.0, radius * 0.5])
    fill_pose = _look_at_pose(fill_eye, center)
    dl_fill   = pyrender.DirectionalLight(color=np.ones(3), intensity=fill_intensity)

    cam_node  = scene.add(cam,    pose=pose)
    key_node  = scene.add(dl_key, pose=pose)
    fill_node = scene.add(dl_fill, pose=fill_pose)

    try:
        renderer = pyrender.OffscreenRenderer(W, H)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()
    except Exception as egl_err:
        import os as _os
        if _os.environ.get("PYOPENGL_PLATFORM") != "osmesa":
            import sys, importlib
            print(f"[turntable] EGL failed ({egl_err}), retrying with osmesa", file=sys.stderr)
            _os.environ["PYOPENGL_PLATFORM"] = "osmesa"
            import pyrender as _pr_reload
            importlib.reload(_pr_reload)
            renderer = _pr_reload.OffscreenRenderer(W, H)
            color, _ = renderer.render(scene, flags=_pr_reload.RenderFlags.RGBA)
            renderer.delete()
        else:
            raise

    scene.remove_node(cam_node)
    scene.remove_node(key_node)
    scene.remove_node(fill_node)

    return color   # (H, W, 4) uint8


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class TurntableRenderNode:
    """
    Renders a 3D model at N azimuth angles and returns a stitched sprite sheet.

    Connect the `model_file` STRING output of TencentImageToModelNode
    (Hunyuan3D) directly to `model_path`.

    The output IMAGE is (1, H, W*num_views, 3) — a single wide texture
    that can be fed straight into SaveImage to produce the cardinal sheet.

    Parameters
    ----------
    model_path    : path to GLB or OBJ file (from TencentImageToModelNode)
    num_views     : number of equally-spaced views  (default 8)
    start_azimuth : angle in degrees for column 0   (default 0)
                    Use 180 if the model's front face points toward -Z.
    elevation_deg : camera height above the horizon (default 20)
                    Higher values look more top-down; 0 = perfectly level.
    camera_dist   : pull-back multiplier × bounding radius (default 3.2)
                    Increase if the model is clipped at the edges.
    fov_deg       : vertical field of view in degrees (default 24 ≈ 85 mm)
                    Lower = more telephoto / less perspective distortion.
    frame_w       : pixel width of each individual frame  (default 512)
    frame_h       : pixel height of each individual frame (default 640)
    key_intensity : brightness of the camera-following key light (default 3.0)
    fill_intensity: brightness of the fixed fill light (default 1.2)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {}),
                "num_views": (
                    "INT",
                    {"default": 8, "min": 1, "max": 16, "step": 1},
                ),
                "start_azimuth": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 359.0, "step": 1.0},
                ),
                "elevation_deg": (
                    "FLOAT",
                    {"default": 20.0, "min": -89.0, "max": 89.0, "step": 1.0},
                ),
                "camera_dist": (
                    "FLOAT",
                    {"default": 3.2, "min": 1.0, "max": 10.0, "step": 0.1},
                ),
                "fov_deg": (
                    "FLOAT",
                    {"default": 24.0, "min": 5.0, "max": 90.0, "step": 1.0},
                ),
                "frame_w": (
                    "INT",
                    {"default": 512, "min": 64, "max": 2048, "step": 64},
                ),
                "frame_h": (
                    "INT",
                    {"default": 640, "min": 64, "max": 2048, "step": 64},
                ),
                "key_intensity": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "fill_intensity": (
                    "FLOAT",
                    {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("frames",)
    FUNCTION      = "render"
    CATEGORY      = "3D/Sprite"
    OUTPUT_NODE   = False

    def render(self, model_path, num_views, start_azimuth, elevation_deg,
               camera_dist, fov_deg, frame_w, frame_h,
               key_intensity, fill_intensity):
        # Hy3DExportMesh outputs a relative path (e.g. "3d/hy3d_mesh_00001_.glb").
        # Resolve it against ComfyUI's output directory if it isn't absolute.
        if not os.path.isabs(model_path):
            try:
                import folder_paths
                model_path = os.path.join(folder_paths.get_output_directory(), model_path)
            except ImportError:
                pass

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"[TurntableRenderNode] model not found: {model_path}"
            )

        scene, center, radius = _load_scene_and_bounds(model_path)

        frames = []
        for i in range(num_views):
            az   = start_azimuth + i * 360.0 / num_views
            rgba = _render_one(scene, center, radius, az, elevation_deg,
                               frame_w, frame_h,
                               camera_dist=camera_dist,
                               fov_rad=math.radians(fov_deg),
                               key_intensity=key_intensity,
                               fill_intensity=fill_intensity)
            frames.append(rgba)

        # Return as a batch of RGBA frames — shape (num_views, H, W, 4).
        # ComfyUI SaveImage saves each frame as a separate file, so Graydient
        # returns up to num_views individual PNGs with proper alpha transparency.
        # forge.py receives these and stitches the cardinal sheet server-side.
        frame_tensors = []
        for f in frames:
            img = PILImage.fromarray(f, "RGBA")
            if img.size != (frame_w, frame_h):
                img = img.resize((frame_w, frame_h), PILImage.LANCZOS)
            rgba = np.array(img).astype(np.float32) / 255.0   # (H, W, 4)
            frame_tensors.append(torch.from_numpy(rgba))

        # Stack into (num_views, H, W, 4)
        batch = torch.stack(frame_tensors, dim=0)
        return (batch,)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "TurntableRenderNode": TurntableRenderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TurntableRenderNode": "Turntable Render (Sprites)",
}
