"""Standalone .obj/.glb/.gltf loader and pure-software multi-angle renderer."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - fallback active inside Blender Python
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]

from fla_blender_to_sl.ir import MaterialInfo, MeshData, NodeKind, SceneGraph, SceneNode

# Right-handed Y-Up (glTF / OBJ) -> Right-handed Z-Up (Blender): (x, y, z) -> (x, -z, y)
_R_Y_TO_Z = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)
_T_Y_TO_Z = np.eye(4, dtype=float)
_T_Y_TO_Z[:3, :3] = _R_Y_TO_Z

_BG_COLOR = (30, 34, 40)
_BORDER_COLOR = (68, 76, 88)
_LABEL_COLOR = (226, 232, 240)
_DEFAULT_LINEAR_RGBA = (0.5, 0.5, 0.5, 1.0)


def _srgb_channel_to_linear(c: float) -> float:
    c = max(0.0, min(1.0, float(c)))
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _linear_channel_to_srgb(c: float) -> float:
    c = max(0.0, min(1.0, float(c)))
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


def _rgba_to_linear(raw: Any, assume_linear_if_float: bool = False) -> tuple[float, float, float, float]:
    arr = np.asarray(raw, dtype=float).ravel()
    if arr.size < 3:
        return _DEFAULT_LINEAR_RGBA
    is_byte_scale = arr.max() > 1.0 or (hasattr(raw, "dtype") and np.issubdtype(raw.dtype, np.integer))
    if is_byte_scale:
        arr = arr / 255.0
    r, g, b = float(arr[0]), float(arr[1]), float(arr[2])
    a = float(arr[3]) if arr.size >= 4 else 1.0
    if not (assume_linear_if_float and not is_byte_scale):
        r = _srgb_channel_to_linear(r)
        g = _srgb_channel_to_linear(g)
        b = _srgb_channel_to_linear(b)
    return (
        max(0.0, min(1.0, r)),
        max(0.0, min(1.0, g)),
        max(0.0, min(1.0, b)),
        max(0.0, min(1.0, a)),
    )


def _mean_image_linear_rgba(img: Image.Image) -> tuple[float, float, float, float]:
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba, dtype=float)
    mean_rgba = arr.reshape(-1, 4).mean(axis=0)
    return _rgba_to_linear(mean_rgba, assume_linear_if_float=False)


def _extract_material(geom: trimesh.Trimesh, fallback_name: str) -> MaterialInfo:
    visual = getattr(geom, "visual", None)
    if visual is None:
        return MaterialInfo(name=fallback_name, base_color_linear=_DEFAULT_LINEAR_RGBA)

    mat = getattr(visual, "material", None)
    mat_name = getattr(mat, "name", None) or fallback_name

    base_color = _DEFAULT_LINEAR_RGBA
    emission_color = (0.0, 0.0, 0.0)
    emission_strength = 0.0

    if mat is not None:
        base_factor = getattr(mat, "baseColorFactor", None)
        base_tex = getattr(mat, "baseColorTexture", None)
        diffuse = getattr(mat, "diffuse", None)
        image = getattr(mat, "image", None)

        if base_factor is not None:
            base_color = _rgba_to_linear(base_factor, assume_linear_if_float=True)
        elif base_tex is not None and isinstance(base_tex, Image.Image):
            base_color = _mean_image_linear_rgba(base_tex)
        elif diffuse is not None:
            base_color = _rgba_to_linear(diffuse, assume_linear_if_float=False)
        elif image is not None and isinstance(image, Image.Image):
            base_color = _mean_image_linear_rgba(image)

        emissive = getattr(mat, "emissiveFactor", None)
        if emissive is not None:
            em_arr = np.asarray(emissive, dtype=float).ravel()
            if em_arr.size >= 3 and float(np.max(em_arr[:3])) > 0.0:
                if em_arr.max() > 1.0:
                    em_arr = em_arr / 255.0
                emission_color = (float(em_arr[0]), float(em_arr[1]), float(em_arr[2]))
                emission_strength = 1.0
    else:
        main_color = getattr(visual, "main_color", None)
        if main_color is not None:
            base_color = _rgba_to_linear(main_color, assume_linear_if_float=False)

    return MaterialInfo(
        name=str(mat_name),
        base_color_linear=base_color,
        emission_color_linear=emission_color,
        emission_strength=emission_strength,
    )


def load_mesh_file(path: Path | str, up_axis: str = "Y") -> SceneGraph:
    """Load an .obj, .glb, or .gltf file into a canonical Blender Z-Up SceneGraph.

    Args:
        path: Path to the input mesh file.
        up_axis: Source coordinate up axis ("Y" for standard glTF/OBJ, or "Z" if already Z-Up).

    Returns:
        SceneGraph containing all extracted mesh nodes in Blender RH Z-Up coordinates.
    """
    import trimesh

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {file_path}")

    axis = up_axis.upper()
    if axis not in ("Y", "Z"):
        raise ValueError(f"Unsupported up_axis={up_axis!r}; expected 'Y' or 'Z'")

    loaded = trimesh.load(file_path, force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene(loaded)
    else:
        scene = loaded

    nodes: list[SceneNode] = []
    convert_y_to_z = axis == "Y"

    geom_nodes = list(scene.graph.nodes_geometry)
    if geom_nodes:
        entries: list[tuple[str, np.ndarray, Any]] = []
        for node_name in geom_nodes:
            transform, geom_name = scene.graph[node_name]
            geom = scene.geometry.get(geom_name)
            entries.append((str(node_name), np.asarray(transform, dtype=float), geom))
    else:
        entries = [
            (str(geom_name), np.eye(4, dtype=float), geom)
            for geom_name, geom in scene.geometry.items()
        ]

    for idx, (node_name, transform, geom) in enumerate(entries):
        if not isinstance(geom, trimesh.Trimesh):
            continue
        if len(geom.vertices) == 0 or len(geom.faces) == 0:
            continue

        verts = np.asarray(geom.vertices, dtype=float)
        faces = np.asarray(geom.faces, dtype=int)
        mat = transform.copy()

        if convert_y_to_z:
            verts = (verts @ _R_Y_TO_Z.T)
            mat = _T_Y_TO_Z @ mat @ _T_Y_TO_Z.T

        material = _extract_material(geom, fallback_name=node_name)
        mesh_data = MeshData(vertices=verts, faces=faces, material=material)
        nodes.append(
            SceneNode(
                id=f"node_{idx}",
                name=node_name,
                kind=NodeKind.MESH,
                parent_id=None,
                local_matrix_blender=mat,
                mesh=mesh_data,
            )
        )

    return SceneGraph(name=file_path.stem, nodes=nodes)


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return np.zeros_like(v, dtype=float)
    return v / norm


def _build_camera_views() -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    iso_view = _normalize_vec(np.array([1.0, -1.0, 0.8], dtype=float))
    iso_right = _normalize_vec(np.cross(np.array([0.0, 0.0, 1.0], dtype=float), iso_view))
    iso_up = _normalize_vec(np.cross(iso_view, iso_right))

    return [
        ("Front", np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])),
        ("Back", np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
        ("Left", np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([-1.0, 0.0, 0.0])),
        ("Right", np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])),
        ("Top", np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        ("Iso", iso_right, iso_up, iso_view),
    ]


def _base_srgb_rgb(mesh: MeshData) -> np.ndarray:
    r, g, b, _ = mesh.material.base_color_linear
    srgb = np.array(
        [
            _linear_channel_to_srgb(r),
            _linear_channel_to_srgb(g),
            _linear_channel_to_srgb(b),
        ],
        dtype=float,
    )
    # Lift near-black materials slightly so silhouette remains legible on dark background
    if float(srgb.max()) < 0.15:
        srgb = np.array([0.55, 0.58, 0.64], dtype=float)
    return srgb * 255.0


def _write_rgb_png(out_path: Path, rgb: np.ndarray) -> None:
    """Write an (H, W, 3) uint8 array to PNG using only Python stdlib zlib/struct."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b"")
    out_path.write_bytes(png)


def _rasterize_tri_numpy(
    buf: np.ndarray,
    x0: float, y0: float,
    x1: float, y1: float,
    x2: float, y2: float,
    rgb: tuple[int, int, int],
    *,
    tri_uv: np.ndarray | None = None,
    texture_srgb: np.ndarray | None = None,
    shade: float = 1.0,
) -> None:
    h, w, _ = buf.shape
    min_x = max(0, int(np.floor(min(x0, x1, x2))))
    max_x = min(w - 1, int(np.ceil(max(x0, x1, x2))))
    min_y = max(0, int(np.floor(min(y0, y1, y2))))
    max_y = min(h - 1, int(np.ceil(max(y0, y1, y2))))
    if max_x < min_x or max_y < min_y:
        return

    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-8:
        return

    xs = np.arange(min_x, max_x + 1, dtype=float) + 0.5
    ys = np.arange(min_y, max_y + 1, dtype=float) + 0.5
    gx, gy = np.meshgrid(xs, ys)
    w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
    w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
    w2 = 1.0 - w0 - w1
    mask = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
    if not np.any(mask):
        return

    if tri_uv is not None and texture_srgb is not None and texture_srgb.ndim == 3:
        th, tw, _ = texture_srgb.shape
        u = w0[mask] * tri_uv[0, 0] + w1[mask] * tri_uv[1, 0] + w2[mask] * tri_uv[2, 0]
        v = w0[mask] * tri_uv[0, 1] + w1[mask] * tri_uv[1, 1] + w2[mask] * tri_uv[2, 1]
        tx = np.clip((np.mod(u, 1.0) * tw).astype(int), 0, tw - 1)
        ty = np.clip(((1.0 - np.mod(v, 1.0)) * th).astype(int), 0, th - 1)
        tex_cols = texture_srgb[ty, tx].astype(float) * shade
        buf[min_y : max_y + 1, min_x : max_x + 1][mask] = np.clip(tex_cols, 0, 255).astype(np.uint8)
    else:
        buf[min_y : max_y + 1, min_x : max_x + 1][mask] = rgb


def _render_multiview_numpy_buf(
    mesh: MeshData,
    tile_size: int = 256,
) -> np.ndarray:
    """Render 3x2 multi-angle sheet into an (H, W, 3) uint8 numpy buffer (supports UV textures)."""
    w, h = tile_size * 3, tile_size * 2
    buf = np.full((h, w, 3), _BG_COLOR, dtype=np.uint8)

    views = _build_camera_views()
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    has_geom = verts.ndim == 2 and verts.shape[0] > 0 and faces.ndim == 2 and faces.shape[0] > 0
    has_tex = (
        mesh.texture_srgb is not None
        and mesh.face_uvs is not None
        and len(mesh.face_uvs) == len(faces)
    )

    if has_geom:
        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)
        center = 0.5 * (bbox_min + bbox_max)
        centered = verts - center
        radius = max(float(np.linalg.norm(centered, axis=1).max()), 1e-9)
        scale = (tile_size * 0.40) / radius

        v0, v1, v2 = centered[faces[:, 0]], centered[faces[:, 1]], centered[faces[:, 2]]
        raw_normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(raw_normals, axis=1, keepdims=True)
        valid_mask = norms[:, 0] > 1e-12
        face_normals = np.zeros_like(raw_normals)
        face_normals[valid_mask] = raw_normals[valid_mask] / norms[valid_mask]
        base_rgb = _base_srgb_rgb(mesh)

        for idx, (_, right, up, view_dir) in enumerate(views):
            ox = (idx % 3) * tile_size
            oy = (idx // 3) * tile_size
            u = centered @ right
            v = centered @ up
            depth = centered @ view_dir
            px = ox + tile_size * 0.5 + u * scale
            py = oy + tile_size * 0.5 - v * scale
            order = np.argsort(depth[faces].mean(axis=1))
            light_dir = _normalize_vec(0.65 * view_dir + 0.55 * up + 0.35 * right)
            shades = 0.28 + 0.72 * np.abs(face_normals @ light_dir)
            for fi in order:
                i0, i1, i2 = faces[fi]
                shade = float(shades[fi])
                rgb = (
                    int(min(255.0, base_rgb[0] * shade)),
                    int(min(255.0, base_rgb[1] * shade)),
                    int(min(255.0, base_rgb[2] * shade)),
                )
                _rasterize_tri_numpy(
                    buf,
                    px[i0], py[i0], px[i1], py[i1], px[i2], py[i2],
                    rgb,
                    tri_uv=mesh.face_uvs[fi] if has_tex else None,
                    texture_srgb=mesh.texture_srgb if has_tex else None,
                    shade=shade,
                )

    for idx in range(len(views)):
        ox = (idx % 3) * tile_size
        oy = (idx // 3) * tile_size
        buf[oy, ox : ox + tile_size] = _BORDER_COLOR
        buf[oy + tile_size - 1, ox : ox + tile_size] = _BORDER_COLOR
        buf[oy : oy + tile_size, ox] = _BORDER_COLOR
        buf[oy : oy + tile_size, ox + tile_size - 1] = _BORDER_COLOR

    return buf


def _render_multiview_numpy(
    mesh: MeshData,
    out_path: Path,
    tile_size: int = 256,
) -> Path:
    """Pure numpy + stdlib PNG fallback when Pillow is unavailable (e.g. inside Blender)."""
    buf = _render_multiview_numpy_buf(mesh, tile_size=tile_size)
    _write_rgb_png(out_path, buf)
    return out_path


def render_multiview_sheet(
    mesh: MeshData,
    output_path: Path | str,
    tile_size: int = 256,
) -> Path:
    """Render a 3x2 labeled multi-angle contact sheet using Pillow or pure-numpy fallback.

    Views: Front, Back, Left, Right, Top, Iso (in Blender Z-Up coordinates).
    Requires no OpenGL context or display server.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if Image is None or ImageDraw is None:
        return _render_multiview_numpy(mesh, out_path, tile_size=tile_size)

    if mesh.texture_srgb is not None and mesh.face_uvs is not None:
        buf = _render_multiview_numpy_buf(mesh, tile_size=tile_size)
        sheet = Image.fromarray(buf, mode="RGB")
        draw = ImageDraw.Draw(sheet)
        for idx, (label, _, _, _) in enumerate(_build_camera_views()):
            ox = (idx % 3) * tile_size
            oy = (idx // 3) * tile_size
            draw.text((ox + 6, oy + 6), label, fill=_LABEL_COLOR)
        sheet.save(out_path, format="PNG")
        return out_path

    sheet = Image.new("RGB", (tile_size * 3, tile_size * 2), color=_BG_COLOR)
    draw = ImageDraw.Draw(sheet)

    views = _build_camera_views()
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    has_geom = verts.ndim == 2 and verts.shape[0] > 0 and faces.ndim == 2 and faces.shape[0] > 0

    if has_geom:
        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)
        center = 0.5 * (bbox_min + bbox_max)
        centered = verts - center
        radius = float(np.linalg.norm(centered, axis=1).max())
        if radius < 1e-9:
            radius = 1.0
        scale = (tile_size * 0.40) / radius

        v0 = centered[faces[:, 0]]
        v1 = centered[faces[:, 1]]
        v2 = centered[faces[:, 2]]
        raw_normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(raw_normals, axis=1, keepdims=True)
        valid_mask = norms[:, 0] > 1e-12
        face_normals = np.zeros_like(raw_normals)
        face_normals[valid_mask] = raw_normals[valid_mask] / norms[valid_mask]

        base_rgb = _base_srgb_rgb(mesh)
    else:
        centered = np.zeros((0, 3), dtype=float)
        scale = 1.0
        face_normals = np.zeros((0, 3), dtype=float)
        base_rgb = np.array([140.0, 150.0, 165.0], dtype=float)

    for idx, (label, right, up, view_dir) in enumerate(views):
        col = idx % 3
        row = idx // 3
        ox = col * tile_size
        oy = row * tile_size

        if has_geom:
            u = centered @ right
            v = centered @ up
            depth = centered @ view_dir

            px = ox + tile_size * 0.5 + u * scale
            py = oy + tile_size * 0.5 - v * scale

            face_depths = depth[faces].mean(axis=1)
            order = np.argsort(face_depths)

            light_dir = _normalize_vec(0.65 * view_dir + 0.55 * up + 0.35 * right)
            shades = 0.28 + 0.72 * np.abs(face_normals @ light_dir)

            for fi in order:
                i0, i1, i2 = faces[fi]
                shade = float(shades[fi])
                rgb = (
                    int(min(255.0, base_rgb[0] * shade)),
                    int(min(255.0, base_rgb[1] * shade)),
                    int(min(255.0, base_rgb[2] * shade)),
                )
                draw.polygon(
                    [
                        (float(px[i0]), float(py[i0])),
                        (float(px[i1]), float(py[i1])),
                        (float(px[i2]), float(py[i2])),
                    ],
                    fill=rgb,
                )

        # Draw tile border and view label
        draw.rectangle(
            [ox, oy, ox + tile_size - 1, oy + tile_size - 1],
            outline=_BORDER_COLOR,
            width=1,
        )
        draw.text((ox + 6, oy + 6), label, fill=_LABEL_COLOR)

    sheet.save(out_path, format="PNG")
    return out_path
