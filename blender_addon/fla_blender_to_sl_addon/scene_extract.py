"""Pure-Python Blender object -> SceneGraph IR extraction adapter (no top-level bpy)."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from fla_blender_to_sl.ir import (
    ExportOverride,
    LightData,
    MaterialInfo,
    MeshData,
    NodeKind,
    SceneGraph,
    SceneNode,
    TextData,
)
from fla_blender_to_sl.schema import LightType, ShadowType, Vec2

_DEFAULT_MESH_RGBA = (0.5, 0.5, 0.5, 1.0)
_DEFAULT_TEXT_RGBA = (1.0, 1.0, 1.0, 1.0)

_ALIGN_MAP = {
    "LEFT": "left",
    "CENTER": "center",
    "RIGHT": "right",
    "JUSTIFY": "justified",
    "FLUSH": "flush",
}


def _to_rgba(raw: Any, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if raw is None:
        return default
    vals = [float(x) for x in raw]
    if len(vals) < 3:
        return default
    a = vals[3] if len(vals) >= 4 else 1.0
    return (vals[0], vals[1], vals[2], a)


def _extract_image_srgb(node_tree: Any) -> np.ndarray | None:
    """Extract first valid TEX_IMAGE node's pixels as (H, W, 3) uint8 top-down sRGB."""
    nodes = getattr(node_tree, "nodes", [])
    for node in nodes:
        if getattr(node, "type", None) != "TEX_IMAGE":
            continue
        image = getattr(node, "image", None)
        if image is None:
            continue
        size = getattr(image, "size", None)
        pixels = getattr(image, "pixels", None)
        if not size or len(size) < 2 or pixels is None:
            continue
        w, h = int(size[0]), int(size[1])
        if w <= 0 or h <= 0:
            continue
        arr = np.asarray(pixels, dtype=float)
        if arr.size < w * h * 4:
            continue
        rgba = arr[: w * h * 4].reshape((h, w, 4))
        # Blender Image.pixels are bottom-up; flip vertically for standard top-down coords
        rgb = np.clip(rgba[::-1, :, :3], 0.0, 1.0)
        return (rgb * 255.0).astype(np.uint8)
    return None


def _extract_material(
    materials: Any,
    fallback_name: str,
    default_rgba: tuple[float, float, float, float],
) -> tuple[MaterialInfo, np.ndarray | None]:
    mat = None
    if materials:
        for candidate in materials:
            if candidate is not None:
                mat = candidate
                break

    if mat is None:
        return MaterialInfo(name=fallback_name, base_color_linear=default_rgba), None

    mat_name = str(getattr(mat, "name", None) or fallback_name)
    base_rgba = default_rgba
    emission_rgb = (0.0, 0.0, 0.0)
    emission_strength = 0.0

    use_nodes = bool(getattr(mat, "use_nodes", False))
    node_tree = getattr(mat, "node_tree", None)
    found_bsdf = False

    texture_srgb: np.ndarray | None = None
    if use_nodes and node_tree is not None:
        texture_srgb = _extract_image_srgb(node_tree)
        nodes = getattr(node_tree, "nodes", [])
        for node in nodes:
            if getattr(node, "type", None) == "BSDF_PRINCIPLED":
                found_bsdf = True
                inputs = getattr(node, "inputs", {})
                base_sock = inputs.get("Base Color") if isinstance(inputs, dict) else _get_socket(inputs, "Base Color")
                if base_sock is not None:
                    base_rgba = _to_rgba(getattr(base_sock, "default_value", None), default_rgba)

                em_sock = (
                    inputs.get("Emission Color") or inputs.get("Emission")
                    if isinstance(inputs, dict)
                    else (_get_socket(inputs, "Emission Color") or _get_socket(inputs, "Emission"))
                )
                if em_sock is not None:
                    em_rgba = _to_rgba(getattr(em_sock, "default_value", None), (0.0, 0.0, 0.0, 1.0))
                    emission_rgb = (em_rgba[0], em_rgba[1], em_rgba[2])

                str_sock = (
                    inputs.get("Emission Strength")
                    if isinstance(inputs, dict)
                    else _get_socket(inputs, "Emission Strength")
                )
                if str_sock is not None:
                    emission_strength = float(getattr(str_sock, "default_value", 0.0))
                break

    if not found_bsdf:
        diffuse = getattr(mat, "diffuse_color", None)
        if diffuse is not None:
            base_rgba = _to_rgba(diffuse, default_rgba)

    return (
        MaterialInfo(
            name=mat_name,
            base_color_linear=base_rgba,
            emission_color_linear=emission_rgb,
            emission_strength=emission_strength,
        ),
        texture_srgb,
    )


def _get_socket(inputs: Any, name: str) -> Any:
    try:
        return inputs[name]
    except Exception:
        return None


def _parse_override(raw: Any) -> ExportOverride:
    if isinstance(raw, ExportOverride):
        return raw
    if isinstance(raw, str):
        key = raw.strip().upper()
        for member in ExportOverride:
            if member.value == key or member.name == key:
                return member
    return ExportOverride.AUTO


def _extract_matrix(obj: Any) -> np.ndarray:
    mat = getattr(obj, "matrix_local", None)
    if mat is None:
        return np.eye(4, dtype=float)
    arr = np.asarray(mat, dtype=float)
    if arr.shape != (4, 4):
        arr = arr.reshape((4, 4))
    return arr.copy()


def _extract_mesh_data(obj: Any) -> MeshData | None:
    data = getattr(obj, "data", None)
    if data is None:
        return None

    raw_verts = getattr(data, "vertices", [])
    raw_polys = getattr(data, "polygons", [])

    if raw_verts:
        vertices = np.array([tuple(v.co) for v in raw_verts], dtype=float)
    else:
        vertices = np.zeros((0, 3), dtype=float)

    uv_layers = getattr(data, "uv_layers", None)
    active_uv = getattr(uv_layers, "active", None) if uv_layers is not None else None
    uv_data = getattr(active_uv, "data", None) if active_uv is not None else None

    tri_faces: list[tuple[int, int, int]] = []
    tri_uvs: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []
    for poly in raw_polys:
        pverts = tuple(getattr(poly, "vertices", ()))
        if len(pverts) < 3:
            continue
        ploops = tuple(getattr(poly, "loop_indices", ()))
        has_poly_uv = uv_data is not None and len(ploops) == len(pverts)
        v0 = int(pverts[0])
        for i in range(1, len(pverts) - 1):
            tri_faces.append((v0, int(pverts[i]), int(pverts[i + 1])))
            if has_poly_uv:
                uv0 = uv_data[ploops[0]].uv
                uv1 = uv_data[ploops[i]].uv
                uv2 = uv_data[ploops[i + 1]].uv
                tri_uvs.append(((float(uv0[0]), float(uv0[1])), (float(uv1[0]), float(uv1[1])), (float(uv2[0]), float(uv2[1]))))

    faces = np.array(tri_faces, dtype=int) if tri_faces else np.zeros((0, 3), dtype=int)
    face_uvs = np.array(tri_uvs, dtype=float) if (tri_uvs and len(tri_uvs) == len(tri_faces)) else None
    materials = getattr(data, "materials", [])
    material, texture_srgb = _extract_material(
        materials,
        fallback_name=str(getattr(obj, "name", "Material")),
        default_rgba=_DEFAULT_MESH_RGBA,
    )
    return MeshData(
        vertices=vertices,
        faces=faces,
        material=material,
        face_uvs=face_uvs,
        texture_srgb=texture_srgb,
    )


def _extract_light_data(obj: Any) -> LightData | None:
    data = getattr(obj, "data", None)
    if data is None:
        return None

    b_type = str(getattr(data, "type", "POINT")).upper()
    light_type = LightType.SPOT if b_type == "SPOT" else LightType.POINT

    color = _to_rgba(getattr(data, "color", (1.0, 1.0, 1.0)), (1.0, 1.0, 1.0, 1.0))
    energy = float(getattr(data, "energy", 50.0))
    unity_intensity = max(0.01, energy / 50.0)

    use_custom_dist = bool(getattr(data, "use_custom_distance", False))
    cutoff = float(getattr(data, "cutoff_distance", 0.0))
    if use_custom_dist and cutoff > 0.0:
        range_m = cutoff
    else:
        range_m = max(1.0, math.sqrt(max(energy, 1.0)) * 1.5)

    spot_rad = float(getattr(data, "spot_size", math.radians(30.0)))
    spot_angle = float(math.degrees(spot_rad))

    use_shadow = bool(getattr(data, "use_shadow", True))
    soft_size = float(getattr(data, "shadow_soft_size", 0.0))
    if not use_shadow:
        shadow_type = ShadowType.NONE
    elif soft_size > 0.0:
        shadow_type = ShadowType.SOFT
    else:
        shadow_type = ShadowType.HARD

    return LightData(
        light_type=light_type,
        color_linear=color,
        intensity=unity_intensity,
        range_m=range_m,
        spot_angle=spot_angle,
        shadow_type=shadow_type,
    )


def _extract_text_data(obj: Any) -> TextData | None:
    data = getattr(obj, "data", None)
    if data is None:
        return None

    body = str(getattr(data, "body", ""))
    size = float(getattr(data, "size", 1.0))
    font_size = max(1.0, size * 14.0)
    align_x = str(getattr(data, "align_x", "CENTER")).upper()
    align = _ALIGN_MAP.get(align_x, "center")

    materials = getattr(data, "materials", [])
    material, _ = _extract_material(materials, fallback_name="TextMat", default_rgba=_DEFAULT_TEXT_RGBA)

    return TextData(
        text=body,
        color_linear=material.base_color_linear,
        font_size=font_size,
        align=align,
        display_size=Vec2(20.0, 5.0),
    )


def extract_blender_objects(objects: Iterable[Any], scene_name: str) -> SceneGraph:
    """Convert an iterable of Blender Object instances (or test stubs) into a SceneGraph."""
    obj_list = list(objects)
    supported_types = {"MESH", "LIGHT", "FONT", "EMPTY"}

    filtered = [o for o in obj_list if str(getattr(o, "type", "")).upper() in supported_types]
    id_by_pyid: dict[int, str] = {id(obj): f"obj_{idx}" for idx, obj in enumerate(filtered)}

    nodes: list[SceneNode] = []
    for obj in filtered:
        node_id = id_by_pyid[id(obj)]
        name = str(getattr(obj, "name", node_id))
        obj_type = str(getattr(obj, "type", "EMPTY")).upper()

        parent = getattr(obj, "parent", None)
        parent_id = id_by_pyid.get(id(parent)) if parent is not None else None

        matrix_local = _extract_matrix(obj)
        collidable = bool(getattr(obj, "sl_collidable", True))
        visible = bool(getattr(obj, "sl_visible", True))
        static = bool(getattr(obj, "sl_static", False))
        override = _parse_override(getattr(obj, "sl_export_override", "AUTO"))

        mesh_data: MeshData | None = None
        light_data: LightData | None = None
        text_data: TextData | None = None

        if obj_type == "MESH":
            kind = NodeKind.MESH
            mesh_data = _extract_mesh_data(obj)
        elif obj_type == "LIGHT":
            kind = NodeKind.LIGHT
            light_data = _extract_light_data(obj)
        elif obj_type == "FONT":
            kind = NodeKind.TEXT
            text_data = _extract_text_data(obj)
        else:
            kind = NodeKind.EMPTY

        nodes.append(
            SceneNode(
                id=node_id,
                name=name,
                kind=kind,
                parent_id=parent_id,
                local_matrix_blender=matrix_local,
                mesh=mesh_data,
                light=light_data,
                text=text_data,
                collidable=collidable,
                visible=visible,
                static=static,
                override=override,
            )
        )

    return SceneGraph(name=scene_name, nodes=nodes)
