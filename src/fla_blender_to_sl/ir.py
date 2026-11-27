"""Shared pure-Python SceneGraph intermediate representation (IR)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from fla_blender_to_sl.schema import LightType, ShadowType, Vec2


@dataclass
class MaterialInfo:
    """Simplified PBR material attributes in linear color space."""

    name: str
    base_color_linear: tuple[float, float, float, float]
    emission_color_linear: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emission_strength: float = 0.0


@dataclass
class MeshData:
    """Local-space mesh geometry in Blender RH Z-Up coordinates."""

    vertices: np.ndarray
    faces: np.ndarray
    material: MaterialInfo
    face_uvs: np.ndarray | None = None
    texture_srgb: np.ndarray | None = None


@dataclass
class LightData:
    """Light source parameters normalized to Unity-equivalent intensity/range."""

    light_type: LightType
    color_linear: tuple[float, float, float, float]
    intensity: float
    range_m: float
    spot_angle: float = 30.0
    shadow_type: ShadowType = ShadowType.NONE


@dataclass
class TextData:
    """Text object parameters for SCP:SL TextMeshPro blocks."""

    text: str
    color_linear: tuple[float, float, float, float]
    font_size: float = 14.0
    align: str = "center"
    display_size: Vec2 = field(default_factory=lambda: Vec2(20.0, 5.0))


class NodeKind(Enum):
    """Supported IR scene node kinds."""

    EMPTY = "EMPTY"
    MESH = "MESH"
    LIGHT = "LIGHT"
    TEXT = "TEXT"


class ExportOverride(Enum):
    """Per-object export mode override."""

    AUTO = "AUTO"
    FORCE_CUBE = "FORCE_CUBE"
    FORCE_CYLINDER = "FORCE_CYLINDER"
    FORCE_SPHERE = "FORCE_SPHERE"
    FORCE_AI_HANDOFF = "FORCE_AI_HANDOFF"


@dataclass
class SceneNode:
    """Single node in the Blender-space SceneGraph hierarchy."""

    id: str
    name: str
    kind: NodeKind
    parent_id: str | None
    local_matrix_blender: np.ndarray
    mesh: MeshData | None = None
    light: LightData | None = None
    text: TextData | None = None
    collidable: bool = True
    visible: bool = True
    static: bool = False
    override: ExportOverride = ExportOverride.AUTO


@dataclass
class SceneGraph:
    """Root container holding all extracted scene nodes."""

    name: str
    nodes: list[SceneNode] = field(default_factory=list)
