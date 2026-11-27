"""Exact Fla-SL-CustomObjects / FlaProjectMER JSON schema types and builders."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import Any


def _clean_float(value: float, ndigits: int = 5) -> float:
    """Round float to ndigits and normalize -0.0 to 0.0."""
    rounded = round(float(value), ndigits)
    return 0.0 if rounded == 0.0 else rounded


class BlockType(IntEnum):
    """Supported Fla-SL-CustomObjects BlockType values (BlockType.cs)."""

    EMPTY = 0
    PRIMITIVE = 1
    LIGHT = 2
    TEXT = 8


class PrimitiveType(IntEnum):
    """Unity PrimitiveType enum values."""

    SPHERE = 0
    CAPSULE = 1
    CYLINDER = 2
    CUBE = 3
    PLANE = 4
    QUAD = 5


class PrimitiveFlags(IntFlag):
    """Fla-SL-CustomObjects PrimitiveFlags bitmask (PrimitiveFlags.cs)."""

    NONE = 0
    COLLIDABLE = 1
    VISIBLE = 2


class LightType(IntEnum):
    """Unity LightType values supported by SCP:SL."""

    SPOT = 0
    POINT = 2


class ShadowType(IntEnum):
    """Unity LightShadows values supported by SCP:SL."""

    NONE = 0
    HARD = 1
    SOFT = 2


@dataclass(frozen=True)
class Vec3:
    """3D vector serialized with lowercase x/y/z keys."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "x": _clean_float(self.x),
            "y": _clean_float(self.y),
            "z": _clean_float(self.z),
        }


@dataclass(frozen=True)
class Vec2:
    """2D vector serialized with lowercase x/y keys (no z key, SerializableVector.cs:13)."""

    x: float = 0.0
    y: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "x": _clean_float(self.x),
            "y": _clean_float(self.y),
        }


@dataclass
class SchematicBlockData:
    """Single block entry in a Fla-SL-CustomObjects schematic."""

    name: str
    object_id: int
    parent_id: int
    position: Vec3 = field(default_factory=Vec3)
    rotation: Vec3 = field(default_factory=Vec3)
    scale: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))
    block_type: BlockType = BlockType.EMPTY
    properties: dict[str, Any] | None = None
    animator_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "Name": self.name,
            "ObjectId": int(self.object_id),
            "ParentId": int(self.parent_id),
            "Position": self.position.to_dict(),
            "Rotation": self.rotation.to_dict(),
            "Scale": self.scale.to_dict(),
            "BlockType": int(self.block_type),
        }
        if self.animator_name is not None:
            data["AnimatorName"] = self.animator_name
        if self.properties is not None:
            data["Properties"] = self.properties
        return data


@dataclass
class SchematicObjectDataList:
    """Root container for a Fla-SL-CustomObjects / FlaProjectMER schematic JSON."""

    root_object_id: int = 1000
    blocks: list[SchematicBlockData] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "RootObjectId": int(self.root_object_id),
            "Blocks": [block.to_dict() for block in self.blocks],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def make_empty_block(
    *,
    name: str,
    object_id: int,
    parent_id: int,
    position: Vec3 = Vec3(0.0, 0.0, 0.0),
    rotation: Vec3 = Vec3(0.0, 0.0, 0.0),
    scale: Vec3 = Vec3(1.0, 1.0, 1.0),
    static: bool = False,
) -> SchematicBlockData:
    """Create an Empty grouping/placeholder block (BlockType.EMPTY)."""
    return SchematicBlockData(
        name=name,
        object_id=object_id,
        parent_id=parent_id,
        position=position,
        rotation=rotation,
        scale=scale,
        block_type=BlockType.EMPTY,
        properties={"Static": bool(static)},
    )


def make_primitive_block(
    *,
    name: str,
    object_id: int,
    parent_id: int,
    position: Vec3,
    rotation: Vec3,
    scale: Vec3,
    primitive_type: PrimitiveType,
    color: str = "FFFFFFFF",
    flags: PrimitiveFlags = PrimitiveFlags.COLLIDABLE | PrimitiveFlags.VISIBLE,
    static: bool = False,
) -> SchematicBlockData:
    """Create a Primitive block (BlockType.PRIMITIVE)."""
    return SchematicBlockData(
        name=name,
        object_id=object_id,
        parent_id=parent_id,
        position=position,
        rotation=rotation,
        scale=scale,
        block_type=BlockType.PRIMITIVE,
        properties={
            "PrimitiveType": int(primitive_type),
            "Color": color,
            "PrimitiveFlags": int(flags),
            "Static": bool(static),
        },
    )


def make_light_block(
    *,
    name: str,
    object_id: int,
    parent_id: int,
    position: Vec3,
    rotation: Vec3,
    light_type: LightType = LightType.POINT,
    color: str = "FFFFFFFF",
    unity_intensity: float = 1.0,
    unity_range: float = 10.0,
    shadow_type: ShadowType = ShadowType.NONE,
    spot_angle: float | None = None,
    inner_spot_angle: float | None = None,
    shadow_strength: float = 1.0,
    static: bool = False,
) -> SchematicBlockData:
    """Create a Light block (BlockType.LIGHT), scaling intensity and range by 100.0."""
    props: dict[str, Any] = {
        "LightType": int(light_type),
        "Color": color,
        "Intensity": _clean_float(unity_intensity * 100.0),
        "Range": _clean_float(unity_range * 100.0),
        "ShadowType": int(shadow_type),
        "ShadowStrength": _clean_float(shadow_strength),
        "Static": bool(static),
    }
    if spot_angle is not None:
        props["SpotAngle"] = _clean_float(spot_angle)
    if inner_spot_angle is not None:
        props["InnerSpotAngle"] = _clean_float(inner_spot_angle)

    return SchematicBlockData(
        name=name,
        object_id=object_id,
        parent_id=parent_id,
        position=position,
        rotation=rotation,
        scale=Vec3(1.0, 1.0, 1.0),
        block_type=BlockType.LIGHT,
        properties=props,
    )


def make_text_block(
    *,
    name: str,
    object_id: int,
    parent_id: int,
    position: Vec3,
    rotation: Vec3,
    scale: Vec3,
    text: str,
    color: str = "FFFFFFFF",
    font_size: float = 14.0,
    align: str = "center",
    display_size: Vec2 | None = None,
    static: bool = False,
) -> SchematicBlockData:
    """Create a TextMeshPro Text block (BlockType.TEXT)."""
    clean_color = color.lstrip("#")
    formatted_text = (
        f"<align={align}><color=#{clean_color}>"
        f"<size={_clean_float(font_size)}>{text}</size></color></align>"
    )
    resolved_display_size = display_size if display_size is not None else Vec2(20.0, 5.0)
    return SchematicBlockData(
        name=name,
        object_id=object_id,
        parent_id=parent_id,
        position=position,
        rotation=rotation,
        scale=scale,
        block_type=BlockType.TEXT,
        properties={
            "Text": formatted_text,
            "DisplaySize": resolved_display_size.to_dict(),
            "Static": bool(static),
        },
    )
