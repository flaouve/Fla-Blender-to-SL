"""Audit validator for Fla-SL-CustomObjects / FlaProjectMER schematic JSON."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from fla_blender_to_sl.schema import BlockType
from fla_blender_to_sl.transforms import MIN_SCALE

_LDR_HEX_RE = re.compile(r"^[0-9A-F]{8}$")
_HDR_COLON_RE = re.compile(r"^\d+(?:\.\d+)?:\d+(?:\.\d+)?:\d+(?:\.\d+)?:\d+(?:\.\d+)?$")
_VALID_BLOCK_TYPES = {int(bt) for bt in BlockType}


@dataclass
class ValidationReport:
    """Validation audit result for a schematic dictionary."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    block_count: int = 0
    primitive_count: int = 0


def validate_schematic(data: dict[str, Any], max_primitives: int = 2000) -> ValidationReport:
    """Validate schematic JSON structure, IDs, scale floors, colors, and primitive budget."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, dict):
        return ValidationReport(valid=False, errors=["Root schematic payload must be a JSON object."])

    root_id = data.get("RootObjectId")
    if not isinstance(root_id, int) or isinstance(root_id, bool):
        errors.append("Missing or non-integer RootObjectId.")
    elif root_id != 1000:
        warnings.append(f"RootObjectId is {root_id}; expected 1000.")

    blocks = data.get("Blocks")
    if not isinstance(blocks, list):
        errors.append("Missing or non-list Blocks field.")
        return ValidationReport(valid=False, errors=errors, warnings=warnings)

    seen_ids: set[int] = set()
    primitive_count = 0

    for idx, block in enumerate(blocks):
        if not isinstance(block, dict):
            errors.append(f"Block[{idx}] is not an object.")
            continue

        obj_id = block.get("ObjectId")
        if not isinstance(obj_id, int) or isinstance(obj_id, bool):
            errors.append(f"Block[{idx}] has missing or non-integer ObjectId.")
        else:
            if obj_id == root_id:
                errors.append(f"Block[{idx}] ObjectId {obj_id} collides with RootObjectId.")
            if obj_id in seen_ids:
                errors.append(f"Duplicate ObjectId {obj_id} at Block[{idx}].")
            seen_ids.add(obj_id)

        block_type = block.get("BlockType")
        if not isinstance(block_type, int) or block_type not in _VALID_BLOCK_TYPES:
            errors.append(f"Block[{idx}] ({block.get('Name', '?')}) has unsupported BlockType {block_type}.")
        elif block_type == int(BlockType.PRIMITIVE):
            primitive_count += 1

        scale = block.get("Scale")
        if not isinstance(scale, dict):
            errors.append(f"Block[{idx}] ({block.get('Name', '?')}) missing Scale object.")
        else:
            for axis in ("x", "y", "z"):
                val = scale.get(axis)
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    errors.append(f"Block[{idx}] Scale.{axis} is missing or non-numeric.")
                elif float(val) < MIN_SCALE:
                    errors.append(
                        f"Block[{idx}] ({block.get('Name', '?')}) Scale.{axis}={val} is below floor {MIN_SCALE}."
                    )
            if block_type == int(BlockType.EMPTY):
                if any(abs(float(scale.get(a, 0.0)) - 1.0) > 1e-4 for a in ("x", "y", "z")):
                    errors.append(
                        f"Empty Block[{idx}] ({block.get('Name', '?')}) must have Scale (1, 1, 1), got {scale}."
                    )

        props = block.get("Properties")
        if isinstance(props, dict) and "Color" in props:
            color = props["Color"]
            if not isinstance(color, str) or not (_LDR_HEX_RE.match(color) or _HDR_COLON_RE.match(color)):
                errors.append(
                    f"Block[{idx}] ({block.get('Name', '?')}) has invalid Color syntax: {color!r}."
                )

    valid_parents = seen_ids | ({root_id} if isinstance(root_id, int) and not isinstance(root_id, bool) else set())
    for idx, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        parent_id = block.get("ParentId")
        obj_id = block.get("ObjectId")
        if not isinstance(parent_id, int) or isinstance(parent_id, bool):
            errors.append(f"Block[{idx}] has missing or non-integer ParentId.")
        elif parent_id not in valid_parents:
            errors.append(
                f"Block[{idx}] ({block.get('Name', '?')}) ParentId {parent_id} does not reference RootObjectId or any block."
            )
        elif parent_id == obj_id:
            errors.append(f"Block[{idx}] ({block.get('Name', '?')}) cannot parent itself (ObjectId {obj_id}).")

    if primitive_count > max_primitives:
        errors.append(
            f"Primitive count {primitive_count} exceeds max_primitives budget ({max_primitives})."
        )

    return ValidationReport(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        block_count=len(blocks),
        primitive_count=primitive_count,
    )
