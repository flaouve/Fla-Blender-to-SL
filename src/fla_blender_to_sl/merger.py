"""Merge AI-reconstructed complex mesh blocks into a base schematic JSON."""

from __future__ import annotations

import copy
from typing import Any


def merge_reconstructed_blocks(
    base_schematic: dict[str, Any],
    placeholder_id: int,
    replacement_schematic: dict[str, Any],
) -> dict[str, Any]:
    """Re-index replacement blocks above max existing ObjectId and parent root children under placeholder_id."""
    merged = copy.deepcopy(base_schematic)
    base_root_id = int(merged.get("RootObjectId", 1000))
    base_blocks: list[dict[str, Any]] = merged.get("Blocks", [])

    existing_ids = {int(b["ObjectId"]) for b in base_blocks if "ObjectId" in b}
    if placeholder_id not in existing_ids:
        raise ValueError(f"Placeholder ObjectId {placeholder_id} not found in base schematic.")

    next_id = max([base_root_id, *existing_ids]) + 1

    repl_root_id = int(replacement_schematic.get("RootObjectId", 1000))
    repl_blocks: list[dict[str, Any]] = copy.deepcopy(replacement_schematic.get("Blocks", []))

    id_map: dict[int, int] = {}
    for block in repl_blocks:
        old_id = int(block["ObjectId"])
        id_map[old_id] = next_id
        next_id += 1

    for block in repl_blocks:
        old_id = int(block["ObjectId"])
        old_parent = int(block["ParentId"])
        block["ObjectId"] = id_map[old_id]
        if old_parent == repl_root_id:
            block["ParentId"] = int(placeholder_id)
        elif old_parent in id_map:
            block["ParentId"] = id_map[old_parent]
        else:
            raise ValueError(
                f"Replacement block {block.get('Name', '?')} references unknown ParentId {old_parent}."
            )
        base_blocks.append(block)

    merged["Blocks"] = base_blocks
    return merged
