import json
from pathlib import Path
import trimesh
from fla_blender_to_sl.cli import main
from fla_blender_to_sl.merger import merge_reconstructed_blocks
from fla_blender_to_sl.validator import validate_schematic


def test_merge_reindexes_ids_and_parents_under_placeholder():
    base = {
        "RootObjectId": 1000,
        "Blocks": [
            {"Name": "Wall", "ObjectId": 1001, "ParentId": 1000, "Position": {"x": 0, "y": 0, "z": 0}, "Rotation": {"x": 0, "y": 0, "z": 0}, "Scale": {"x": 1, "y": 1, "z": 1}, "BlockType": 1, "Properties": {"PrimitiveType": 3, "Color": "FFFFFFFF", "PrimitiveFlags": 3, "Static": False}},
            {"Name": "LogoPlaceholder", "ObjectId": 1002, "ParentId": 1000, "Position": {"x": 2, "y": 1, "z": 0}, "Rotation": {"x": 0, "y": 0, "z": 0}, "Scale": {"x": 1, "y": 1, "z": 1}, "BlockType": 0, "Properties": {"Static": False}},
        ],
    }
    replacement = {
        "RootObjectId": 1000,
        "Blocks": [
            {"Name": "RingPart1", "ObjectId": 1001, "ParentId": 1000, "Position": {"x": 0, "y": 0.5, "z": 0}, "Rotation": {"x": 0, "y": 0, "z": 0}, "Scale": {"x": 0.5, "y": 0.5, "z": 0.1}, "BlockType": 1, "Properties": {"PrimitiveType": 3, "Color": "343A44FF", "PrimitiveFlags": 2, "Static": False}},
        ],
    }
    merged = merge_reconstructed_blocks(base, placeholder_id=1002, replacement_schematic=replacement)
    report = validate_schematic(merged)
    assert report.valid
    added = next(b for b in merged["Blocks"] if b["Name"] == "RingPart1")
    assert added["ObjectId"] == 1003
    assert added["ParentId"] == 1002


def test_cli_export_and_validate_end_to_end(tmp_path: Path):
    glb = tmp_path / "room.glb"
    trimesh.creation.box(extents=(4.0, 3.0, 0.2)).export(glb)
    out_dir = tmp_path / "out"

    assert main(["export", str(glb), "-o", str(out_dir), "--complex-mode", "decompose"]) == 0
    json_path = out_dir / "room.json"
    assert json_path.exists()
    assert main(["validate", str(json_path)]) == 0
    data = json.loads(json_path.read_text())
    assert data["RootObjectId"] == 1000


def test_cli_ai_handoff_writes_ai_instructions_md(tmp_path: Path):
    glb = tmp_path / "ring.glb"
    trimesh.creation.torus(major_radius=1.0, minor_radius=0.25).export(glb)
    out_dir = tmp_path / "handoff_out"

    assert main(["export", str(glb), "-o", str(out_dir), "--complex-mode", "ai_handoff"]) == 0
    instructions = out_dir / "AI_INSTRUCTIONS.md"
    assert instructions.exists()
    content = instructions.read_text(encoding="utf-8")
    assert "ring.json" in content
    assert "handoff_manifest.json" in content
    assert "fla-blender-to-sl merge" in content
