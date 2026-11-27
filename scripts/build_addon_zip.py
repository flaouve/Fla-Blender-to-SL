"""Build the distributable Blender addon zip archive."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

EXCLUDED_DIRS = {"__pycache__"}
EXCLUDED_SUFFIXES = {".pyc"}


def _iter_package_files(source_dir: Path, archive_prefix: str) -> list[tuple[Path, str]]:
    """Collect (source_path, archive_path) pairs under source_dir, excluding caches."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    entries: list[tuple[Path, str]] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source_dir)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        arcname = f"{archive_prefix}/{rel.as_posix()}"
        entries.append((path, arcname))
    return entries


def build_addon_zip(repo_root: Path, output_path: Path) -> Path:
    """Package the Blender addon and bundled fla_blender_to_sl library into a zip."""
    addon_dir = repo_root / "blender_addon" / "fla_blender_to_sl_addon"
    lib_dir = repo_root / "src" / "fla_blender_to_sl"

    entries = _iter_package_files(addon_dir, "fla_blender_to_sl_addon")
    entries.extend(_iter_package_files(lib_dir, "fla_blender_to_sl_addon/fla_blender_to_sl"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src_path, arcname in entries:
            zf.write(src_path, arcname)

    return output_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for building the Blender addon zip."""
    repo_root = Path(__file__).resolve().parent.parent
    default_output = repo_root / "dist" / "fla_blender_to_sl_addon.zip"

    parser = argparse.ArgumentParser(
        description="Package fla_blender_to_sl_addon and fla_blender_to_sl into a Blender addon zip."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output zip path (default: {default_output})",
    )
    args = parser.parse_args(argv)

    output_path = args.output
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    built_path = build_addon_zip(repo_root, output_path)
    size_bytes = built_path.stat().st_size
    print(f"Built {built_path} ({size_bytes} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
