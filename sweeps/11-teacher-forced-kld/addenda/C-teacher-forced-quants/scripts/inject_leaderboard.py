#!/usr/bin/env python3
"""Inject the unified leaderboard tables into addenda/README.md, idempotently.

On first run, replaces:
  <!-- LEADERBOARD_PLACEHOLDER:multi -->
  <!-- LEADERBOARD_PLACEHOLDER:single -->
with wrapped blocks bounded by:
  <!-- LEADERBOARD_BEGIN:{multi,single} -->
  ... content ...
  <!-- LEADERBOARD_END:{multi,single} -->

On subsequent runs, replaces the content between BEGIN/END markers.
Run AFTER build_unified_leaderboard.py.
"""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
README = ROOT / "addenda" / "README.md"
MD_SRC = ROOT / "addenda" / "C-teacher-forced-quants" / "leaderboard_unified.md"


def extract_section(md_text: str, section_header: str) -> str:
    """Extract everything from a section header until the next `## ` or end-of-file."""
    pattern = rf"^{re.escape(section_header)}\s*$\n(.*?)(?=^## |\Z)"
    m = re.search(pattern, md_text, flags=re.MULTILINE | re.DOTALL)
    if not m:
        return ""
    return m.group(1).strip()


def wrap_block(name: str, body: str) -> str:
    return (
        f"<!-- LEADERBOARD_BEGIN:{name} -->\n"
        f"{body}\n"
        f"<!-- LEADERBOARD_END:{name} -->"
    )


def inject_or_replace(readme: str, name: str, body: str) -> str:
    """Replace placeholder OR existing BEGIN/END block with new wrapped body."""
    wrapped = wrap_block(name, body)
    placeholder = f"<!-- LEADERBOARD_PLACEHOLDER:{name} -->"
    begin = f"<!-- LEADERBOARD_BEGIN:{name} -->"
    end = f"<!-- LEADERBOARD_END:{name} -->"
    if begin in readme and end in readme:
        pattern = re.compile(
            re.escape(begin) + r".*?" + re.escape(end),
            re.DOTALL,
        )
        return pattern.sub(wrapped, readme, count=1)
    if placeholder in readme:
        return readme.replace(placeholder, wrapped, 1)
    print(f"WARNING: neither placeholder nor BEGIN/END block found for '{name}'", file=sys.stderr)
    return readme


def main() -> int:
    if not MD_SRC.exists():
        print(f"ERROR: {MD_SRC} does not exist. Run build_unified_leaderboard.py first.", file=sys.stderr)
        return 1
    if not README.exists():
        print(f"ERROR: {README} does not exist.", file=sys.stderr)
        return 1

    src = MD_SRC.read_text()
    multi_block = extract_section(src, "## Multi-prompt (504 positions = 8 prompts × 63 tokens)")
    single_block = extract_section(src, "## Single-prompt (16 positions, 1 prompt × ~17 tokens)")

    if not multi_block or not single_block:
        print("ERROR: could not extract multi/single sections from leaderboard MD", file=sys.stderr)
        return 2

    multi_section = (
        "**Multi-prompt (504 positions = 8 prompts × 63 tokens, vs `bf16-ref-multi`):**\n\n"
        + multi_block
    )
    single_section = (
        "**Single-prompt (16 positions, 1 prompt × ~17 tokens, vs `bf16-ref-single`):**\n\n"
        + single_block
    )

    readme = README.read_text()
    new_readme = inject_or_replace(readme, "multi", multi_section)
    new_readme = inject_or_replace(new_readme, "single", single_section)

    if new_readme == readme:
        print("WARNING: nothing changed in README.", file=sys.stderr)
        return 3

    README.write_text(new_readme)
    print(f"Injected leaderboard tables into {README}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
