"""Authority checks before opening validation inputs, independent of geometry."""

from pathlib import Path
import stat


def is_linked_path(path: Path) -> bool:
    """lstat-only link/reparse check, including NTFS junctions on Python 3.11.

    Non-Windows stat results have no file attributes; symlink mode is the
    portable fallback. All Windows reparse points are conservatively excluded.
    """
    try:
        metadata = Path(path).lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def exclusion_reason(path: Path) -> str | None:
    """Check lexical evidence *before* resolution, then resolved components.

    Missing ordinary files are eligible pathnames (consumers report absence).
    Candidate components and linked files/directories, including dangling links,
    are never authoritative. This is not a lock against concurrent mutation.
    """
    path = Path(path).absolute()
    if any(part.lower().endswith("_candidate") for part in path.parts):
        return "candidate pathname"
    # Inspect ancestors first so an excluded parent is not traversed.
    for component in (*reversed(path.parents), path):
        if is_linked_path(component):
            return "linked pathname"
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return "unresolvable pathname"
    if any(part.lower().endswith("_candidate") for part in resolved.parts):
        return "resolved candidate pathname"
    for component in (*reversed(resolved.parents), resolved):
        if is_linked_path(component):
            return "resolved linked pathname"
    return None


def require_eligible_input(path: Path) -> None:
    reason = exclusion_reason(path)
    if reason is not None:
        raise ValueError(f"Excluded non-authoritative input ({reason}): {path}")
