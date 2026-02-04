from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Segment:
    start: int  # inclusive
    end: int  # inclusive
    grade: int  # 2=HIGH, 1=MID, 0=LOW

    @property
    def length(self) -> int:
        return int(self.end - self.start + 1)


def grades_to_segments(grades: np.ndarray) -> List[Segment]:
    grades = np.asarray(grades, dtype=int).reshape(-1)
    if grades.size == 0:
        return []

    segments: List[Segment] = []
    start = 0
    current = int(grades[0])
    for i in range(1, grades.size):
        g = int(grades[i])
        if g != current:
            segments.append(Segment(start=start, end=i - 1, grade=current))
            start = i
            current = g
    segments.append(Segment(start=start, end=int(grades.size) - 1, grade=current))
    return segments


def merge_adjacent(segments: List[Segment]) -> List[Segment]:
    if not segments:
        return []
    merged: List[Segment] = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg.grade == last.grade and seg.start == last.end + 1:
            last.end = seg.end
        else:
            merged.append(seg)
    return merged


def segments_to_grades(segments: List[Segment], length: int) -> np.ndarray:
    out = np.zeros(int(length), dtype=int)
    for seg in segments:
        out[int(seg.start) : int(seg.end) + 1] = int(seg.grade)
    return out


def enforce_min_lengths(
    base_grades: np.ndarray,
    min_lengths: Dict[int, int],
    *,
    short_segment_policy: str,
    max_passes: int = 10000,
) -> np.ndarray:
    """Enforce minimum segment lengths by modifying grades (exclusive ownership).

    short_segment_policy:
      - "extend": extend a short segment by taking time from lower-grade neighbors.
      - "discard": do not extend; merge the short segment into a neighbor segment.

    This function never drops timestamps; it only reassigns them to neighbor segments.
    """
    grades = np.asarray(base_grades, dtype=int).copy().reshape(-1)
    if grades.size == 0:
        return grades

    policy = str(short_segment_policy).strip().lower()
    if policy not in {"extend", "discard"}:
        policy = "extend"

    segments = grades_to_segments(grades)
    segments = merge_adjacent(segments)

    def _choose_merge_grade(left: Optional[Segment], right: Optional[Segment]) -> Optional[int]:
        candidates: List[int] = []
        if left is not None:
            candidates.append(int(left.grade))
        if right is not None:
            candidates.append(int(right.grade))
        if not candidates:
            return None
        # Prefer the higher-priority neighbor. If tied, prefer the right neighbor.
        if left is not None and right is not None and left.grade == right.grade:
            return int(left.grade)
        if right is not None and (left is None or right.grade >= left.grade):
            return int(right.grade)
        return int(left.grade) if left is not None else int(right.grade)

    passes = 0
    changed = True
    while changed and passes < max_passes:
        passes += 1
        changed = False
        segments = merge_adjacent(segments)

        for i, seg in enumerate(segments):
            min_len = int(min_lengths.get(int(seg.grade), 1))
            if min_len <= 1 or seg.length >= min_len:
                continue

            left = segments[i - 1] if i - 1 >= 0 else None
            right = segments[i + 1] if i + 1 < len(segments) else None

            if policy == "discard":
                target = _choose_merge_grade(left, right)
                if target is None:
                    continue
                seg.grade = int(target)
                changed = True
                break

            # policy == "extend"
            needed = min_len - seg.length

            left_avail = left.length if left is not None and left.grade < seg.grade else 0
            right_avail = right.length if right is not None and right.grade < seg.grade else 0

            if left_avail + right_avail <= 0:
                # No lower-grade neighbor to borrow from -> merge into a neighbor.
                target = _choose_merge_grade(left, right)
                if target is None:
                    continue
                seg.grade = int(target)
                changed = True
                break

            take_left = min(left_avail, needed // 2)
            take_right = min(right_avail, needed - take_left)

            remaining = needed - take_left - take_right
            if remaining > 0:
                extra_left = min(left_avail - take_left, remaining)
                take_left += extra_left
                remaining -= extra_left
            if remaining > 0:
                extra_right = min(right_avail - take_right, remaining)
                take_right += extra_right
                remaining -= extra_right

            # Apply boundary moves (steal from lower-grade neighbors).
            if take_left > 0 and left is not None:
                seg.start -= int(take_left)
                left.end -= int(take_left)
            if take_right > 0 and right is not None:
                seg.end += int(take_right)
                right.start += int(take_right)

            # Remove empty neighbor segments.
            new_segments: List[Segment] = []
            for s in segments:
                if s.start > s.end:
                    continue
                new_segments.append(s)
            segments = new_segments

            # If still short (not enough to steal), merge into neighbor.
            seg_len = seg.length
            if seg_len < min_len:
                left = segments[i - 1] if i - 1 >= 0 else None
                right = segments[i + 1] if i + 1 < len(segments) else None
                target = _choose_merge_grade(left, right)
                if target is not None:
                    seg.grade = int(target)

            changed = True
            break

    segments = merge_adjacent(segments)
    return segments_to_grades(segments, length=int(grades.size))
