"""Read and normalize Dataturks resume-annotation datasets."""

import json
from bisect import bisect_right
from pathlib import Path


ENTITY_LABELS = (
    "Name",
    "Designation",
    "Skills",
    "Companies worked at",
    "College Name",
    "Degree",
    "Graduation Year",
    "Email Address",
    "Location",
    "Years of Experience",
)


def read_dataturks_records(path):
    """Read a JSON array, one object, JSONL, or concatenated JSON objects."""
    source = Path(path)
    content = source.read_text(encoding="utf-8-sig")
    if not content.strip():
        raise ValueError(f"Dataset is empty: {source}")

    try:
        records = json.loads(content)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        records = []
        offset = 0
        while offset < len(content):
            while offset < len(content) and content[offset].isspace():
                offset += 1
            if offset == len(content):
                break
            try:
                record, offset = decoder.raw_decode(content, offset)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {source} at character {error.pos}: "
                    f"{error.msg}"
                ) from error
            records.append(record)

    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list) or any(
        not isinstance(record, dict) for record in records
    ):
        raise ValueError(f"Expected JSON resume objects in {source}")
    return records


def _choose_non_overlapping(candidates):
    """Select the largest non-overlapping set; prefer tighter spans on ties."""
    priority = {label: index for index, label in enumerate(ENTITY_LABELS)}
    ordered = sorted(
        set(candidates),
        key=lambda span: (span[1], span[0], priority[span[2]]),
    )
    ends = [span[1] for span in ordered]
    best = [()]

    def selection_score(selection):
        return (
            len(selection),
            -sum(end - start for start, end, _ in selection),
            tuple(
                (-start, -end, -priority[label])
                for start, end, label in selection
            ),
        )

    for index, span in enumerate(ordered):
        previous = bisect_right(ends, span[0], 0, index)
        with_span = best[previous] + (span,)
        without_span = best[index]
        best.append(
            max((with_span, without_span), key=selection_score)
        )

    return sorted(
        best[-1], key=lambda span: (span[0], span[1], priority[span[2]])
    )


def clean_annotations(text, annotations):
    """Validate offsets, rebuild span text from resume text, and resolve NER conflicts."""
    stats = {
        "removed_unknown_labels": 0,
        "removed_malformed_spans": 0,
        "removed_bad_offsets": 0,
        "removed_empty_spans": 0,
        "corrected_span_text": 0,
        "trimmed_span_whitespace": 0,
        "removed_duplicate_spans": 0,
        "removed_overlapping_spans": 0,
    }
    if not isinstance(annotations, list):
        return [], stats

    candidates = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            stats["removed_malformed_spans"] += 1
            continue

        raw_labels = annotation.get("label")
        raw_labels = raw_labels if isinstance(raw_labels, list) else [raw_labels]
        labels = []
        for label in raw_labels:
            if label not in ENTITY_LABELS:
                stats["removed_unknown_labels"] += 1
            elif label not in labels:
                labels.append(label)

        points = annotation.get("points")
        if not isinstance(points, list) or not points:
            stats["removed_malformed_spans"] += max(1, len(labels))
            continue

        for point in points:
            if not isinstance(point, dict):
                stats["removed_malformed_spans"] += max(1, len(labels))
                continue
            start = point.get("start")
            inclusive_end = point.get("end")
            if (
                isinstance(start, bool)
                or isinstance(inclusive_end, bool)
                or not isinstance(start, int)
                or not isinstance(inclusive_end, int)
            ):
                stats["removed_malformed_spans"] += max(1, len(labels))
                continue

            end = inclusive_end + 1
            if start < 0 or start >= end or end > len(text):
                stats["removed_bad_offsets"] += max(1, len(labels))
                continue

            if point.get("text") != text[start:end]:
                stats["corrected_span_text"] += 1

            original_start, original_end = start, end
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if (start, end) != (original_start, original_end):
                stats["trimmed_span_whitespace"] += 1
            if start == end:
                stats["removed_empty_spans"] += max(1, len(labels))
                continue
            candidates.extend((start, end, label) for label in labels)

    unique_candidates = set(candidates)
    stats["removed_duplicate_spans"] = len(candidates) - len(unique_candidates)
    selected = _choose_non_overlapping(unique_candidates)
    stats["removed_overlapping_spans"] = len(unique_candidates) - len(selected)
    rebuilt = [
        {
            "label": [label],
            "points": [
                {"start": start, "end": end - 1, "text": text[start:end]}
            ],
        }
        for start, end, label in selected
    ]
    return rebuilt, stats


def clean_dataturks_records(records):
    """Merge identical resume texts and return normalized records and statistics."""
    stats = {
        "input_records": len(records),
        "merged_duplicate_records": 0,
        "removed_records_without_entities": 0,
    }
    grouped = {}
    for record in records:
        text = record.get("content")
        if not isinstance(text, str) or not text:
            stats["removed_records_without_entities"] += 1
            continue
        annotations = record.get("annotation", [])
        if not isinstance(annotations, list):
            annotations = []
        if text not in grouped:
            grouped[text] = dict(record, annotation=list(annotations))
        else:
            grouped[text]["annotation"].extend(annotations)
            stats["merged_duplicate_records"] += 1

    cleaned = []
    for record in grouped.values():
        annotations, annotation_stats = clean_annotations(
            record["content"], record["annotation"]
        )
        for key, value in annotation_stats.items():
            stats[key] = stats.get(key, 0) + value
        if annotations:
            record["annotation"] = annotations
            cleaned.append(record)
        else:
            stats["removed_records_without_entities"] += 1

    stats["output_records"] = len(cleaned)
    return cleaned, stats
