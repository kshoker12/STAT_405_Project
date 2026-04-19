#!/usr/bin/env python3
"""Analyze how many nodes in each question are adjacent to leaf nodes."""
from __future__ import annotations

import ast
import csv
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SUBJECT_CSV = PROJECT_ROOT / "data" / "metadata" / "subject_metadata.csv"
QUESTION_CSV = PROJECT_ROOT / "data" / "metadata" / "question_metadata_task_1_2.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "question_leaf_adjacency_counts.csv"


def parse_subject_chain(raw: str) -> list[int]:
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    if isinstance(value, list):
        return [int(subject_id) for subject_id in value]
    return []


def print_progress(current: int, total: int, *, prefix: str = "Progress", width: int = 40) -> None:
    if total <= 0:
        return
    fraction = min(max(current / total, 0.0), 1.0)
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{prefix}: [{bar}] {fraction * 100:6.2f}% ({current:,}/{total:,})", end="")
    if current >= total:
        print()


def load_subject_tree(subject_csv: Path) -> tuple[dict[int, str], dict[int, set[int]], set[int]]:
    subject_name_by_id: dict[int, str] = {}
    children_by_parent: dict[int, set[int]] = defaultdict(set)
    all_subject_ids: set[int] = set()

    print("Loading subject metadata...")
    with subject_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = int(row["SubjectId"])
            subject_name_by_id[subject_id] = row["Name"].strip()
            all_subject_ids.add(subject_id)

            parent_id = row.get("ParentId", "")
            if parent_id and parent_id != "NULL":
                children_by_parent[int(parent_id)].add(subject_id)

    leaf_subject_ids = {subject_id for subject_id in all_subject_ids if subject_id not in children_by_parent}
    print(f"Found {len(all_subject_ids)} total subjects")
    print(f"Found {len(leaf_subject_ids)} leaf nodes")
    return subject_name_by_id, children_by_parent, leaf_subject_ids


def count_leaf_adjacent_nodes(
    subject_chain: list[int],
    children_by_parent: dict[int, set[int]],
    leaf_subject_ids: set[int],
) -> int:
    chain_set = set(subject_chain)
    adjacent_nodes: set[int] = set()

    for subject_id in chain_set:
        if subject_id in leaf_subject_ids:
            continue
        if any(child_id in chain_set and child_id in leaf_subject_ids for child_id in children_by_parent.get(subject_id, set())):
            adjacent_nodes.add(subject_id)

    return len(adjacent_nodes)


def main() -> None:
    subject_name_by_id, children_by_parent, leaf_subject_ids = load_subject_tree(SUBJECT_CSV)

    total_questions = sum(1 for _ in QUESTION_CSV.open("r", encoding="utf-8-sig", newline="")) - 1
    if total_questions < 0:
        total_questions = 0

    counts: Counter[int] = Counter()
    max_count = 0
    max_question_ids: list[int] = []

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("\nAnalyzing question metadata...")
    with QUESTION_CSV.open("r", encoding="utf-8-sig", newline="") as input_handle, OUTPUT_CSV.open(
        "w", encoding="utf-8", newline=""
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        writer = csv.DictWriter(output_handle, fieldnames=["QuestionId", "AdjacentLeafNodeCount"])
        writer.writeheader()

        for index, row in enumerate(reader, start=1):
            question_id = int(row["QuestionId"])
            subject_chain = parse_subject_chain(row["SubjectId"])
            adjacent_count = count_leaf_adjacent_nodes(subject_chain, children_by_parent, leaf_subject_ids)

            writer.writerow({"QuestionId": question_id, "AdjacentLeafNodeCount": adjacent_count})
            counts[adjacent_count] += 1

            if adjacent_count > max_count:
                max_count = adjacent_count
                max_question_ids = [question_id]
            elif adjacent_count == max_count:
                max_question_ids.append(question_id)

            if index == 1 or index % 5000 == 0 or index == total_questions:
                print_progress(index, total_questions, prefix="Counting leaf-adjacent nodes")

    print("\n✓ Analysis complete!")
    print(f"Maximum number of adjacent-to-leaf nodes in any question: {max_count}")
    print(f"Questions with that maximum: {len(max_question_ids)}")
    print(f"Output written to: {OUTPUT_CSV}")
    print("\nDistribution (top 20):")
    for count, freq in sorted(counts.items(), reverse=True)[:20]:
        pct = 100 * freq / sum(counts.values()) if counts else 0.0
        print(f"  {count:3d} adjacent nodes: {freq:7,} questions ({pct:6.2f}%)")


if __name__ == "__main__":
    main()
