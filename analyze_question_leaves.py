#!/usr/bin/env python3
"""
Analyze question metadata to find the maximum number of leaf subject nodes per question.
"""
import csv
import ast
from pathlib import Path
from collections import Counter


def main():
    # Load subject metadata to find leaf nodes
    subject_csv = Path("data/metadata/subject_metadata.csv")
    subject_name_by_id = {}
    all_ids = set()
    parent_ids = set()

    print("Loading subject metadata...")
    with subject_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row["SubjectId"])
            all_ids.add(sid)
            subject_name_by_id[sid] = row["Name"].strip()
            parent = row.get("ParentId", "")
            if parent and parent != "NULL":
                parent_ids.add(int(parent))

    leaf_ids = all_ids - parent_ids
    print(f"Found {len(all_ids)} total subjects")
    print(f"Found {len(leaf_ids)} leaf nodes (subjects with no children)")

    # Analyze question metadata
    question_csv = Path("data/metadata/question_metadata_task_1_2.csv")
    max_leaf_count = 0
    leaf_count_dist = Counter()
    total_questions = 0

    print("\nAnalyzing question metadata...")
    with question_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            total_questions = i + 1
            subject_chain = ast.literal_eval(row["SubjectId"])
            leaf_count = sum(1 for sid in subject_chain if sid in leaf_ids)
            
            max_leaf_count = max(max_leaf_count, leaf_count)
            leaf_count_dist[leaf_count] += 1
            
            if total_questions % 5000 == 0:
                pct = total_questions / 27615 * 100
                print(f"  Processed {total_questions:,} questions ({pct:.1f}%)")

    print(f"\n✓ Analysis complete!")
    print(f"\nMaximum number of leaf nodes in any question: {max_leaf_count}")
    print(f"\nLeaf count distribution (top 20):")
    for count, freq in sorted(leaf_count_dist.items(), reverse=True)[:20]:
        pct = 100 * freq / total_questions
        print(f"  {count:3d} leaf nodes: {freq:7,} questions ({pct:6.2f}%)")


if __name__ == "__main__":
    main()
