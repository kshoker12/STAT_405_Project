"""
Check whether leaf nodes inside each question's SubjectId set are also
leaf nodes in the full subject metadata forest.

Interpretation used:
- For each question, take its subject IDs as an induced subgraph of the full
  subject hierarchy.
- A question-level leaf is any subject in that set with no children inside
  that same set.
- Every such question-level leaf should also be a global leaf
  (i.e., have no children in subject_metadata.csv).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Load subject metadata and both question metadata files."""
	subject_metadata = pd.read_csv(data_dir / "subject_metadata.csv")
	question_meta_1_2 = pd.read_csv(data_dir / "question_metadata_task_1_2.csv")
	question_meta_3_4 = pd.read_csv(data_dir / "question_metadata_task_3_4.csv")
	return subject_metadata, question_meta_1_2, question_meta_3_4


def build_subject_children_map(subject_metadata: pd.DataFrame) -> Dict[int, Set[int]]:
	"""Build parent -> children mapping from subject metadata."""
	children_map: Dict[int, Set[int]] = defaultdict(set)

	for _, row in subject_metadata.iterrows():
		subject_id = int(row["SubjectId"])
		parent_id = row["ParentId"]

		# Ensure every subject appears as a key, even if it has no children
		_ = children_map[subject_id]

		if pd.notna(parent_id):
			children_map[int(parent_id)].add(subject_id)

	return children_map


def identify_global_leaf_nodes(subject_metadata: pd.DataFrame) -> Set[int]:
	"""Return all subject IDs that have no children globally."""
	children_map = build_subject_children_map(subject_metadata)
	return {subject_id for subject_id, children in children_map.items() if not children}


def parse_subject_ids(subject_ids_str: str) -> List[int]:
	"""Parse JSON list stored in question metadata SubjectId column."""
	parsed = json.loads(subject_ids_str)
	return [int(x) for x in parsed]


def get_question_level_leaf_nodes(
	question_subject_ids: Iterable[int],
	children_map: Dict[int, Set[int]],
) -> Set[int]:
	"""Find leaf nodes within the question-induced subgraph."""
	question_set = set(question_subject_ids)
	question_leaf_nodes: Set[int] = set()

	for subject_id in question_set:
		global_children = children_map.get(subject_id, set())
		children_inside_question = global_children.intersection(question_set)
		if not children_inside_question:
			question_leaf_nodes.add(subject_id)

	return question_leaf_nodes


def validate_question_leaves_against_global_leaves(
	question_metadata: pd.DataFrame,
	children_map: Dict[int, Set[int]],
	global_leaf_nodes: Set[int],
	known_subject_ids: Set[int],
) -> List[Dict]:
	"""
	Return violations where a question-level leaf is not a global leaf.
	"""
	violations: List[Dict] = []

	for _, row in question_metadata.iterrows():
		question_id = int(row["QuestionId"])

		try:
			subject_ids = parse_subject_ids(row["SubjectId"])
		except (json.JSONDecodeError, TypeError, ValueError):
			violations.append(
				{
					"question_id": question_id,
					"reason": "Could not parse SubjectId JSON list",
					"question_leaf_not_global_leaf": [],
					"missing_subject_ids": [],
				}
			)
			continue

		subject_set = set(subject_ids)
		missing_subject_ids = sorted(s for s in subject_set if s not in known_subject_ids)

		question_leaf_nodes = get_question_level_leaf_nodes(subject_set, children_map)
		not_global_leaf = sorted(s for s in question_leaf_nodes if s not in global_leaf_nodes)

		if not_global_leaf or missing_subject_ids:
			violations.append(
				{
					"question_id": question_id,
					"reason": "Question-level leaf nodes are not global leaves"
					if not_global_leaf
					else "Question references unknown subject IDs",
					"question_leaf_not_global_leaf": not_global_leaf,
					"missing_subject_ids": missing_subject_ids,
					"all_question_subject_ids": sorted(subject_set),
				}
			)

	return violations


def print_report(task_name: str, violations: List[Dict], max_rows: int = 20) -> None:
	"""Print concise validation report for one task split."""
	if not violations:
		print(f"✓ {task_name}: no violations")
		return

	print(f"❌ {task_name}: {len(violations)} violation(s)")
	for violation in violations[:max_rows]:
		print(f"\n  Question {violation['question_id']}:")
		print(f"    Reason: {violation['reason']}")
		if violation.get("question_leaf_not_global_leaf"):
			print(
				"    Question-level leaves that are not global leaves: "
				f"{violation['question_leaf_not_global_leaf']}"
			)
		if violation.get("missing_subject_ids"):
			print(f"    Missing subject IDs: {violation['missing_subject_ids']}")

	if len(violations) > max_rows:
		print(f"\n  ... and {len(violations) - max_rows} more")


def main() -> int:
	data_dir = Path(__file__).parent.parent.parent / "data" / "metadata"

	subject_metadata, question_meta_1_2, question_meta_3_4 = load_data(data_dir)

	children_map = build_subject_children_map(subject_metadata)
	global_leaf_nodes = identify_global_leaf_nodes(subject_metadata)
	known_subject_ids = set(int(x) for x in subject_metadata["SubjectId"].values)

	print(f"Total subjects: {len(known_subject_ids)}")
	print(f"Global leaf nodes: {len(global_leaf_nodes)}")

	violations_1_2 = validate_question_leaves_against_global_leaves(
		question_meta_1_2,
		children_map,
		global_leaf_nodes,
		known_subject_ids,
	)
	violations_3_4 = validate_question_leaves_against_global_leaves(
		question_meta_3_4,
		children_map,
		global_leaf_nodes,
		known_subject_ids,
	)

	print()
	print_report("Task 1-2", violations_1_2)
	print_report("Task 3-4", violations_3_4)

	total_violations = len(violations_1_2) + len(violations_3_4)
	print(f"\nTotal violations: {total_violations}")

	return 0 if total_violations == 0 else 1


if __name__ == "__main__":
	sys.exit(main())