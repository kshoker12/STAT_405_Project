from __future__ import annotations

import argparse
import ast
import csv
from collections import defaultdict, deque
from pathlib import Path
import shutil


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SUBJECTS = BASE_DIR / "data" / "metadata" / "subject_metadata.csv"
DEFAULT_QUESTIONS = BASE_DIR / "data" / "metadata" / "question_metadata_task_1_2.csv"
DEFAULT_TRAIN = BASE_DIR / "data" / "train_data" / "train_task_1_2.csv"
DEFAULT_OUTPUT = BASE_DIR / "data" / "algebra_task_1_2.csv"
DEFAULT_IMAGES_DIR = BASE_DIR / "data" / "images"
DEFAULT_ALGEBRA_IMAGES_DIR = BASE_DIR / "data" / "algebra_images_task_1_2"


def load_algebra_subject_ids(subjects_path: Path, algebra_root_id: int = 49) -> set[int]:
	children: dict[int, list[int]] = defaultdict(list)
	with subjects_path.open(newline="", encoding="utf-8-sig") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			parent_id = row["ParentId"]
			if parent_id and parent_id != "NULL":
				children[int(parent_id)].append(int(row["SubjectId"]))

	algebra_subject_ids = {algebra_root_id}
	queue = deque([algebra_root_id])
	while queue:
		subject_id = queue.popleft()
		for child_id in children.get(subject_id, []):
			if child_id not in algebra_subject_ids:
				algebra_subject_ids.add(child_id)
				queue.append(child_id)

	return algebra_subject_ids


def parse_subject_chain(raw_subject_chain: str) -> list[int]:
	value = ast.literal_eval(raw_subject_chain)
	return [int(subject_id) for subject_id in value]


def load_algebra_question_ids(questions_path: Path, algebra_subject_ids: set[int]) -> set[int]:
	algebra_question_ids: set[int] = set()
	with questions_path.open(newline="", encoding="utf-8-sig") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			subject_chain = parse_subject_chain(row["SubjectId"])
			if any(subject_id in algebra_subject_ids for subject_id in subject_chain):
				algebra_question_ids.add(int(row["QuestionId"]))
	return algebra_question_ids


def build_algebra_dataset(
	subjects_path: Path,
	questions_path: Path,
	train_path: Path,
	output_path: Path,
) -> int:
	algebra_subject_ids = load_algebra_subject_ids(subjects_path)
	algebra_question_ids = load_algebra_question_ids(questions_path, algebra_subject_ids)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = ["question_id", "user_id", "is_correct", "correct_answer", "answer", "img_url"]
	row_count = 0

	with train_path.open(newline="", encoding="utf-8-sig") as train_handle, output_path.open(
		"w", newline="", encoding="utf-8"
	) as output_handle:
		reader = csv.DictReader(train_handle)
		writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
		writer.writeheader()

		for row in reader:
			question_id = int(row["QuestionId"])
			if question_id not in algebra_question_ids:
				continue

			writer.writerow(
				{
					"question_id": question_id,
					"user_id": int(row["UserId"]),
					"is_correct": int(row["IsCorrect"]),
					"correct_answer": row["CorrectAnswer"],
					"answer": row["AnswerValue"],
					"img_url": f"data/images/{question_id}.jpg",
				}
			)
			row_count += 1

	return row_count


def copy_algebra_images(algebra_csv_path: Path, source_images_dir: Path, output_images_dir: Path) -> int:
	output_images_dir.mkdir(parents=True, exist_ok=True)
	question_ids: set[int] = set()

	with algebra_csv_path.open(newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			question_ids.add(int(row["question_id"]))

	copied_count = 0
	for question_id in sorted(question_ids):
		source_image = source_images_dir / f"{question_id}.jpg"
		target_image = output_images_dir / source_image.name
		if source_image.exists():
			shutil.copy2(source_image, target_image)
			copied_count += 1

	return copied_count


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Build an algebra-only training CSV.")
	parser.add_argument("--subjects", type=Path, default=DEFAULT_SUBJECTS)
	parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
	parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
	parser.add_argument("--algebra-images-dir", type=Path, default=DEFAULT_ALGEBRA_IMAGES_DIR)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	row_count = build_algebra_dataset(args.subjects, args.questions, args.train, args.output)
	copied_count = copy_algebra_images(args.output, args.images_dir, args.algebra_images_dir)
	print(f"Wrote {row_count} algebra rows to {args.output}")
	print(f"Copied {copied_count} algebra images to {args.algebra_images_dir}")


if __name__ == "__main__":
	main()