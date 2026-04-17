from __future__ import annotations

import argparse
import base64
import csv
import os
from pathlib import Path

import anthropic


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "algebra_images_task_1_2"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "algebra_images_task_1_2_ocr.csv"


def extract_text_from_image(image_path: Path) -> str:
	"""Extract text from an image using Claude Vision API."""
	client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
	
	with open(image_path, "rb") as image_file:
		image_data = base64.standard_b64encode(image_file.read()).decode("utf-8")
	
	message = client.messages.create(
		model="claude-3-5-sonnet-20241022",
		max_tokens=1024,
		messages=[
			{
				"role": "user",
				"content": [
					{
						"type": "image",
						"source": {
							"type": "base64",
							"media_type": "image/jpeg",
							"data": image_data,
						},
					},
					{
						"type": "text",
						"text": "Extract all visible text from this image exactly as it appears. Include mathematical symbols, numbers, and all text content.",
					},
				],
			}
		],
	)
	
	return message.content[0].text


def extract_directory(images_dir: Path, output_path: Path, pattern: str = "*.jpg") -> int:
	image_paths = sorted(images_dir.glob(pattern))
	output_path.parent.mkdir(parents=True, exist_ok=True)

	with output_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=["image_name", "extracted_text"])
		writer.writeheader()

		for i, image_path in enumerate(image_paths, start=1):
			try:
				text = extract_text_from_image(image_path)
				writer.writerow(
					{
						"image_name": image_path.name,
						"extracted_text": text,
					}
				)
				print(f"[{i}/{len(image_paths)}] Processed {image_path.name}")
			except Exception as e:
				print(f"[{i}/{len(image_paths)}] Error processing {image_path.name}: {e}")
				writer.writerow(
					{
						"image_name": image_path.name,
						"extracted_text": f"[ERROR: {str(e)}]",
					}
				)

	return len(image_paths)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Extract text from algebra task images using OCR.")
	parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument("--pattern", default="*.jpg", help="Glob pattern for images inside the folder.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	
	if not os.environ.get("ANTHROPIC_API_KEY"):
		raise SystemExit(
			"ANTHROPIC_API_KEY environment variable is not set. "
			"Please set it to your Claude API key and try again."
		)
	
	count = extract_directory(args.images_dir, args.output, args.pattern)
	print(f"\nExtracted text from {count} images into {args.output}")


if __name__ == "__main__":
	main()
