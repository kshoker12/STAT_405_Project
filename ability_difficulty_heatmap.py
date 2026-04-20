"""
Large-scale heatmap visualization: student ability vs question difficulty.

This version uses the full training export, which contains more than 10k students
and more than 10k questions, and avoids materializing all interactions in memory.

Axes are empirical IRT-style proxies:
- student ability = smoothed logit of a student's correct rate
- question difficulty = negative smoothed logit of a question's correct rate

The final figure overlays smoothed incorrect density in red and correct density in
green, so the two response classes remain visible at the same time.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


PROJECT_ROOT = Path("/Users/atlasbuchholz/PycharmProjects/UBC/STAT_405_Project")
OUTPUT_DIR = PROJECT_ROOT / "output"
TRAIN_PATH = PROJECT_ROOT / "data" / "train_data" / "train_task_1_2.csv"

CHUNK_SIZE = 1_000_000
N_BINS = 240
SIGMA = 2.6
EPS = 0.5
PLOT_RANGE = (-5.0, 5.0)


def smoothed_logit(correct_count: float, total_count: float) -> float:
    rate = (correct_count + EPS) / (total_count + 2 * EPS)
    rate = np.clip(rate, 1e-6, 1 - 1e-6)
    return float(np.log(rate / (1.0 - rate)))


def accumulate_chunk_totals(path: Path) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    user_correct = pd.Series(dtype="float64")
    user_total = pd.Series(dtype="float64")
    question_correct = pd.Series(dtype="float64")
    question_total = pd.Series(dtype="float64")

    print(f"Reading {path.name} in chunks...")
    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            path,
            usecols=["QuestionId", "UserId", "IsCorrect"],
            dtype={"QuestionId": np.int32, "UserId": np.int32, "IsCorrect": np.int8},
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        user_grouped = chunk.groupby("UserId", sort=False)["IsCorrect"].agg(["sum", "count"])
        question_grouped = chunk.groupby("QuestionId", sort=False)["IsCorrect"].agg(["sum", "count"])

        user_correct = user_correct.add(user_grouped["sum"], fill_value=0)
        user_total = user_total.add(user_grouped["count"], fill_value=0)
        question_correct = question_correct.add(question_grouped["sum"], fill_value=0)
        question_total = question_total.add(question_grouped["count"], fill_value=0)

        if chunk_idx % 5 == 0:
            print(f"  processed {chunk_idx:,} chunks")

    return user_correct, user_total, question_correct, question_total


def build_metrics(
    user_correct: pd.Series,
    user_total: pd.Series,
    question_correct: pd.Series,
    question_total: pd.Series,
):
    user_ability_raw = pd.Series(
        {int(uid): smoothed_logit(user_correct.loc[uid], user_total.loc[uid]) for uid in user_total.index}
    )
    question_difficulty_raw = pd.Series(
        {int(qid): -smoothed_logit(question_correct.loc[qid], question_total.loc[qid]) for qid in question_total.index}
    )

    user_ability = (user_ability_raw - user_ability_raw.mean()) / user_ability_raw.std(ddof=0)
    question_difficulty = (
        question_difficulty_raw - question_difficulty_raw.mean()
    ) / question_difficulty_raw.std(ddof=0)

    return user_ability, question_difficulty


def main() -> None:
    print("Loading large training data and computing empirical ability/difficulty...")
    user_correct, user_total, question_correct, question_total = accumulate_chunk_totals(TRAIN_PATH)

    print(f"  unique students : {len(user_total):,}")
    print(f"  unique questions: {len(question_total):,}")

    user_ability, question_difficulty = build_metrics(
        user_correct, user_total, question_correct, question_total
    )

    print(f"  student ability range (z): [{user_ability.min():.3f}, {user_ability.max():.3f}]")
    print(
        f"  question difficulty range (z): [{question_difficulty.min():.3f}, {question_difficulty.max():.3f}]"
    )

    ability_lo, ability_hi = PLOT_RANGE
    difficulty_lo, difficulty_hi = PLOT_RANGE

    print(f"  plotting window ability   : [{ability_lo:.3f}, {ability_hi:.3f}]")
    print(f"  plotting window difficulty: [{difficulty_lo:.3f}, {difficulty_hi:.3f}]")

    x_edges = np.linspace(difficulty_lo, difficulty_hi, N_BINS + 1)
    y_edges = np.linspace(ability_lo, ability_hi, N_BINS + 1)

    hist_correct = np.zeros((N_BINS, N_BINS), dtype=np.float32)
    hist_incorrect = np.zeros((N_BINS, N_BINS), dtype=np.float32)

    print("Aggregating smoothed densities in a second pass...")
    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            TRAIN_PATH,
            usecols=["QuestionId", "UserId", "IsCorrect"],
            dtype={"QuestionId": np.int32, "UserId": np.int32, "IsCorrect": np.int8},
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        chunk["student_ability"] = chunk["UserId"].map(user_ability)
        chunk["question_difficulty"] = chunk["QuestionId"].map(question_difficulty)

        chunk = chunk.dropna(subset=["student_ability", "question_difficulty"])
        if chunk.empty:
            continue

        x = np.clip(chunk["question_difficulty"].to_numpy(dtype=np.float32), difficulty_lo, difficulty_hi)
        y = np.clip(chunk["student_ability"].to_numpy(dtype=np.float32), ability_lo, ability_hi)
        correct_mask = chunk["IsCorrect"].to_numpy(dtype=np.int8) == 1

        if correct_mask.any():
            hist_correct += np.histogram2d(
                y[correct_mask],
                x[correct_mask],
                bins=[y_edges, x_edges],
            )[0].astype(np.float32)

        if (~correct_mask).any():
            hist_incorrect += np.histogram2d(
                y[~correct_mask],
                x[~correct_mask],
                bins=[y_edges, x_edges],
            )[0].astype(np.float32)

        if chunk_idx % 5 == 0:
            print(f"  aggregated {chunk_idx:,} chunks")

    hist_correct = gaussian_filter(hist_correct, sigma=SIGMA)
    hist_incorrect = gaussian_filter(hist_incorrect, sigma=SIGMA)

    total = hist_correct + hist_incorrect

    def density_to_rgba(density: np.ndarray, base_rgb: tuple[float, float, float]) -> np.ndarray:
        positive = density[density > 0]
        if positive.size == 0:
            norm = np.zeros_like(density, dtype=np.float32)
        else:
            scale = np.percentile(positive, 99.5)
            scale = max(float(scale), 1e-8)
            norm = np.clip(density / scale, 0.0, 1.0).astype(np.float32)

        rgba = np.zeros((density.shape[0], density.shape[1], 4), dtype=np.float32)
        rgba[..., 0] = base_rgb[0] * norm
        rgba[..., 1] = base_rgb[1] * norm
        rgba[..., 2] = base_rgb[2] * norm
        rgba[..., 3] = 0.04 + 0.96 * norm
        return rgba

    rgba_incorrect = density_to_rgba(hist_incorrect, (1.0, 0.0, 0.0))
    rgba_correct = density_to_rgba(hist_correct, (0.0, 0.7, 0.0))

    fig, ax = plt.subplots(figsize=(13, 10), dpi=120)

    ax.imshow(
        rgba_incorrect,
        origin="lower",
        extent=[difficulty_lo, difficulty_hi, ability_lo, ability_hi],
        aspect="auto",
        interpolation="bilinear",
    )
    ax.imshow(
        rgba_correct,
        origin="lower",
        extent=[difficulty_lo, difficulty_hi, ability_lo, ability_hi],
        aspect="auto",
        interpolation="bilinear",
    )

    ax.contour(
        np.linspace(difficulty_lo, difficulty_hi, N_BINS),
        np.linspace(ability_lo, ability_hi, N_BINS),
        total,
        levels=6,
        colors="black",
        linewidths=0.45,
        alpha=0.25,
    )

    ax.set_xlim(difficulty_lo, difficulty_hi)
    ax.set_ylim(ability_lo, ability_hi)
    ax.set_xlabel("Question Difficulty (standardized empirical logit)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Student Ability (standardized empirical logit)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Student Ability vs Question Difficulty\n"
        f"{len(user_total):,} students, {len(question_total):,} questions, all interactions\n"
        "Color + alpha both encode response density",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    correct_proxy = plt.Line2D([0], [0], color="green", lw=6, alpha=0.55, label="Correct")
    incorrect_proxy = plt.Line2D([0], [0], color="red", lw=6, alpha=0.55, label="Incorrect")
    ax.legend(handles=[correct_proxy, incorrect_proxy], loc="upper left", frameon=True)

    output_path = OUTPUT_DIR / "ability_difficulty_heatmap_large_blurred.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to: {output_path}")

    plt.show()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Students used      : {len(user_total):,}")
    print(f"Questions used     : {len(question_total):,}")
    total_responses = int(user_total.sum())
    correct_responses = int(user_correct.sum())
    incorrect_responses = total_responses - correct_responses
    print(f"Total responses    : {total_responses:,}")
    print(f"Correct responses  : {correct_responses:,}")
    print(f"Incorrect responses: {incorrect_responses:,}")
    print(f"Accuracy           : {100.0 * correct_responses / total_responses:.2f}%")


if __name__ == "__main__":
    main()
