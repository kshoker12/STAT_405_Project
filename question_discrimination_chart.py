"""
Question discrimination chart.

For each question:
- estimate student ability from the full training export
- split students into top/bottom 27% by ability
- compute discrimination D = P(correct | top 27%) - P(correct | bottom 27%)
- plot D on the x-axis and question difficulty on the y-axis

This is designed for the full training file, so it processes data in chunks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


PROJECT_ROOT = Path("/Users/atlasbuchholz/PycharmProjects/UBC/STAT_405_Project")
OUTPUT_DIR = PROJECT_ROOT / "output"
TRAIN_PATH = PROJECT_ROOT / "data" / "train_data" / "train_task_1_2.csv"

CHUNK_SIZE = 1_000_000
EPS = 0.5
TOP_SHARE = 0.27
MIN_TOP_BOTTOM_OBS = 20
MIN_QUESTION_RESPONSES = 100
THETA_HIGH = 1.03
THETA_LOW = -1.03
CHANCE_FLOOR = 0.25
RESIDUAL_STD_THRESHOLD = 1.5


def smoothed_logit(correct_count: float, total_count: float) -> float:
    rate = (correct_count + EPS) / (total_count + 2 * EPS)
    rate = np.clip(rate, 1e-6, 1 - 1e-6)
    return float(np.log(rate / (1.0 - rate)))


def logistic(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def expected_discrimination_3pl(difficulty: np.ndarray, a: float) -> np.ndarray:
    scale = 1.0 - CHANCE_FLOOR
    return scale * (
        logistic(a * (THETA_HIGH - difficulty)) - logistic(a * (THETA_LOW - difficulty))
    )


def collect_totals(path: Path) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
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


def build_ability_and_difficulty(
    user_correct: pd.Series,
    user_total: pd.Series,
    question_correct: pd.Series,
    question_total: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    user_ability_raw = pd.Series(
        {int(uid): smoothed_logit(user_correct.loc[uid], user_total.loc[uid]) for uid in user_total.index}
    )
    question_difficulty_raw = pd.Series(
        {int(qid): -smoothed_logit(question_correct.loc[qid], question_total.loc[qid]) for qid in question_total.index}
    )

    user_ability = (user_ability_raw - user_ability_raw.mean()) / user_ability_raw.std(ddof=0)
    question_difficulty = (question_difficulty_raw - question_difficulty_raw.mean()) / question_difficulty_raw.std(ddof=0)
    return user_ability, question_difficulty


def collect_top_bottom_question_stats(
    path: Path,
    top_users: set[int],
    bottom_users: set[int],
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    top_correct = pd.Series(dtype="float64")
    top_total = pd.Series(dtype="float64")
    bottom_correct = pd.Series(dtype="float64")
    bottom_total = pd.Series(dtype="float64")

    print("Collecting top/bottom group response counts...")
    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            path,
            usecols=["QuestionId", "UserId", "IsCorrect"],
            dtype={"QuestionId": np.int32, "UserId": np.int32, "IsCorrect": np.int8},
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        top_chunk = chunk[chunk["UserId"].isin(top_users)]
        bottom_chunk = chunk[chunk["UserId"].isin(bottom_users)]

        if not top_chunk.empty:
            grouped = top_chunk.groupby("QuestionId", sort=False)["IsCorrect"].agg(["sum", "count"])
            top_correct = top_correct.add(grouped["sum"], fill_value=0)
            top_total = top_total.add(grouped["count"], fill_value=0)

        if not bottom_chunk.empty:
            grouped = bottom_chunk.groupby("QuestionId", sort=False)["IsCorrect"].agg(["sum", "count"])
            bottom_correct = bottom_correct.add(grouped["sum"], fill_value=0)
            bottom_total = bottom_total.add(grouped["count"], fill_value=0)

        if chunk_idx % 5 == 0:
            print(f"  processed {chunk_idx:,} chunks")

    return top_correct, top_total, bottom_correct, bottom_total


def main() -> None:
    print("Computing question discrimination from the full training export...")
    user_correct, user_total, question_correct, question_total = collect_totals(TRAIN_PATH)

    print(f"  unique students : {len(user_total):,}")
    print(f"  unique questions: {len(question_total):,}")

    user_ability, question_difficulty = build_ability_and_difficulty(
        user_correct, user_total, question_correct, question_total
    )

    top_cutoff = user_ability.quantile(1.0 - TOP_SHARE)
    bottom_cutoff = user_ability.quantile(TOP_SHARE)
    top_users = set(user_ability.index[user_ability >= top_cutoff])
    bottom_users = set(user_ability.index[user_ability <= bottom_cutoff])

    print(f"  top group size    : {len(top_users):,}")
    print(f"  bottom group size : {len(bottom_users):,}")
    print(f"  ability cutoff(s) : top >= {top_cutoff:.3f}, bottom <= {bottom_cutoff:.3f}")

    top_correct, top_total, bottom_correct, bottom_total = collect_top_bottom_question_stats(
        TRAIN_PATH, top_users, bottom_users
    )

    question_ids = question_total.index.astype(int)
    stats = pd.DataFrame(index=question_ids)
    stats["difficulty"] = question_difficulty.reindex(question_ids).astype(float)
    stats["question_total"] = question_total.reindex(question_ids).fillna(0.0)
    stats["top_total"] = top_total.reindex(question_ids).fillna(0.0)
    stats["bottom_total"] = bottom_total.reindex(question_ids).fillna(0.0)
    stats["top_correct"] = top_correct.reindex(question_ids).fillna(0.0)
    stats["bottom_correct"] = bottom_correct.reindex(question_ids).fillna(0.0)

    stats = stats[(stats["top_total"] >= MIN_TOP_BOTTOM_OBS) & (stats["bottom_total"] >= MIN_TOP_BOTTOM_OBS)].copy()
    stats = stats[stats["question_total"] >= MIN_QUESTION_RESPONSES].copy()
    stats["top_rate"] = (stats["top_correct"] + EPS) / (stats["top_total"] + 2 * EPS)
    stats["bottom_rate"] = (stats["bottom_correct"] + EPS) / (stats["bottom_total"] + 2 * EPS)
    stats["discrimination"] = stats["top_rate"] - stats["bottom_rate"]

    print(f"  questions retained after support filter: {len(stats):,}")
    print(
        f"  discrimination range: [{stats['discrimination'].min():.3f}, {stats['discrimination'].max():.3f}]"
    )
    print(f"  difficulty range     : [{stats['difficulty'].min():.3f}, {stats['difficulty'].max():.3f}]")

    # Fit one global 3PL discrimination parameter a by weighted least squares.
    # More responses => more reliable D estimate.
    b_obs = stats["difficulty"].to_numpy(dtype=float)
    d_obs = stats["discrimination"].to_numpy(dtype=float)
    response_counts = stats["question_total"].to_numpy(dtype=float)
    weights = np.sqrt(np.clip(response_counts, 1.0, None))

    popt, _ = curve_fit(
        expected_discrimination_3pl,
        b_obs,
        d_obs,
        p0=[1.0],
        sigma=1.0 / weights,
        absolute_sigma=False,
        bounds=(0.01, 10.0),
        maxfev=20000,
    )
    a_hat = float(popt[0])
    stats["expected_discrimination"] = expected_discrimination_3pl(b_obs, a_hat)
    stats["residual"] = stats["discrimination"] - stats["expected_discrimination"]

    residual_sd = float(stats["residual"].std(ddof=0))
    residual_cutoff = -RESIDUAL_STD_THRESHOLD * residual_sd
    stats["is_flagged_residual"] = stats["residual"] < residual_cutoff
    flagged = stats[stats["is_flagged_residual"]].copy()

    # Plot configuration for the "bad questions" view.
    fig, ax = plt.subplots(figsize=(12, 9), dpi=120)

    hb = ax.hexbin(
        stats["discrimination"],
        stats["difficulty"],
        gridsize=70,
        bins="log",
        mincnt=1,
        cmap="viridis",
        linewidths=0,
        alpha=0.9,
        rasterized=True,
    )

    # Overlay expected D(b) curve with difficulty on y-axis and D on x-axis.
    b_grid = np.linspace(stats["difficulty"].min(), stats["difficulty"].max(), 300)
    d_grid = expected_discrimination_3pl(b_grid, a_hat)
    ax.plot(
        d_grid,
        b_grid,
        color="white",
        linewidth=2.2,
        alpha=0.95,
        label=f"Expected 3PL D(b), â={a_hat:.2f}",
    )
    ax.plot(
        d_grid,
        b_grid,
        color="black",
        linewidth=0.8,
        alpha=0.9,
    )


    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax.axvline(0.2, color="gray", linewidth=1, linestyle=":", alpha=0.7)
    ax.axhline(0, color="gray", linewidth=1, linestyle=":", alpha=0.5)

    ax.set_xlabel("Discrimination index D = P(correct | top 27%) - P(correct | bottom 27%)", fontweight="bold")
    ax.set_ylabel("Question difficulty (standardized empirical logit)", fontweight="bold")
    ax.set_title(
        "Question Discrimination Chart\n"
        f"{len(user_total):,} students, {len(question_total):,} questions | "
        f"filtered to ≥{MIN_QUESTION_RESPONSES} responses per question",
        fontweight="bold",
        pad=14,
    )

    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label("Questions per hex bin (log scale)", rotation=90)
    ax.legend(loc="upper right", frameon=True)

    bad_questions = stats.nsmallest(20, "residual")
    print("\nMost negative residual questions (observed D far below expected curve):")
    print(
        bad_questions[
            ["discrimination", "expected_discrimination", "residual", "difficulty", "top_rate", "bottom_rate"]
        ]
        .head(10)
        .to_string()
    )

    output_path = OUTPUT_DIR / "question_discrimination_chart.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to: {output_path}")

    plt.show()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Students used        : {len(user_total):,}")
    print(f"Questions used       : {len(question_total):,}")
    print(f"Questions plotted    : {len(stats):,}")
    print(f"Fitted discrimination â : {a_hat:.3f}")
    print(f"Residual SD             : {residual_sd:.4f}")
    print(f"Residual cutoff         : {residual_cutoff:.4f}")
    print(f"Flagged (residual)      : {len(flagged):,}")
    print(f"Top group size       : {len(top_users):,}")
    print(f"Bottom group size    : {len(bottom_users):,}")
    print(f"Mean discrimination  : {stats['discrimination'].mean():.3f}")
    print(f"Median discrimination: {stats['discrimination'].median():.3f}")
    print(f"Mean residual        : {stats['residual'].mean():.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
