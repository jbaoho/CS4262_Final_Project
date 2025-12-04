#!/usr/bin/env python3
# AI Acknowledgement: AI was used for help with coding and for documentation purposes in this file
"""
analysis.py

Produces:
  - outputs/kaggle_tables.png: PNG containing three even tables:
      * Main 4x4 Kaggle accuracy table (DR x Model)
      * Averages by DR (4x2, sorted)
      * Averages by Model (4x2, sorted)
  - outputs/kaggle_vs_cv_scatter.png: Kaggle vs predicted CV mean accuracy (all 16 combos)
  - outputs/train_time_bar.png: Grouped bar chart of mean training time per DR x Model
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Configuration
# ----------------------------

RESULTS_CSV_PATH = "outputs/results.csv"
OUT_DIR = "outputs"

# Map between human labels and results.csv keys
DR_TO_RESULTS = {
    "RawClean": "raw_clean",
    "PCA": "pca8",
    "AE": "ae16",
    "SigmoidKPCA": "sigmoid_kpca",
}
RESULTS_TO_DR = {v: k for k, v in DR_TO_RESULTS.items()}

MODEL_TO_RESULTS = {
    "LogisticReg": "logistic",
    "SVM": "svm",
    "KNN": "knn",
    "NN": "nn",
}
RESULTS_TO_MODEL = {v: k for k, v in MODEL_TO_RESULTS.items()}

# Kaggle scores (as provided)
KAGGLE_SCORES = {
    "RawClean":     {"LogisticReg": 0.79728, "SVM": 0.73322, "KNN": 0.77180, "NN": 0.81271},
    "PCA":          {"LogisticReg": 0.44821, "SVM": 0.44119, "KNN": 0.51578, "NN": 0.47767},
    "AE":           {"LogisticReg": 0.78816, "SVM": 0.63993, "KNN": 0.78629, "NN": 0.79798},
    "SigmoidKPCA":  {"LogisticReg": 0.78816, "SVM": 0.71732, "KNN": 0.78746, "NN": 0.79822},
}

# ----------------------------
# Helpers
# ----------------------------

def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)

def build_kaggle_df() -> pd.DataFrame:
    rows = []
    for dr, model_scores in KAGGLE_SCORES.items():
        for model, score in model_scores.items():
            rows.append({"DRMethod": dr, "Model": model, "kaggle_score": float(score)})
    return pd.DataFrame(rows)

def load_results_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"dataset", "model", "accuracy_mean", "train_time_mean_sec"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"results.csv missing required columns: {missing}")
    # Map keys to human-readable labels
    df["DRMethod"] = df["dataset"].map(lambda x: RESULTS_TO_DR.get(x, x))
    df["Model"] = df["model"].map(lambda x: RESULTS_TO_MODEL.get(x, x))
    return df

# ----------------------------
# Table rendering to PNG
# ----------------------------

def render_df_table(ax, df: pd.DataFrame, title: str, index_name: str,
                    float_fmt="{:.5f}", fontsize=12, header_color="#f0f0f0",
                    row_colors=("#ffffff", "#fafafa"), edge_color="#cccccc",
                    col_align="center", scale=(1.0, 1.4)):
    """
    Render a DataFrame as an even-looking table on the given Matplotlib axis.
    - Shows index as first column.
    - Uniform column widths.
    - Alternating row colors, styled header.
    """
    # Prepare cell text with index as the first column
    cols = list(df.columns)
    rows = []
    for idx, row in df.iterrows():
        row_vals = []
        # Index value
        row_vals.append(str(idx))
        # Data values
        for c in cols:
            val = row[c]
            if isinstance(val, (int, float, np.floating)):
                row_vals.append(float_fmt.format(val))
            else:
                # Already formatted string (e.g., "{:.5f}" applied)
                row_vals.append(str(val))
        rows.append(row_vals)

    # Column labels include index name
    col_labels = [index_name] + cols
    ncols = len(col_labels)
    col_widths = [1.0 / ncols] * ncols

    # Create table
    ax.axis("off")
    table = ax.table(cellText=rows,
                     colLabels=col_labels,
                     colWidths=col_widths,
                     cellLoc=col_align,
                     loc="center")

    # Style header
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor(header_color)
        else:
            # Alternating row colors (data rows start at row=1)
            cell.set_facecolor(row_colors[(row - 1) % 2])

    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(scale[0], scale[1])

    # Title
    ax.set_title(title, fontsize=fontsize + 2, pad=8)

def save_kaggle_tables_png(kaggle_df: pd.DataFrame, out_path: str):
    """
    Create a single PNG with:
      - Main 4x4 Kaggle accuracy table
      - Averages by DR (4x2, sorted)
      - Averages by Model (4x2, sorted)
    """
    # Main 4x4 (ordered)
    pivot = kaggle_df.pivot(index="DRMethod", columns="Model", values="kaggle_score").loc[
        ["RawClean", "PCA", "AE", "SigmoidKPCA"], ["LogisticReg", "SVM", "KNN", "NN"]
    ]

    # Averages (sorted)
    avg_by_dr = kaggle_df.groupby("DRMethod")["kaggle_score"].mean().sort_values(ascending=False).to_frame("AverageScore")
    avg_by_model = kaggle_df.groupby("Model")["kaggle_score"].mean().sort_values(ascending=False).to_frame("AverageScore")

    # Figure layout: main table on top; two small tables below side-by-side
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[2.2, 1.0], width_ratios=[1.0, 1.0], hspace=0.3, wspace=0.2)

    ax_main = fig.add_subplot(gs[0, :])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    # Render tables
    render_df_table(ax_main, pivot, title="Kaggle Accuracy (DR x Model)", index_name="DRMethod",
                    float_fmt="{:.5f}", fontsize=12, scale=(1.0, 1.4))
    render_df_table(ax_left, avg_by_dr, title="Averages by DR method (sorted)", index_name="DRMethod",
                    float_fmt="{:.5f}", fontsize=12, scale=(1.0, 1.6))
    render_df_table(ax_right, avg_by_model, title="Averages by Model (sorted)", index_name="Model",
                    float_fmt="{:.5f}", fontsize=12, scale=(1.0, 1.6))

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved Kaggle tables PNG to {out_path}")

# ----------------------------
# Plots
# ----------------------------

def save_kaggle_vs_cv_scatter(kaggle_df: pd.DataFrame, results_df: pd.DataFrame, out_path: str):
    """
    Scatter: Kaggle score vs CV accuracy_mean for all 16 combos.
    Color by DR method, shape by model; include y=x reference.
    """
    merged = pd.merge(
        kaggle_df, results_df[["DRMethod", "Model", "accuracy_mean"]],
        on=["DRMethod", "Model"], how="inner", validate="one_to_one"
    )
    if merged.shape[0] != 16:
        print(f"Warning: merged pairs = {merged.shape[0]}, expected 16.")

    sns.set(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 7))
    ax = sns.scatterplot(
        data=merged,
        x="accuracy_mean",
        y="kaggle_score",
        hue="DRMethod",
        style="Model",
        s=120, alpha=0.85, edgecolor="black"
    )
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
    ax.set_xlim(0.3, 0.9)
    ax.set_ylim(0.3, 0.9)
    ax.set_xlabel("Predicted mean CV accuracy")
    ax.set_ylabel("Kaggle accuracy (test)")
    ax.set_title("Kaggle vs Predicted CV Accuracy (16 model/DR combos)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved scatter plot to {out_path}")

def save_train_time_bar(results_df: pd.DataFrame, out_path: str):
    """
    Grouped bar chart of mean training time (seconds) by DR method and model.
    """
    dr_order = ["RawClean", "PCA", "AE", "SigmoidKPCA"]
    model_order = ["LogisticReg", "SVM", "KNN", "NN"]

    sns.set(style="whitegrid", context="talk")
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=results_df,
        x="DRMethod",
        y="train_time_mean_sec",
        hue="Model",
        order=dr_order,
        hue_order=model_order,
        edgecolor="black"
    )
    ax.set_xlabel("DR Method")
    ax.set_ylabel("Mean training time (sec)")
    ax.set_title("Training Time by DR Method and Model")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved training time bar chart to {out_path}")

# ----------------------------
# Main
# ----------------------------

def main():
    ensure_out_dir(OUT_DIR)

    # Load data
    if not os.path.exists(RESULTS_CSV_PATH):
        raise FileNotFoundError(f"{RESULTS_CSV_PATH} not found. Run your training script to create it first.")
    results_df = load_results_csv(RESULTS_CSV_PATH)
    kaggle_df = build_kaggle_df()

    # Output: PNG tables
    kaggle_png_path = os.path.join(OUT_DIR, "kaggle_tables.png")
    save_kaggle_tables_png(kaggle_df, kaggle_png_path)

    # Output: scatter vs CV accuracy
    scatter_path = os.path.join(OUT_DIR, "kaggle_vs_cv_scatter.png")
    save_kaggle_vs_cv_scatter(kaggle_df, results_df, scatter_path)

    # Output: training time bar chart
    time_bar_path = os.path.join(OUT_DIR, "train_time_bar.png")
    save_train_time_bar(results_df, time_bar_path)

if __name__ == "__main__":
    main()