import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.lines import Line2D
import numpy as np


model_colors = {
    "unet": "#1F4F88",
    "pix2pix": "#D0B05F",
    "sde": "#D5302D",
}


def plot_chex_performance(results, results_dir, name):

    model_labels = {"unet": "U-Net", "pix2pix": "GAN", "sde": "SDE"}

    metrics = [
        "ec",
        "cardiomegaly",
        "lung-opacity",
        "lung-lesion",
        "edema",
        "consolidation",
        "pneumonia",
        "atelectasis",
        "pneumothorax",
        "pleural-effusion",
        "pleural-other",
        "fracture",
        "average",
    ]

    metric_labels = {
        "ec": "Enlarged Cardiomediastinum",
        "cardiomegaly": "Cardiomegaly",
        "lung-opacity": "Lung Opacity",
        "lung-lesion": "Lung Lesion",
        "edema": "Edema",
        "consolidation": "Consolidation",
        "pneumonia": "Pneumonia",
        "atelectasis": "Atelectasis",
        "pneumothorax": "Pneumothorax",
        "pleural-effusion": "Pleural Effusion",
        "pleural-other": "Pleural Other",
        "fracture": "Fracture",
    }

    # Set font sizes and line widths
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 32,
            "axes.labelsize": 24,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 24,
            "legend.title_fontsize": 24,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
        }
    )

    for metric in metrics:
        # Create new figure for each pair
        plt.figure(figsize=(12, 6))

        # Add bold metric title if not average
        if metric != "average":
            plt.title(metric_labels[metric].title(), fontweight="bold", pad=20)
        else:
            plt.title("Average Classification (CheXpert)", fontweight="bold", pad=20)

        # Get baseline value if not average
        baseline_value = None
        baseline_data = results[
            (results["metric"] == metric) & (results["model"] == "baseline")
        ]
        if not baseline_data.empty:
            baseline_value = baseline_data["value"].iloc[0]

        # Filter data for the current metrics
        data1 = results[
            (results["metric"] == metric) & (results["model"] != "baseline")
        ].copy()
        if metric == "average":
            data2 = results[
                (results["metric"] == "psnr") & (results["model"] != "baseline")
            ].copy()
        else:
            data2 = results[
                (results["metric"] == f"psnr-{metric}")
                & (results["model"] != "baseline")
            ].copy()

        # Create first axis and plot
        ax1 = plt.gca()

        # Set light gray background
        ax1.set_facecolor("#f5f5f5")  # Light gray background

        line1 = sns.lineplot(
            data=data1,
            x="photon_count",
            y="value",
            hue="model",
            palette=model_colors,
            marker="o",
            ax=ax1,
            linewidth=2.5,
        )

        # Create second y-axis and plot
        ax2 = ax1.twinx()
        line2 = sns.lineplot(
            data=data2,
            x="photon_count",
            y="value",
            hue="model",
            palette=model_colors,
            marker="s",
            linestyle="--",
            ax=ax2,
            linewidth=2.5,
        )

        # Set labels and title with bold font
        ax1.set_xlabel("Photon Count", fontweight="bold", labelpad=10)
        ax1.set_ylabel("AUROC", fontweight="bold", labelpad=10)
        ax2.set_ylabel("PSNR", fontweight="bold", labelpad=10)

        # Make tick labels bold
        ax1.tick_params(axis="both", which="major")
        ax2.tick_params(axis="both", which="major")

        # Get lines and labels for both plots
        lines1, labels1 = ax1.get_legend_handles_labels()

        # Update custom lines to match new thickness
        custom_lines = [
            Line2D(
                [0],
                [0],
                color="gray",
                linestyle="-",
                marker="o",
                label="AUROC",
                linewidth=2.5,
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                color="gray",
                linestyle="--",
                marker="s",
                label="PSNR",
                linewidth=2.5,
                markersize=8,
            ),
        ]

        """ax1.legend(lines1[:3] + custom_lines,
                    [model_labels[model] for model in labels1[:3]] + ["AUROC", "PSNR"],
                    bbox_to_anchor=(1.15, 1))"""
        ax1.get_legend().remove()
        ax2.get_legend().remove()

        # Set x-axis to treat values as categorical
        available_photon_counts = sorted(data1["photon_count"].unique(), reverse=True)
        ax1.set_xticks(range(len(available_photon_counts)))
        ax1.set_xticklabels(available_photon_counts)

        # Update the data points to use categorical positions
        for line in ax1.lines:
            if len(line.get_xdata()) > 0:  # Check if line has data
                old_x = line.get_xdata()
                new_x = [list(available_photon_counts).index(x) for x in old_x]
                line.set_xdata(new_x)

        for line in ax2.lines:
            if len(line.get_xdata()) > 0:  # Check if line has data
                old_x = line.get_xdata()
                new_x = [list(available_photon_counts).index(x) for x in old_x]
                line.set_xdata(new_x)

        # Add white grid lines - only manual PSNR values to avoid clutter
        for y in [26, 27, 28, 29, 30, 31, 32]:
            ax2.axhline(
                y=y, color="white", linestyle="-", linewidth=1, alpha=0.8, zorder=0
            )

        # Set x-axis limits to remove extra whitespace
        ax1.set_xlim(-0.2, len(available_photon_counts) - 0.8)

        # Calculate percentage drops for both metrics
        y1_max = data1["value"].max()
        y1_min = data1["value"].min()
        y2_max = data2["value"].max()
        y2_min = data2["value"].min()

        # Calculate the larger percentage drop
        drop1 = (y1_max - y1_min) / y1_max
        drop2 = (y2_max - y2_min) / y2_max
        max_drop = max(drop1, drop2)

        # Set limits to show the same percentage drop for both axes
        y1_bottom = y1_max * (1 - max_drop)
        y2_bottom = y2_max * (1 - max_drop)

        # Add small padding at top (5%)
        ax1.set_ylim(y1_bottom * 0.95, y1_max * 1.05)
        ax2.set_ylim(y2_bottom * 0.95, y2_max * 1.05)

        # Adjust figure size to accommodate legend
        plt.gcf().set_size_inches(12, 6)  # Wider figure to fit legend

        # Adjust layout to prevent legend overlap with consistent margins
        plt.subplots_adjust(left=0.12, bottom=0.15, right=0.88, top=0.85)

        # Add baseline reference line if available
        ax1.axhline(
            y=baseline_value,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5,
            zorder=0,
        )

        # Save plots in both formats
        for fmt in ["eps", "pdf", "png"]:
            save_dir = os.path.join(results_dir, "chex_performance", fmt)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                f"{save_dir}/{name}_performance_{metric}_psnr.{fmt}",
                bbox_inches="tight",
                dpi=300,
                format=fmt,
            )
        plt.close()

    # Create a separate legend figure
    plt.figure(figsize=(12, 1))
    ax = plt.gca()
    ax.set_axis_off()

    # Create legend elements
    model_lines = [
        Line2D([0], [0], color=color, label=model_labels[model])
        for model, color in model_colors.items()
    ]
    metric_lines = [
        Line2D([0], [0], color="gray", linestyle="-", marker="o", label="AUROC"),
        Line2D([0], [0], color="gray", linestyle="--", marker="s", label="PSNR"),
        Line2D(
            [0],
            [0],
            color="gray",
            linestyle="--",
            label="Baseline",
            alpha=0.5,
            linewidth=1.5,
        ),
    ]

    # Create horizontal legend
    plt.legend(
        handles=model_lines + metric_lines,
        loc="center",
        ncol=len(model_lines) + len(metric_lines),
        bbox_to_anchor=(0.5, 0.5),
    )

    # Save legend
    for fmt in ["eps", "pdf", "png"]:
        save_dir = os.path.join(results_dir, "chex_performance", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/{name}_performance_legend.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()
