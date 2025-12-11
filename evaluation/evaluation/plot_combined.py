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


def plot_fairness_change(
    df_standard, df_eodd, df_reweighted, df_adv, results_dir, name
):
    """
    Create a combined plot showing fairness metrics across different models and mitigation techniques.
    This function combines the functionality of plot_ucsf_additional_bias, plot_chex_additional_bias,
    and plot_fairness_change_mitigation into a single plot with the following changes:
    - Uses mean values with error bars instead of bars
    - Uses normal (non-bootstrapped) values
    - Compares all methods against the same baseline
    - Uses shapes to distinguish between mitigation techniques
    - Maintains the same color scheme and faceting structure

    Args:
        df_standard: DataFrame with standard (baseline) results
        df_eodd: DataFrame with EODD mitigation results
        df_reweighted: DataFrame with reweighted results
        df_adv: DataFrame with adversarial mitigation results
        results_dir: Directory to save the plots
        name: Name prefix for output files
    """
    # Define metrics to compare (using non-bootstrapped versions)
    metrics = ["delta-EODD", "delta-EOP", "delta-delta-dice", "delta-SER"]

    # Define metric mapping for display
    metric_display_map = {
        "delta-EODD": "EODD",
        "delta-EOP": "EOP",
        "delta-delta-dice": "DICE",
        "delta-SER": "SER",
    }

    # Define interpreters for all datasets
    interpreters = [
        # CheXpert interpreters
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
        # UCSF interpreters
        "ttype",
        "tgrade",
        # Segmentation interpreter
        "dice",
    ]

    # Define labels for interpreters
    metric_labels = {
        # CheXpert labels
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
        "average": "Average",
        # UCSF labels
        "tgrade": "Tumor Grade",
        "ttype": "Tumor Type",
        # Segmentation label
        "dice": "Segmentation",
    }

    # Define model labels
    model_labels = {"unet": "U-Net", "pix2pix": "GAN", "sde": "SDE"}

    # Define colors for each model (using the same colors as original plots)
    model_colors = {
        "unet": "#1F4F88",
        "pix2pix": "#D0B05F",
        "sde": "#D5302D",
    }

    # Define markers for different mitigation techniques
    mitigation_markers = {
        "standard": "o",  # Circle for standard
        "reweighted": "s",  # Square for reweighted
        "eodd": "D",  # Diamond for EODD
        "adv": "^",  # Triangle for adversarial
    }

    # Define significance levels and their symbols
    significance_levels = {0.05: "*"}  # One star for p < 0.05

    # Set plotting parameters
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 32,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 18,
            "legend.fontsize": 24,
            "legend.title_fontsize": 24,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
        }
    )

    for interpreter in interpreters:
        # Skip if interpreter not in any of the dataframes
        if not any(
            interpreter in df["interpreter"].unique()
            for df in [df_standard, df_eodd, df_reweighted, df_adv]
        ):
            continue

        # Filter data for current interpreter from each dataframe
        standard_data = df_standard[df_standard["interpreter"] == interpreter].copy()
        eodd_data = df_eodd[df_eodd["interpreter"] == interpreter].copy()
        reweight_data = df_reweighted[
            df_reweighted["interpreter"] == interpreter
        ].copy()
        adv_data = df_adv[df_adv["interpreter"] == interpreter].copy()

        # Skip if no data for this interpreter
        if any(
            len(df) == 0 for df in [standard_data, eodd_data, reweight_data, adv_data]
        ):
            continue

        # Create combined data for plotting
        plot_data = pd.DataFrame()

        # Process each metric
        for metric in metrics:
            # Get data for each mitigation type
            for df, mitigation_type in [
                (standard_data, "standard"),
                (reweight_data, "reweighted"),
                (eodd_data, "eodd"),
                (adv_data, "adv"),
            ]:
                # Get metric data and its standard error
                metric_data = df[df["metric"] == metric].copy()

                # Skip if no metric data found
                if len(metric_data) == 0:
                    continue

                # Handle different standard error naming conventions
                if metric.startswith("delta-delta-dice"):
                    std_err_metric = "delta-delta-dice-std-err"
                elif metric.startswith("delta-SER"):
                    std_err_metric = "delta-SER-std-err"
                else:
                    std_err_metric = f"{metric.split('-')[1]}-std-err-bootstrapped"

                std_err_data = df[
                    (df["metric"] == std_err_metric) & (df["model"] != "baseline")
                ].copy()
                significance_data = df[(df["metric"] == f"{metric}-p-value")].copy()

                # Add mitigation type and metric type
                metric_data["type"] = mitigation_type
                metric_data["metric_type"] = metric_display_map[
                    metric
                ]  # Use mapped name for display

                # Add standard error
                metric_data["std_err"] = std_err_data["value"].values

                # Add p-value and significance
                metric_data["p_value"] = significance_data["value"].values
                metric_data["significance"] = metric_data["p_value"].apply(
                    lambda p: next(
                        (
                            stars
                            for threshold, stars in significance_levels.items()
                            if p < threshold
                        ),
                        "",
                    )
                )

                # Add to plot data
                plot_data = pd.concat([plot_data, metric_data])

        # Calculate y limits with padding for error bars
        y_min = (plot_data["value"] - plot_data["std_err"]).min()
        y_max = (plot_data["value"] + plot_data["std_err"]).max()

        # Round to nearest 0.05
        y_min = np.floor(y_min / 0.05) * 0.05
        y_max = np.ceil(y_max / 0.05) * 0.05

        # For tgrade, ttype, and dice, only show age and gender attributes
        if interpreter in ["tgrade", "ttype", "dice"]:
            plot_data = plot_data[plot_data["attribute"].isin(["gender", "age"])]
            col_order = ["gender", "age"]
        else:
            col_order = ["gender", "age", "ethnicity"]

        # Create facet grid
        g = sns.FacetGrid(
            plot_data,
            col="attribute",
            col_order=col_order,
            height=6,
            aspect=1.2,
            ylim=(y_min, y_max),
        )

        def plot_points(data, **kwargs):
            ax = plt.gca()
            unique_metrics = data["metric_type"].unique()
            n_metrics = len(unique_metrics)
            n_models = len(model_colors)
            n_mitigations = len(mitigation_markers)

            # Set light gray background
            ax.set_facecolor("#f5f5f5")  # Light gray background

            # Calculate spacing parameters
            metric_width = 0.8  # Width allocated for each metric group
            model_spacing = metric_width / n_models  # Space between model groups
            mitigation_spacing = (
                model_spacing / n_mitigations
            )  # Space between mitigation techniques

            # Add white horizontal grid lines
            for y in np.arange(y_min, y_max + 0.05, 0.05):
                ax.axhline(
                    y=y, color="white", linestyle="-", linewidth=1, alpha=0.8, zorder=0
                )

            # Handle y-axis visibility
            if ax.get_position().x0 > 0.1:
                ax.spines["left"].set_visible(False)
                ax.yaxis.set_visible(False)
            else:
                ax.set_yticks(np.arange(y_min, y_max + 0.05, 0.05))

            # Add zero line
            ax.axhline(
                y=0, color="black", linestyle="-", linewidth=1.0, alpha=0.5, zorder=1
            )

            # Plot points for each model
            for i, (model, color) in enumerate(model_colors.items()):
                model_data = data[data["model"] == model]
                if len(model_data) == 0:
                    continue

                # Calculate base position for this model
                model_offset = (i - (n_models - 1) / 2) * model_spacing

                # Plot each metric
                for j, metric_type in enumerate(unique_metrics):
                    metric_base_x = (
                        j * 1.2 + model_offset
                    )  # 1.2 provides spacing between metric groups

                    # Plot each mitigation type
                    for k, (mitigation_type, marker) in enumerate(
                        mitigation_markers.items()
                    ):
                        mitigation_data = model_data[
                            model_data["type"] == mitigation_type
                        ]
                        metric_mask = mitigation_data["metric_type"] == metric_type

                        if not any(metric_mask):
                            continue

                        # Calculate x position for this specific point
                        mitigation_offset = (
                            k - (n_mitigations - 1) / 2
                        ) * mitigation_spacing
                        x_pos = metric_base_x + mitigation_offset

                        value = mitigation_data[metric_mask]["value"].iloc[0]
                        std_err = mitigation_data[metric_mask]["std_err"].iloc[0]
                        significance = mitigation_data[metric_mask][
                            "significance"
                        ].iloc[0]

                        # Plot error bar and point
                        # Make standard markers hollow (no infill)
                        markerfacecolor = (
                            "none" if mitigation_type == "standard" else color
                        )
                        plt.errorbar(
                            x_pos,
                            value,
                            yerr=std_err,
                            color=color,
                            marker=marker,
                            markerfacecolor=markerfacecolor,
                            capsize=5,
                            capthick=1.5,
                            elinewidth=1.5,
                            markersize=10,
                            zorder=2,
                        )

                        # Add significance stars if any
                        if significance:
                            # Calculate vertical offset based on model index
                            # This ensures stars are stacked vertically for each metric
                            model_index = list(model_colors.keys()).index(model)
                            base_vertical_offset = 0.005 * (
                                y_max - y_min
                            )  # Further reduced base offset

                            # Adjust vertical offset based on model index to stack groups of stars
                            group_vertical_offset = (
                                base_vertical_offset * model_index * 2
                            )  # Keep groups separated but start closer

                            # Position the stars
                            y_pos = (
                                value
                                + std_err
                                + base_vertical_offset
                                + group_vertical_offset
                            )

                            # Adjust line spacing for vertically stacked stars
                            line_spacing = (
                                0.5 if "\n" in significance else 1.0
                            )  # Reduced line spacing

                            plt.text(
                                x_pos,
                                y_pos,
                                significance,
                                ha="center",
                                va="bottom",
                                fontsize=16,
                                fontweight="bold",
                                linespacing=line_spacing,
                            )

            # Set x-ticks at the center of each metric group
            plt.xticks(np.arange(n_metrics) * 1.2, unique_metrics)
            plt.setp(ax.get_xticklabels(), weight="bold")

        g.map_dataframe(plot_points)

        # Set titles and labels
        col_names = {"gender": "Sex", "age": "Age", "ethnicity": "Race"}
        g.set_titles(template="{col_name}")

        for ax, title in zip(g.axes.flat, [col_names[col] for col in g.col_names]):
            ax.set_title(title, fontweight="bold", pad=20, fontsize=28)

        # Add interpreter title
        g.fig.suptitle(
            metric_labels[interpreter], fontweight="bold", fontsize=34, y=1.1
        )

        # Add y-axis label to the leftmost plot
        g.axes[0, 0].set_ylabel("Bias Change", fontweight="bold")

        # Remove individual legends
        for ax in g.axes.flat:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        # Save the plots
        for fmt in ["eps", "pdf", "png"]:
            save_dir = os.path.join(results_dir, "combined_fairness", fmt)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                f"{save_dir}/{name}_fairness_{interpreter}.{fmt}",
                bbox_inches="tight",
                dpi=300,
                format=fmt,
            )
        plt.close()

    # Create separate legend figure
    plt.figure(figsize=(14, 2.5))
    ax = plt.gca()
    ax.set_axis_off()

    # Create legend elements
    legend_elements = []

    # Add model elements
    for model, color in model_colors.items():
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color=color,
                marker="o",
                linestyle="None",
                markersize=10,
                label=model_labels[model],
            )
        )

    # Add a small space in legend
    legend_elements.append(plt.Line2D([0], [0], color="none", label=""))

    # Add mitigation type elements
    for mitigation_type, marker in mitigation_markers.items():
        label = {
            "standard": "Reconstruction",
            "reweighted": "Reweighted",
            "eodd": "EODD Loss",
            "adv": "Adversarial Loss",
        }[mitigation_type]
        # Make standard marker hollow in legend too
        markerfacecolor = "none" if mitigation_type == "standard" else "black"
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color="black",
                marker=marker,
                linestyle="None",
                markerfacecolor=markerfacecolor,
                markersize=10,
                label=label,
            )
        )

    # Add a small space in legend
    legend_elements.append(plt.Line2D([0], [0], color="none", label=""))

    # Add significance level elements
    legend_elements.append(plt.Line2D([0], [0], color="none", label="Significance:"))
    legend_elements.append(plt.Line2D([0], [0], color="none", label="* p < 0.05"))

    # Create legend
    plt.legend(
        handles=legend_elements,
        loc="center",
        ncol=len(legend_elements),
        bbox_to_anchor=(0.5, 0.5),
    )

    # Save legend
    for fmt in ["eps", "pdf", "png"]:
        save_dir = os.path.join(results_dir, "combined_fairness", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/{name}_fairness_legend.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()


def plot_fairness_change_separate(
    df_standard, df_eodd, df_reweighted, df_adv, results_dir, name
):
    """
    Create separate plots for EODD+SER and EOP+DICE fairness metrics across different models and mitigation techniques.
    This function creates two versions of the fairness plots:
    - EODD version: Shows EODD and SER metrics
    - EOP version: Shows EOP and DICE metrics

    Args:
        df_standard: DataFrame with standard (baseline) results
        df_eodd: DataFrame with EODD mitigation results
        df_reweighted: DataFrame with reweighted results
        df_adv: DataFrame with adversarial mitigation results
        results_dir: Directory to save the plots
        name: Name prefix for output files
    """
    # Define the two metric combinations
    metric_combinations = {
        "eodd": {
            "metrics": ["delta-EODD", "delta-SER"],
            "display_map": {"delta-EODD": "EODD", "delta-SER": "SER"},
        },
        "eop": {
            "metrics": ["delta-EOP", "delta-delta-dice"],
            "display_map": {"delta-EOP": "EOP", "delta-delta-dice": "DICE"},
        },
    }

    # Define interpreters for all datasets
    interpreters = [
        # CheXpert interpreters
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
        # UCSF interpreters
        "ttype",
        "tgrade",
        # Segmentation interpreter
        "dice",
    ]

    # Define labels for interpreters
    metric_labels = {
        # CheXpert labels
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
        "average": "Average",
        # UCSF labels
        "tgrade": "Tumor Grade",
        "ttype": "Tumor Type",
        # Segmentation label
        "dice": "Segmentation",
    }

    # Define model labels
    model_labels = {"unet": "U-Net", "pix2pix": "GAN", "sde": "SDE"}

    # Define colors for each model (using the same colors as original plots)
    model_colors = {
        "unet": "#1F4F88",
        "pix2pix": "#D0B05F",
        "sde": "#D5302D",
    }

    # Define markers for different mitigation techniques
    mitigation_markers = {
        "standard": "o",  # Circle for standard
        "reweighted": "s",  # Square for reweighted
        "eodd": "D",  # Diamond for EODD
        "adv": "^",  # Triangle for adversarial
    }

    # Define significance levels and their symbols
    significance_levels = {0.05: "*"}  # One star for p < 0.05

    # Set plotting parameters
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 36,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 18,
            "legend.fontsize": 24,
            "legend.title_fontsize": 24,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
        }
    )

    # Create plots for each metric combination
    for combo_name, combo_info in metric_combinations.items():
        metrics = combo_info["metrics"]
        metric_display_map = combo_info["display_map"]

        for interpreter in interpreters:
            # Skip if interpreter not in any of the dataframes
            if not any(
                interpreter in df["interpreter"].unique()
                for df in [df_standard, df_eodd, df_reweighted, df_adv]
            ):
                continue

            # Filter data for current interpreter from each dataframe
            standard_data = df_standard[
                df_standard["interpreter"] == interpreter
            ].copy()
            eodd_data = df_eodd[df_eodd["interpreter"] == interpreter].copy()
            reweight_data = df_reweighted[
                df_reweighted["interpreter"] == interpreter
            ].copy()
            adv_data = df_adv[df_adv["interpreter"] == interpreter].copy()

            # Skip if no data for this interpreter
            if any(
                len(df) == 0
                for df in [standard_data, eodd_data, reweight_data, adv_data]
            ):
                continue

            # Create combined data for plotting
            plot_data = pd.DataFrame()

            # Process each metric
            for metric in metrics:
                # Get data for each mitigation type
                for df, mitigation_type in [
                    (standard_data, "standard"),
                    (reweight_data, "reweighted"),
                    (eodd_data, "eodd"),
                    (adv_data, "adv"),
                ]:
                    # Get metric data and its standard error
                    metric_data = df[df["metric"] == metric].copy()

                    # Skip if no metric data found
                    if len(metric_data) == 0:
                        continue

                    # Handle different standard error naming conventions
                    if metric.startswith("delta-delta-dice"):
                        std_err_metric = "delta-delta-dice-std-err"
                    elif metric.startswith("delta-SER"):
                        std_err_metric = "delta-SER-std-err"
                    else:
                        std_err_metric = f"{metric.split('-')[1]}-std-err-bootstrapped"

                    std_err_data = df[
                        (df["metric"] == std_err_metric) & (df["model"] != "baseline")
                    ].copy()
                    significance_data = df[(df["metric"] == f"{metric}-p-value")].copy()

                    # Add mitigation type and metric type
                    metric_data["type"] = mitigation_type
                    metric_data["metric_type"] = metric_display_map[
                        metric
                    ]  # Use mapped name for display

                    # Add standard error
                    metric_data["std_err"] = std_err_data["value"].values

                    # Add p-value and significance
                    metric_data["p_value"] = significance_data["value"].values
                    metric_data["significance"] = metric_data["p_value"].apply(
                        lambda p: next(
                            (
                                stars
                                for threshold, stars in significance_levels.items()
                                if p < threshold
                            ),
                            "",
                        )
                    )

                    # Add to plot data
                    plot_data = pd.concat([plot_data, metric_data])

            # Skip if no plot data
            if plot_data.empty:
                continue

            # Calculate y limits with padding for error bars
            y_min = (plot_data["value"] - plot_data["std_err"]).min()
            y_max = (plot_data["value"] + plot_data["std_err"]).max()

            # Round to nearest 0.05
            y_min = np.floor(y_min / 0.05) * 0.05
            y_max = np.ceil(y_max / 0.05) * 0.05

            # For tgrade, ttype, and dice, only show age and gender attributes
            if interpreter in ["tgrade", "ttype", "dice"]:
                plot_data = plot_data[plot_data["attribute"].isin(["gender", "age"])]
                col_order = ["gender", "age"]
            else:
                col_order = ["gender", "age", "ethnicity"]

            # Create facet grid
            g = sns.FacetGrid(
                plot_data,
                col="attribute",
                col_order=col_order,
                height=6,
                aspect=1.2,
                ylim=(y_min, y_max),
            )

            def plot_points(data, **kwargs):
                ax = plt.gca()
                unique_metrics = data["metric_type"].unique()
                n_metrics = len(unique_metrics)
                n_models = len(model_colors)
                n_mitigations = len(mitigation_markers)

                # Set light gray background
                ax.set_facecolor("#f5f5f5")  # Light gray background

                # Calculate spacing parameters - group by model with appropriate spacing
                model_group_width = 0.8  # Width allocated for each model group
                mitigation_spacing = (
                    0.12  # Spacing between mitigation techniques within a model
                )
                metric_spacing_within_model = (
                    0.3  # Spacing between metrics within the same model
                )

                # Add white horizontal grid lines
                for y in np.arange(y_min, y_max + 0.05, 0.05):
                    ax.axhline(
                        y=y,
                        color="white",
                        linestyle="-",
                        linewidth=1,
                        alpha=0.8,
                        zorder=0,
                    )

                # Handle y-axis visibility
                if ax.get_position().x0 > 0.1:
                    ax.spines["left"].set_visible(False)
                    ax.yaxis.set_visible(False)
                else:
                    ax.set_yticks(np.arange(y_min, y_max + 0.05, 0.05))

                # Add zero line
                ax.axhline(
                    y=0,
                    color="black",
                    linestyle="-",
                    linewidth=1.0,
                    alpha=0.5,
                    zorder=1,
                )

                # Plot points for each model
                for i, (model, color) in enumerate(model_colors.items()):
                    model_data = data[data["model"] == model]
                    if len(model_data) == 0:
                        continue

                    # Calculate base position for this model group
                    model_base_x = i * model_group_width

                    # Plot each metric within the model group
                    for j, metric_type in enumerate(unique_metrics):
                        metric_base_x = (
                            model_base_x
                            + (j - (n_metrics - 1) / 2) * metric_spacing_within_model
                        )

                        # Plot each mitigation type
                        for k, (mitigation_type, marker) in enumerate(
                            mitigation_markers.items()
                        ):
                            mitigation_data = model_data[
                                model_data["type"] == mitigation_type
                            ]
                            metric_mask = mitigation_data["metric_type"] == metric_type

                            if not any(metric_mask):
                                continue

                            # Calculate x position for this specific point
                            mitigation_offset = (
                                k - (n_mitigations - 1) / 2
                            ) * mitigation_spacing
                            x_pos = metric_base_x + mitigation_offset

                            value = mitigation_data[metric_mask]["value"].iloc[0]
                            std_err = mitigation_data[metric_mask]["std_err"].iloc[0]
                            significance = mitigation_data[metric_mask][
                                "significance"
                            ].iloc[0]

                            # Plot error bar and point
                            # Make standard markers hollow (no infill)
                            markerfacecolor = (
                                "none" if mitigation_type == "standard" else color
                            )
                            plt.errorbar(
                                x_pos,
                                value,
                                yerr=std_err,
                                color=color,
                                marker=marker,
                                markerfacecolor=markerfacecolor,
                                capsize=5,
                                capthick=1.5,
                                elinewidth=1.5,
                                markersize=10,
                                zorder=2,
                            )

                            # Add significance stars if any
                            if significance:
                                # Calculate vertical offset based on model index
                                # This ensures stars are stacked vertically for each metric
                                model_index = list(model_colors.keys()).index(model)
                                base_vertical_offset = 0.005 * (
                                    y_max - y_min
                                )  # Further reduced base offset

                                # Adjust vertical offset based on model index to stack groups of stars
                                group_vertical_offset = (
                                    base_vertical_offset * model_index * 2
                                )  # Keep groups separated but start closer

                                # Position the stars
                                y_pos = (
                                    value
                                    + std_err
                                    + base_vertical_offset
                                    + group_vertical_offset
                                )

                                # Adjust line spacing for vertically stacked stars
                                line_spacing = (
                                    0.5 if "\n" in significance else 1.0
                                )  # Reduced line spacing

                                plt.text(
                                    x_pos,
                                    y_pos,
                                    significance,
                                    ha="center",
                                    va="bottom",
                                    fontsize=16,
                                    fontweight="bold",
                                    linespacing=line_spacing,
                                )

                # Remove x-tick labels
                plt.xticks([])
                ax.set_xlabel("")

            g.map_dataframe(plot_points)

            # Set titles and labels
            col_names = {"gender": "Sex", "age": "Age", "ethnicity": "Race"}
            g.set_titles(template="{col_name}")

            for ax, title in zip(g.axes.flat, [col_names[col] for col in g.col_names]):
                ax.set_title(title, fontweight="bold", pad=20, fontsize=28)

            # Add interpreter title
            g.fig.suptitle(
                metric_labels[interpreter], fontweight="bold", fontsize=34, y=1.1
            )

            # Add y-axis label to the leftmost plot with metric info
            # Determine which metric to show based on interpreter and combo type
            if interpreter == "dice":  # Segmentation
                if combo_name == "eodd":
                    metric_label = "SER"
                else:  # eop
                    metric_label = "Δ Dice"
            else:  # Classification
                if combo_name == "eodd":
                    metric_label = "EODD"
                else:  # eop
                    metric_label = "EOP"

            g.axes[0, 0].set_ylabel(f"Bias Change [{metric_label}]", fontweight="bold")

            # Remove individual legends
            for ax in g.axes.flat:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

            # Save the plots in the appropriate subfolder
            for fmt in ["eps", "pdf", "png"]:
                save_dir = os.path.join(
                    results_dir, "combined_fairness", combo_name, fmt
                )
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(
                    f"{save_dir}/{name}_fairness_{interpreter}.{fmt}",
                    bbox_inches="tight",
                    dpi=300,
                    format=fmt,
                )
            plt.close()

        # Create separate legend figure for each combination
        plt.figure(figsize=(14, 2.5))
        ax = plt.gca()
        ax.set_axis_off()

        # Create legend elements
        legend_elements = []

        # Add model elements
        for model, color in model_colors.items():
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=color,
                    marker="o",
                    linestyle="None",
                    markersize=10,
                    label=model_labels[model],
                )
            )

        # Add a small space in legend
        legend_elements.append(plt.Line2D([0], [0], color="none", label=""))

        # Add mitigation type elements
        for mitigation_type, marker in mitigation_markers.items():
            label = {
                "standard": "Reconstruction",
                "reweighted": "Reweighted",
                "eodd": "EODD Loss",
                "adv": "Adversarial Loss",
            }[mitigation_type]
            # Make standard marker hollow in legend too
            markerfacecolor = "none" if mitigation_type == "standard" else "black"
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="black",
                    marker=marker,
                    linestyle="None",
                    markerfacecolor=markerfacecolor,
                    markersize=10,
                    label=label,
                )
            )

        # Add a small space in legend
        legend_elements.append(plt.Line2D([0], [0], color="none", label=""))

        # Add significance level elements
        legend_elements.append(
            plt.Line2D([0], [0], color="none", label="Significance:")
        )
        legend_elements.append(plt.Line2D([0], [0], color="none", label="* p < 0.05"))

        # Create legend
        plt.legend(
            handles=legend_elements,
            loc="center",
            ncol=len(legend_elements),
            bbox_to_anchor=(0.5, 0.5),
        )

        # Save legend in the appropriate subfolder
        for fmt in ["eps", "pdf", "png"]:
            save_dir = os.path.join(results_dir, "combined_fairness", combo_name, fmt)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                f"{save_dir}/{name}_fairness_legend.{fmt}",
                bbox_inches="tight",
                dpi=300,
                format=fmt,
            )
        plt.close()


# Add new function after fairness_performance_scatter function
def _performance_scatter(
    df, results_dir, vertical_lines, mitigation_type="reweighting", dataset="chex"
):
    """
    Create a scatter plot showing only performance changes for a single dataset.

    Args:
        df: DataFrame with the following columns:
            - metric: Metric (string)
            - performance_percent: Percentage change in performance
            - dataset: Dataset type (string)
            - model: Model name (string)
        vertical_lines: DataFrame with columns:
            - performance_percent: x-position of vertical line
            - model: Model name for color
        results_dir: Directory to save the plots
        mitigation_type: Type of mitigation (for filename)
        dataset: Dataset name for this plot
    """
    import numpy as np

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 28,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "legend.title_fontsize": 18,
        }
    )

    model_colors = {
        "unet": "#1F4F88",
        "pix2pix": "#D0B05F",
        "sde": "#D5302D",
    }

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
        "average": "Average",
        "tgrade": "Tumor Grade",
        "ttype": "Tumor Type",
        "dice": "Segmentation",
    }

    dataset_titles = {"chex": "CheXpert", "ucsf": "UCSF-PDGM"}

    plt.figure(figsize=(12, 3))  # Reduced height from 4 to 3

    # Filter data for this dataset only
    df_filtered = df[df["dataset"] == dataset].copy()
    vertical_lines_filtered = vertical_lines[
        vertical_lines["dataset"] == dataset
    ].copy()

    # First plot vertical lines (so they appear behind the scatter points)
    for _, line in vertical_lines_filtered.iterrows():
        color = model_colors.get(line["model"], "#777777")

        # Plot vertical line from bottom to top of plot with model color
        plt.axvline(
            x=line["performance_percent"],
            color=color,
            linestyle="-",
            linewidth=2.0,
            alpha=0.7,  # Increased alpha for better visibility
            zorder=1,
        )

    # Calculate statistics to identify outliers
    performance_mean = df_filtered["performance_percent"].mean()
    performance_std = df_filtered["performance_percent"].std()

    # Define outlier thresholds
    performance_threshold = 1.5

    # Collect all outliers to identify the most extreme ones
    outliers = []

    # Plot scatter points with higher zorder to ensure they're on top of lines
    for idx, row in df_filtered.iterrows():
        model = row["model"]
        metric = row["metric"]
        performance_pct = row["performance_percent"]

        color = model_colors.get(model, "#777777")

        # Add slight random y-jitter to spread points vertically
        y_jitter = np.random.uniform(-0.01, 0.01)  # Small random offset

        # Use round markers with 3D-style effects
        plt.scatter(
            performance_pct,
            y_jitter,
            s=200,
            marker="o",
            color=color,
            edgecolors="black",  # Black border for 3D effect
            linewidths=0.8,  # Thinner border thickness
            alpha=0.8,  # Slight transparency
            zorder=2,
        )  # Plot points on top of lines

        # Check if this point is an outlier
        performance_z = (
            abs(performance_pct - performance_mean) / performance_std
            if performance_std > 0
            else 0
        )

        if performance_z > performance_threshold:
            outlier_score = performance_z
            outliers.append(
                (outlier_score, idx, performance_pct, y_jitter, metric, model)
            )

    # Sort outliers and add labels (with highest zorder to be on top)
    outliers.sort(reverse=True)
    top_outliers = outliers[: min(5, len(outliers))]

    for _, idx, performance_pct, y_jitter, metric, model in top_outliers:
        label = metric_labels.get(metric, metric)

        plt.annotate(
            label,
            xy=(performance_pct, y_jitter),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
            zorder=3,
        )

    # Add reference line at x=0
    plt.axvline(x=0, color="gray", linestyle="-", linewidth=1, alpha=0.6, zorder=0)

    # Center x-axis around 0 while showing all data points (both scatter points and vertical lines)
    if not df_filtered.empty or not vertical_lines_filtered.empty:
        all_x_values = []

        # Include scatter point x-values
        if not df_filtered.empty:
            all_x_values.extend(df_filtered["performance_percent"].tolist())

        # Include vertical line x-values (PSNR values)
        if not vertical_lines_filtered.empty:
            all_x_values.extend(vertical_lines_filtered["performance_percent"].tolist())

        if all_x_values:
            x_min = min(all_x_values)
            x_max = max(all_x_values)

            # Find the maximum absolute value to ensure all points are visible
            max_abs_value = max(abs(x_min), abs(x_max))

            # Add padding (10% of the maximum absolute value) and set symmetric limits
            padding = max_abs_value * 0.1
            plt.xlim(-max_abs_value - padding, max_abs_value + padding)

    # Set up the plot
    plt.xlabel("Performance Change (%)", fontweight="bold")
    plt.ylabel("")  # No y-axis label since we're only showing x-axis

    # Set light gray background
    plt.gca().set_facecolor("#f5f5f5")  # Light gray background

    # Set y-axis limits to accommodate the jitter with some padding
    plt.ylim(-0.04, 0.04)  # Slightly wider than jitter range for visual padding

    # Hide y-axis ticks and labels since all points are at y=0
    plt.yticks([])

    # Hide y-axis spine to make it cleaner
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    # Set title with dataset name
    plt.title(dataset_titles.get(dataset, dataset), fontweight="bold", pad=20)

    # Add white grid lines at tick marks
    plt.grid(
        True, linestyle="-", alpha=0.8, zorder=0, axis="x", color="white", linewidth=1
    )

    plt.tight_layout()

    for fmt in ["eps", "pdf", "png"]:
        os.makedirs(os.path.join(results_dir, fmt), exist_ok=True)
        plt.savefig(
            os.path.join(
                results_dir,
                fmt,
                f"performance_change_{mitigation_type}_{dataset}.{fmt}",
            ),
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )

    plt.close()


def _create_performance_legend(results_dir):
    """
    Create a shared legend figure for the performance change plots.

    Args:
        results_dir: Directory to save the legend
    """
    # Model name mappings
    model_map = {"unet": "U-Net", "pix2pix": "GAN", "sde": "SDE"}

    # Define colors for models
    model_colors = {
        "unet": "#1F4F88",
        "pix2pix": "#D0B05F",
        "sde": "#D5302D",
    }

    # Create figure for legend
    plt.figure(figsize=(12, 3))
    ax = plt.gca()
    ax.set_axis_off()

    legend_elements = []

    # Add model elements (colors)
    for model in sorted(model_colors.keys()):
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=model_colors[model],
                markersize=15,
                label=model_map.get(model, model),
            )
        )

    # Add PSNR vertical line explanation
    legend_elements.append(
        plt.Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="PSNR")
    )

    # Add AUROC marker explanation
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            linestyle="None",
            markersize=15,
            label="AUROC",
        )
    )

    # Create single legend
    plt.legend(
        handles=legend_elements, ncol=len(legend_elements), loc="center", fontsize=18
    )

    # Save the legend
    for fmt in ["eps", "pdf", "png"]:
        os.makedirs(os.path.join(results_dir, fmt), exist_ok=True)
        plt.savefig(
            os.path.join(results_dir, fmt, f"performance_change_legend.{fmt}"),
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )

    plt.close()


def _get_performance_change(df, model, metric, dataset, mitigation):
    mitigation_df = df[
        (df["model"] == model)
        & (df["dataset"] == dataset)
        & (df["metric"] == metric)
        & (df["mitigation"] == mitigation)
    ]

    standard_df = df[
        (df["model"] == model)
        & (df["dataset"] == dataset)
        & (df["metric"] == metric)
        & (df["mitigation"] == "standard")
    ]

    if len(mitigation_df) == 0 or len(standard_df) == 0:
        # print(f"No data found for model: {model}, dataset: {dataset}, metric: {metric}, mitigation: {mitigation}")
        return None

    mitigation_value = mitigation_df["value"].iloc[0]
    standard_value = standard_df["value"].iloc[0]

    # return the percentage change
    return (mitigation_value - standard_value) / standard_value * 100


def plot_performance_change(performance_df, results_dir):
    """
    Plot only performance changes split by dataset.

    Args:
        performance_df: DataFrame with performance data
        results_dir: Directory to save the plots
    """
    performance_df_og = performance_df.copy()
    results_dir = os.path.join(results_dir, "plot_performance_change")
    os.makedirs(results_dir, exist_ok=True)

    # Print performance statistics
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)

    interpreters = [
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
        "ttype",
        "tgrade",
        "dice",
    ]

    for mitigation in performance_df["mitigation"].unique():
        if mitigation == "standard":
            continue

        print(f"\n{mitigation.upper()}:")

        for dataset in ["chex", "ucsf"]:
            print(f"  {dataset.upper()}:")

            for model in ["unet", "pix2pix", "sde"]:
                # Get PSNR change
                psnr_change = _get_performance_change(
                    performance_df, model, "psnr", dataset, mitigation
                )

                # Get average downstream performance change
                model_downstream_changes = []
                for interpreter in interpreters:
                    change = _get_performance_change(
                        performance_df, model, interpreter, dataset, mitigation
                    )
                    if change is not None:
                        model_downstream_changes.append(change)

                if psnr_change is not None and model_downstream_changes:
                    avg_downstream = sum(model_downstream_changes) / len(
                        model_downstream_changes
                    )
                    print(
                        f"    {model}: PSNR {psnr_change:+5.2f}%, Downstream {avg_downstream:+5.2f}%"
                    )

    print("\n" + "=" * 60)

    # Create a shared legend first
    _create_performance_legend(results_dir)

    interpreters = [
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
        "ttype",
        "tgrade",
        "dice",
    ]

    for mitigation in performance_df["mitigation"].unique():
        if mitigation == "standard":
            continue

        performance_df = performance_df_og.copy()

        for dataset in ["chex", "ucsf"]:
            # Create a list to collect rows
            rows_list = []
            vertical_lines = []

            for model in performance_df["model"].unique():
                if model == "baseline":
                    continue

                # Get PSNR performance change for vertical line
                psnr_performance_change = _get_performance_change(
                    performance_df, model, "psnr", dataset, mitigation
                )
                if psnr_performance_change is not None:
                    vertical_lines.append(
                        {
                            "performance_percent": psnr_performance_change,
                            "model": model,
                            "dataset": dataset,
                        }
                    )

                # Get performance changes for all interpreters
                for interpreter in interpreters:
                    performance_change = _get_performance_change(
                        performance_df, model, interpreter, dataset, mitigation
                    )

                    if performance_change is None:
                        continue

                    # Add to rows list
                    rows_list.append(
                        {
                            "metric": interpreter,
                            "performance_percent": performance_change,
                            "dataset": dataset,
                            "model": model,
                        }
                    )

            # Create the DataFrame from the list of rows
            df = pd.DataFrame(rows_list)
            vertical_lines = pd.DataFrame(vertical_lines)

            # Only create plot if we have data
            if not df.empty:
                _performance_scatter(
                    df, results_dir, vertical_lines, mitigation, dataset
                )


def _get_bias_change(df, model, interpreter, attribute, metric, dataset):

    baseline_df = df[
        (df["model"] == "baseline")
        & (df["dataset"] == dataset)
        & (df["interpreter"] == interpreter)
        & (df["attribute"] == attribute)
        & (df["metric"] == metric)
    ]

    model_df = df[
        (df["model"] == model)
        & (df["dataset"] == dataset)
        & (df["interpreter"] == interpreter)
        & (df["attribute"] == attribute)
        & (df["metric"] == metric)
    ]

    if len(baseline_df) == 0 or len(model_df) == 0:
        # print(f"No data found for model: {model}, dataset: {dataset}, metric: {metric}, mitigation: {mitigation}")
        return None

    baseline_value = baseline_df["value"].iloc[0]
    model_value = model_df["value"].iloc[0]

    # return the percentage change
    return (model_value - baseline_value) / baseline_value * 100


def plot_bias_change_histogram_by_attribute(bias_df, results_dir):

    attributes = ["ethnicity", "gender", "age"]

    attribute_labels = {"ethnicity": "Race", "gender": "Sex", "age": "Age"}

    metrics = ["EODD", "EOP", "delta-dice", "SER"]

    # Define interpreters for all datasets
    interpreters = [
        # CheXpert interpreters
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
        # UCSF interpreters
        "ttype",
        "tgrade",
        # Segmentation interpreter
        "dice",
    ]

    models = ["unet", "pix2pix", "sde"]

    datasets = ["chex", "ucsf"]

    results_dir = os.path.join(results_dir, "plot_bias_change_histogram")
    os.makedirs(results_dir, exist_ok=True)

    bias_changes = {}

    for attribute in attributes:
        bias_changes_attribute = []
        for metric in metrics:
            for dataset in datasets:
                for model in models:
                    for interpreter in interpreters:
                        bias_change = _get_bias_change(
                            bias_df, model, interpreter, attribute, metric, dataset
                        )
                        bias_changes_attribute.append(bias_change)

        bias_changes[attribute] = bias_changes_attribute

    # Remove None values from each attribute's data
    for attribute in attributes:
        bias_changes[attribute] = [x for x in bias_changes[attribute] if x is not None]

    # Set plotting parameters (same as other functions)
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 32,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 18,
            "legend.fontsize": 24,
            "legend.title_fontsize": 24,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
        }
    )

    # Define colors for attributes
    attribute_colors = {
        "ethnicity": "#D5302D",  # Red
        "gender": "#1F4F88",  # Blue
        "age": "#D0B05F",  # Yellow
    }

    # Create the plot
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Set light gray background
    ax.set_facecolor("#f5f5f5")

    # Define fixed bins of size 10 centered around zero
    bins = np.arange(
        -205, 206, 10
    )  # Bins centered on zero: [-5,5], [-15,-5], [5,15], etc.

    # Create overlapping histogram bars for each attribute
    for attribute in attributes:
        print(len(bias_changes[attribute]))
        if len(bias_changes[attribute]) > 0:
            # Add median line for this attribute (behind bars)
            median_value = np.median(bias_changes[attribute])
            plt.axvline(
                x=median_value,
                color=attribute_colors[attribute],
                linestyle="--",
                linewidth=2.5,
                alpha=0.8,
                zorder=1,
            )  # Behind bars

            # Plot as transparent histogram bars with fixed bins
            plt.hist(
                bias_changes[attribute],
                bins=bins,
                density=True,
                color=attribute_colors[attribute],
                alpha=0.7,  # Higher opacity
                label=attribute_labels[attribute],
                edgecolor="white",  # White edges for better separation
                linewidth=0.5,
                zorder=2,
            )  # In front of lines

    # Add white grid lines
    plt.grid(True, color="white", linestyle="-", linewidth=1, alpha=0.8, zorder=0)

    # Add zero line
    plt.axvline(x=0, color="black", linestyle="-", linewidth=1.0, alpha=0.5, zorder=1)

    # Set x-axis limits to cut at 400
    plt.xlim(-200, 200)

    # Set labels and title
    plt.xlabel("Bias Change (%)", fontweight="bold")
    plt.ylabel("Density", fontweight="bold")
    plt.title("Distribution of Bias Changes", fontweight="bold", pad=20)

    # Add legend
    plt.legend(frameon=True, fancybox=True, shadow=True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save the plot
    for fmt in ["eps", "pdf", "png"]:
        save_dir = os.path.join(results_dir, fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/bias_change_histogram.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()


def csv_bias_change_histogram_by_model(bias_df, results_dir):
    attributes = ["ethnicity", "gender", "age"]

    metrics = ["EODD", "EOP", "delta-dice", "SER"]

    # Define interpreters for all datasets
    interpreters = [
        # CheXpert interpreters
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
        # UCSF interpreters
        "ttype",
        "tgrade",
        # Segmentation interpreter
        "dice",
    ]

    models = ["unet", "pix2pix", "sde"]

    datasets = ["chex", "ucsf"]

    results_dir = os.path.join(results_dir, "csv_bias_change_histogram_by_model")
    os.makedirs(results_dir, exist_ok=True)

    bias_changes_by_model = {}

    for model in models:
        bias_changes_attribute = []
        for metric in metrics:
            for dataset in datasets:
                for attribute in attributes:
                    for interpreter in interpreters:
                        bias_change = _get_bias_change(
                            bias_df, model, interpreter, attribute, metric, dataset
                        )
                        bias_changes_attribute.append(bias_change)

        bias_changes_by_model[model] = bias_changes_attribute

    # Remove None values from each attribute's data
    for model in models:
        bias_changes_by_model[model] = [
            x for x in bias_changes_by_model[model] if x is not None
        ]

    csv_results = pd.DataFrame()
    for model in models:
        median_bias_change = np.median(bias_changes_by_model[model])
        absolute_median_bias_change = np.median(np.abs(bias_changes_by_model[model]))
        csv_results[model] = [median_bias_change, absolute_median_bias_change]

    csv_results.index = ["median_bias_change", "absolute_median_bias_change"]
    csv_results.to_csv(
        os.path.join(results_dir, "bias_change_by_model.csv"), index=True
    )


def csv_bias_change_histogram_by_attribute(bias_df, results_dir, mitigation="standard"):
    attributes = ["ethnicity", "gender", "age"]

    attribute_labels = {"ethnicity": "Race", "gender": "Sex", "age": "Age"}

    metrics = ["EODD", "EOP", "delta-dice", "SER"]

    # Define interpreters for all datasets
    interpreters = [
        # CheXpert interpreters
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
        # UCSF interpreters
        "ttype",
        "tgrade",
        # Segmentation interpreter
        "dice",
    ]

    models = ["unet", "pix2pix", "sde"]

    datasets = ["chex", "ucsf"]

    results_dir = os.path.join(results_dir, "csv_bias_change_histogram_by_attribute")
    os.makedirs(results_dir, exist_ok=True)

    bias_changes_by_attribute = {}

    for attribute in attributes:
        bias_changes_attribute = []
        for metric in metrics:
            for dataset in datasets:
                for model in models:
                    for interpreter in interpreters:
                        bias_change = _get_bias_change(
                            bias_df, model, interpreter, attribute, metric, dataset
                        )
                        bias_changes_attribute.append(bias_change)

        bias_changes_by_attribute[attribute] = bias_changes_attribute

    # Remove None values from each attribute's data
    for attribute in attributes:
        bias_changes_by_attribute[attribute] = [
            x for x in bias_changes_by_attribute[attribute] if x is not None
        ]

    csv_results = pd.DataFrame()
    for attribute in attributes:
        median_bias_change = np.median(bias_changes_by_attribute[attribute])
        absolute_median_bias_change = np.median(
            np.abs(bias_changes_by_attribute[attribute])
        )
        csv_results[attribute] = [median_bias_change, absolute_median_bias_change]

    csv_results.index = ["median_bias_change", "absolute_median_bias_change"]
    csv_results.to_csv(
        os.path.join(results_dir, f"bias_change_by_attribute_{mitigation}.csv"),
        index=True,
    )
