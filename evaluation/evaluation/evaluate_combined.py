import pandas as pd
from plot_combined import (
    plot_fairness_change,
    plot_performance_change,
    plot_fairness_change_separate,
    plot_bias_change_histogram_by_attribute,
    csv_bias_change_histogram_by_model,
    csv_bias_change_histogram_by_attribute,
)


def evaluate_combined(config, results_dir, name):
    """
    This function evaluates the mitigation and plots the results.
    For the GR, we have the following tables and plots:
    - Tables:
        - Summary from IDP: Additional bias introduced by the reconstruction in the standard model (for each sensitive attribute)
        - Performance: Change in performance of the standard model vs. the mitigated models (for each dataset x each model)
        - Fairness: Change in fairness of the mitigated models vs. the standard model (for each mitigation x each sensitive attribute)
    - Plots:
        - Fairness change: Increase in fairness of the mitigated models vs. the standard model (average and each pathology)
        - Fairness performance tradeoff: Change in performance vs. fairness for each mitigation (for each mitigation x each sensitive attribute x fairness metrics)
    """

    performance_config = config["performance"]
    fairness_config = config["fairness"]

    # --- Performance Dataframes ---
    ucsf_performance_standard_df = pd.read_csv(
        performance_config["ucsf"]["csv_standard"]
    )
    ucsf_performance_standard_df = ucsf_performance_standard_df[
        ucsf_performance_standard_df["acceleration"] == 8
    ]
    ucsf_performance_eodd_df = pd.read_csv(performance_config["ucsf"]["csv_eodd"])
    ucsf_performance_reweighted_df = pd.read_csv(
        performance_config["ucsf"]["csv_reweighted"]
    )
    ucsf_performance_adv_df = pd.read_csv(performance_config["ucsf"]["csv_adv"])

    # Add columns for dataset and mitigation type
    ucsf_performance_standard_df["dataset"] = "ucsf"
    ucsf_performance_standard_df["mitigation"] = "standard"
    ucsf_performance_eodd_df["dataset"] = "ucsf"
    ucsf_performance_eodd_df["mitigation"] = "eodd"
    ucsf_performance_reweighted_df["dataset"] = "ucsf"
    ucsf_performance_reweighted_df["mitigation"] = "reweighted"
    ucsf_performance_adv_df["dataset"] = "ucsf"
    ucsf_performance_adv_df["mitigation"] = "adv"

    chex_performance_standard_df = pd.read_csv(
        performance_config["chex"]["csv_standard"]
    )
    chex_performance_standard_df = chex_performance_standard_df[
        chex_performance_standard_df["photon_count"] == 10000
    ]
    chex_performance_eodd_df = pd.read_csv(performance_config["chex"]["csv_eodd"])
    chex_performance_reweighted_df = pd.read_csv(
        performance_config["chex"]["csv_reweighted"]
    )
    chex_performance_adv_df = pd.read_csv(performance_config["chex"]["csv_adv"])

    # Add columns for dataset and mitigation type
    chex_performance_standard_df["dataset"] = "chex"
    chex_performance_standard_df["mitigation"] = "standard"
    chex_performance_eodd_df["dataset"] = "chex"
    chex_performance_eodd_df["mitigation"] = "eodd"
    chex_performance_reweighted_df["dataset"] = "chex"
    chex_performance_reweighted_df["mitigation"] = "reweighted"
    chex_performance_adv_df["dataset"] = "chex"
    chex_performance_adv_df["mitigation"] = "adv"

    # Combine all performance dataframes
    performance_df = pd.concat(
        [
            ucsf_performance_standard_df,
            ucsf_performance_eodd_df,
            ucsf_performance_reweighted_df,
            ucsf_performance_adv_df,
            chex_performance_standard_df,
            chex_performance_eodd_df,
            chex_performance_reweighted_df,
            chex_performance_adv_df,
        ]
    )

    plot_performance_change(performance_df, results_dir)

    # --- Fairness Dataframes ---
    ucsf_fairness_standard_df = pd.read_csv(fairness_config["ucsf"]["csv_standard"])
    ucsf_fairness_eodd_df = pd.read_csv(fairness_config["ucsf"]["csv_eodd"])
    ucsf_fairness_reweighted_df = pd.read_csv(fairness_config["ucsf"]["csv_reweighted"])
    ucsf_fairness_adv_df = pd.read_csv(fairness_config["ucsf"]["csv_adv"])

    # Add columns for dataset and mitigation type
    ucsf_fairness_standard_df["dataset"] = "ucsf"
    ucsf_fairness_standard_df["mitigation"] = "standard"
    ucsf_fairness_eodd_df["dataset"] = "ucsf"
    ucsf_fairness_eodd_df["mitigation"] = "eodd"
    ucsf_fairness_reweighted_df["dataset"] = "ucsf"
    ucsf_fairness_reweighted_df["mitigation"] = "reweighted"
    ucsf_fairness_adv_df["dataset"] = "ucsf"
    ucsf_fairness_adv_df["mitigation"] = "adv"

    chex_fairness_standard_df = pd.read_csv(fairness_config["chex"]["csv_standard"])
    chex_fairness_eodd_df = pd.read_csv(fairness_config["chex"]["csv_eodd"])
    chex_fairness_reweighted_df = pd.read_csv(fairness_config["chex"]["csv_reweighted"])
    chex_fairness_adv_df = pd.read_csv(fairness_config["chex"]["csv_adv"])

    # Add columns for dataset and mitigation type
    chex_fairness_standard_df["dataset"] = "chex"
    chex_fairness_standard_df["mitigation"] = "standard"
    chex_fairness_eodd_df["dataset"] = "chex"
    chex_fairness_eodd_df["mitigation"] = "eodd"
    chex_fairness_reweighted_df["dataset"] = "chex"
    chex_fairness_reweighted_df["mitigation"] = "reweighted"
    chex_fairness_adv_df["dataset"] = "chex"
    chex_fairness_adv_df["mitigation"] = "adv"

    df_standard = pd.concat([ucsf_fairness_standard_df, chex_fairness_standard_df])
    df_eodd = pd.concat([ucsf_fairness_eodd_df, chex_fairness_eodd_df])
    df_reweighted = pd.concat(
        [ucsf_fairness_reweighted_df, chex_fairness_reweighted_df]
    )
    df_adv = pd.concat([ucsf_fairness_adv_df, chex_fairness_adv_df])

    plot_fairness_change(df_standard, df_eodd, df_reweighted, df_adv, results_dir, name)
    plot_fairness_change_separate(
        df_standard, df_eodd, df_reweighted, df_adv, results_dir, name
    )

    # plot_bias_change_histogram_by_attribute(df_standard, results_dir)
    csv_bias_change_histogram_by_model(df_standard, results_dir)

    csv_bias_change_histogram_by_attribute(df_standard, results_dir)
    csv_bias_change_histogram_by_attribute(df_eodd, results_dir, "eodd")
    csv_bias_change_histogram_by_attribute(df_reweighted, results_dir, "reweighted")
    csv_bias_change_histogram_by_attribute(df_adv, results_dir, "adv")
