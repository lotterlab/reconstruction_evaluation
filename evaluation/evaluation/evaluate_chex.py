import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from plot_chex import (
    plot_chex_performance,
)
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


def bootstrap_fairness(
    interpreter_results,
    attribute,
    attribute_values,
    interpreter_name,
    n_iterations=1000,
):
    """
    Bootstraps the difference in equalized odds (EODD) between baseline (class) and reconstruction predictions.

    Parameters:
        interpreter_results (pd.DataFrame): The filtered DataFrame for the current interpreter.
        attribute (str): The column name for the sensitive attribute (e.g., 'sex').
        attribute_values (list): The list of subgroup values (e.g., ['Male', 'Female']).
        interpreter_name (str): The prefix used in your column names for this interpreter.
        n_iterations (int): The number of bootstrap iterations.

    Returns:
        observed_delta (float): The observed difference in EODD (EODD_recon - EODD_class) from the full sample.
        p_value (float): The bootstrap p-value for the difference.
        boot_deltas (np.array): The array of bootstrapped differences.
    """

    # Calculate the observed metrics:
    tpr_class = []
    fpr_class = []
    tpr_recon = []
    fpr_recon = []

    for val in attribute_values:
        subgroup = interpreter_results[interpreter_results[attribute] == val]
        if len(subgroup) == 0:
            continue
        tpr_class.append(
            get_tpr(
                subgroup[f"{interpreter_name}_class"], subgroup[f"{interpreter_name}"]
            )
        )
        fpr_class.append(
            get_fpr(
                subgroup[f"{interpreter_name}_class"], subgroup[f"{interpreter_name}"]
            )
        )
        tpr_recon.append(
            get_tpr(
                subgroup[f"{interpreter_name}_recon"], subgroup[f"{interpreter_name}"]
            )
        )
        fpr_recon.append(
            get_fpr(
                subgroup[f"{interpreter_name}_recon"], subgroup[f"{interpreter_name}"]
            )
        )

    # Compute EODD for baseline and recon:
    observed_eodd_class = (
        (max(tpr_class) - min(tpr_class)) + (max(fpr_class) - min(fpr_class))
    ) / 2
    observed_eodd_recon = (
        (max(tpr_recon) - min(tpr_recon)) + (max(fpr_recon) - min(fpr_recon))
    ) / 2

    observed_eop_class = max(tpr_class) - min(tpr_class)
    observed_eop_recon = max(tpr_recon) - min(tpr_recon)

    observed_delta_eodd = observed_eodd_recon - observed_eodd_class
    observed_delta_eop = observed_eop_recon - observed_eop_class

    # Prepare to store bootstrap differences:
    boot_deltas_eodd = []
    boot_deltas_eop = []
    boot_baseline_eodd = []
    boot_baseline_eop = []
    boot_recon_eodd = []
    boot_recon_eop = []
    # Bootstrap iterations:
    for i in range(n_iterations):
        boot_tpr_class = []
        boot_fpr_class = []
        boot_tpr_recon = []
        boot_fpr_recon = []

        boot_interpreter_results = interpreter_results.sample(
            n=len(interpreter_results), replace=True
        )

        for val in attribute_values:
            subgroup = boot_interpreter_results[
                boot_interpreter_results[attribute] == val
            ]
            # Sample with replacement within this subgroup
            boot_tpr_class.append(
                get_tpr(
                    subgroup[f"{interpreter_name}_class"],
                    subgroup[f"{interpreter_name}"],
                )
            )
            boot_fpr_class.append(
                get_fpr(
                    subgroup[f"{interpreter_name}_class"],
                    subgroup[f"{interpreter_name}"],
                )
            )
            boot_tpr_recon.append(
                get_tpr(
                    subgroup[f"{interpreter_name}_recon"],
                    subgroup[f"{interpreter_name}"],
                )
            )
            boot_fpr_recon.append(
                get_fpr(
                    subgroup[f"{interpreter_name}_recon"],
                    subgroup[f"{interpreter_name}"],
                )
            )

        # Compute EODD for this bootstrap sample:
        boot_eodd_class = (
            (max(boot_tpr_class) - min(boot_tpr_class))
            + (max(boot_fpr_class) - min(boot_fpr_class))
        ) / 2
        boot_eodd_recon = (
            (max(boot_tpr_recon) - min(boot_tpr_recon))
            + (max(boot_fpr_recon) - min(boot_fpr_recon))
        ) / 2
        boot_delta_eodd = boot_eodd_recon - boot_eodd_class
        boot_deltas_eodd.append(boot_delta_eodd)
        boot_baseline_eodd.append(boot_eodd_class)
        boot_recon_eodd.append(boot_eodd_recon)

        boot_eop_class = max(boot_tpr_class) - min(boot_tpr_class)
        boot_eop_recon = max(boot_tpr_recon) - min(boot_tpr_recon)
        boot_delta_eop = boot_eop_recon - boot_eop_class
        boot_deltas_eop.append(boot_delta_eop)
        boot_baseline_eop.append(boot_eop_class)
        boot_recon_eop.append(boot_eop_recon)

    boot_deltas_eodd = np.array(boot_deltas_eodd)
    boot_deltas_eop = np.array(boot_deltas_eop)
    boot_baseline_eodd = np.array(boot_baseline_eodd)
    boot_baseline_eop = np.array(boot_baseline_eop)
    boot_recon_eodd = np.array(boot_recon_eodd)
    boot_recon_eop = np.array(boot_recon_eop)
    # Compute the one-tailed p-value:
    if observed_delta_eodd >= 0:
        p_value_eodd = np.mean(boot_deltas_eodd <= 0)
    else:
        p_value_eodd = np.mean(boot_deltas_eodd >= 0)

    if observed_delta_eop >= 0:
        p_value_eop = np.mean(boot_deltas_eop <= 0)
    else:
        p_value_eop = np.mean(boot_deltas_eop >= 0)

    std_eodd_delta = np.std(boot_deltas_eodd)
    std_eop_delta = np.std(boot_deltas_eop)

    results = {
        "eodd_class": observed_eodd_class,
        "eodd_recon": observed_eodd_recon,
        "eop_class": observed_eop_class,
        "eop_recon": observed_eop_recon,
        "delta_eodd": observed_delta_eodd,
        "delta_eodd_bootstrapped": boot_deltas_eodd.mean(),
        "delta_eop": observed_delta_eop,
        "delta_eop_bootstrapped": boot_deltas_eop.mean(),
        "delta_eodd_p_value": p_value_eodd,
        "delta_eop_p_value": p_value_eop,
        "delta_eodd_std": std_eodd_delta,
        "delta_eop_std": std_eop_delta,
        "eodd_class_bootstrapped": boot_baseline_eodd.mean(),
        "eop_class_bootstrapped": boot_baseline_eop.mean(),
        "eodd_recon_bootstrapped": boot_recon_eodd.mean(),
        "eop_recon_bootstrapped": boot_recon_eop.mean(),
        "eodd_class_std_bootstrapped": boot_baseline_eodd.std(),
        "eop_class_std_bootstrapped": boot_baseline_eop.std(),
        "eodd_recon_std_bootstrapped": boot_recon_eodd.std(),
        "eop_recon_std_bootstrapped": boot_recon_eop.std(),
    }

    return results


def get_tpr(y_pred, y):
    # calculate true positive rate
    y_pred = y_pred.astype(int)
    y = y.astype(int)
    if y.sum() == 0:
        return 1
    tpr = y_pred[y == 1].sum() / y.sum()
    return tpr


def get_fpr(y_pred, y):
    # calculate false positive rate
    y_pred = y_pred.astype(int)
    y = y.astype(int)
    n_neg = len(y) - y.sum()
    if n_neg == 0:
        return 0.0
    fpr = y_pred[y == 0].sum() / n_neg
    return fpr


def calculate_threshold(y_pred, y_true):
    # Convert to numpy arrays to avoid pandas indexing issues
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # If all ground truth values are 0 or 1, use a reasonable default threshold
    if len(np.unique(y_true)) == 1:
        print(
            f"Warning: All ground truth values are {y_true[0]}. Using default threshold of 0.5"
        )
        return 0.5

    # calculate threshold for the metric
    fpr, sens, threshs = roc_curve(y_true, y_pred)
    spec = 1 - fpr
    return threshs[np.argmin(np.abs(spec - sens))]


def fairness_prediction(predictions, fairness_path):
    if fairness_path is not None:
        df = pd.read_csv(fairness_path)
        return df
    predictions = predictions.copy()  # Make a shallow copy of the list

    sensitive_attributes = {
        "sex": (["Male", "Female"], "gender"),
        "age_bin": (["O", "Y"], "age"),
        "race": (
            [
                "White",
                "Asian",
                "Other",
                "American Indian or Alaska Native",
                "Native Hawaiian or Other Pacific Islander",
                "Black",
            ],
            "ethnicity",
        ),
    }

    interpreters = {
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

    all_results = []
    for prediction in predictions:
        if prediction["fairness"]:
            original_model = prediction["model"]
            # Make a fresh copy for each interpreter to avoid data loss
            prediction_results = prediction["prediction_results"].copy()
            prediction_results["age_bin"] = prediction_results["age"].apply(
                lambda x: (
                    "Y" if float(str(x).split("(")[1].split(",")[0]) <= 62 else "O"
                )
            )

            average_EODD_class = {}
            average_EOP_class = {}
            average_EODD_recon = {}
            average_EOP_recon = {}
            average_delta_EODD_recon = {}
            average_delta_EOP_recon = {}
            average_delta_EOP_bootstrapped = {}
            average_delta_EODD_bootstrapped = {}
            average_p_value_eop = {}
            average_p_value_eodd = {}
            average_std_eodd_delta = {}
            average_std_eop_delta = {}
            average_eodd_class_bootstrapped = {}
            average_eop_class_bootstrapped = {}
            average_eodd_recon_bootstrapped = {}
            average_eop_recon_bootstrapped = {}
            average_eodd_class_std_bootstrapped = {}
            average_eop_class_std_bootstrapped = {}
            average_eodd_recon_std_bootstrapped = {}
            average_eop_recon_std_bootstrapped = {}

            for interpreter, interpreter_name in interpreters.items():
                print(f"Evaluating {interpreter_name}")
                # Create a filtered copy for this specific interpreter
                interpreter_results = prediction_results.copy()

                # Filter nan values for just this interpreter's columns
                mask = (
                    interpreter_results[f"{interpreter_name}"].notna()
                    & interpreter_results[f"{interpreter_name}_class"].notna()
                    & interpreter_results[f"{interpreter_name}_recon"].notna()
                )
                interpreter_results = interpreter_results[mask]

                baseline_threshold = calculate_threshold(
                    interpreter_results[f"{interpreter_name}_class"],
                    interpreter_results[f"{interpreter_name}"],
                )
                recon_threshold = calculate_threshold(
                    interpreter_results[f"{interpreter_name}_recon"],
                    interpreter_results[f"{interpreter_name}"],
                )

                # Store thresholded predictions back in the original prediction_results
                interpreter_results.loc[mask, f"{interpreter_name}_class"] = (
                    interpreter_results[f"{interpreter_name}_class"].apply(
                        lambda x: 1 if x >= baseline_threshold else 0
                    )
                )
                interpreter_results.loc[mask, f"{interpreter_name}_recon"] = (
                    interpreter_results[f"{interpreter_name}_recon"].apply(
                        lambda x: 1 if x >= recon_threshold else 0
                    )
                )

                for attribute, (
                    attribute_values,
                    attribute_name,
                ) in sensitive_attributes.items():
                    if not attribute_name in average_EODD_recon:
                        average_EODD_class[attribute_name] = []
                        average_EOP_class[attribute_name] = []
                        average_EODD_recon[attribute_name] = []
                        average_EOP_recon[attribute_name] = []
                        average_delta_EODD_recon[attribute_name] = []
                        average_delta_EOP_recon[attribute_name] = []
                        average_p_value_eop[attribute_name] = []
                        average_p_value_eodd[attribute_name] = []
                        average_std_eodd_delta[attribute_name] = []
                        average_std_eop_delta[attribute_name] = []
                        average_delta_EOP_bootstrapped[attribute_name] = []
                        average_delta_EODD_bootstrapped[attribute_name] = []
                        average_eodd_class_bootstrapped[attribute_name] = []
                        average_eop_class_bootstrapped[attribute_name] = []
                        average_eodd_recon_bootstrapped[attribute_name] = []
                        average_eop_recon_bootstrapped[attribute_name] = []
                        average_eodd_class_std_bootstrapped[attribute_name] = []
                        average_eop_class_std_bootstrapped[attribute_name] = []
                        average_eodd_recon_std_bootstrapped[attribute_name] = []
                        average_eop_recon_std_bootstrapped[attribute_name] = []

                    results = bootstrap_fairness(
                        interpreter_results,
                        attribute,
                        attribute_values,
                        interpreter_name,
                    )

                    average_EODD_class[attribute_name].append(results["eodd_class"])
                    average_EOP_class[attribute_name].append(results["eop_class"])
                    average_EODD_recon[attribute_name].append(results["eodd_recon"])
                    average_EOP_recon[attribute_name].append(results["eop_recon"])
                    average_delta_EODD_recon[attribute_name].append(
                        results["delta_eodd"]
                    )
                    average_delta_EOP_recon[attribute_name].append(results["delta_eop"])
                    average_p_value_eop[attribute_name].append(
                        results["delta_eop_p_value"]
                    )
                    average_p_value_eodd[attribute_name].append(
                        results["delta_eodd_p_value"]
                    )
                    average_std_eodd_delta[attribute_name].append(
                        results["delta_eodd_std"]
                    )
                    average_std_eop_delta[attribute_name].append(
                        results["delta_eop_std"]
                    )
                    average_delta_EOP_bootstrapped[attribute_name].append(
                        results["delta_eop_bootstrapped"]
                    )
                    average_delta_EODD_bootstrapped[attribute_name].append(
                        results["delta_eodd_bootstrapped"]
                    )
                    average_eodd_class_bootstrapped[attribute_name].append(
                        results["eodd_class_bootstrapped"]
                    )
                    average_eop_class_bootstrapped[attribute_name].append(
                        results["eop_class_bootstrapped"]
                    )
                    average_eodd_recon_bootstrapped[attribute_name].append(
                        results["eodd_recon_bootstrapped"]
                    )
                    average_eop_recon_bootstrapped[attribute_name].append(
                        results["eop_recon_bootstrapped"]
                    )
                    average_eodd_class_std_bootstrapped[attribute_name].append(
                        results["eodd_class_std_bootstrapped"]
                    )
                    average_eop_class_std_bootstrapped[attribute_name].append(
                        results["eop_class_std_bootstrapped"]
                    )
                    average_eodd_recon_std_bootstrapped[attribute_name].append(
                        results["eodd_recon_std_bootstrapped"]
                    )
                    average_eop_recon_std_bootstrapped[attribute_name].append(
                        results["eop_recon_std_bootstrapped"]
                    )

                    all_results.append(
                        {
                            "model": "baseline",
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "EODD",
                            "value": results["eodd_class"],
                        }
                    )
                    all_results.append(
                        {
                            "model": "baseline",
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "EOP",
                            "value": results["eop_class"],
                        }
                    )

                    all_results.append(
                        {
                            "model": original_model,
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "EODD",
                            "value": results["eodd_recon"],
                        }
                    )
                    all_results.append(
                        {
                            "model": original_model,
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "EOP",
                            "value": results["eop_recon"],
                        }
                    )
                    all_results.append(
                        {
                            "model": original_model,
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "delta-EODD",
                            "value": results["delta_eodd"],
                        }
                    )
                    all_results.append(
                        {
                            "model": original_model,
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "delta-EOP",
                            "value": results["delta_eop"],
                        }
                    )
                    all_results.append(
                        {
                            "model": original_model,
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "delta-EOP-p-value",
                            "value": results["delta_eop_p_value"],
                        }
                    )
                    all_results.append(
                        {
                            "model": original_model,
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "delta-EODD-p-value",
                            "value": results["delta_eodd_p_value"],
                        }
                    )
                    all_results.append(
                        {
                            "model": original_model,
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "delta-EODD-std-err",
                            "value": results["delta_eodd_std"],
                        }
                    )
                    all_results.append(
                        {
                            "model": original_model,
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "delta-EOP-std-err",
                            "value": results["delta_eop_std"],
                        }
                    )
                    all_results.append(
                        {
                            "model": original_model,
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "delta-EODD-bootstrapped",
                            "value": results["delta_eodd_bootstrapped"],
                        }
                    )
                    all_results.append(
                        {
                            "model": original_model,
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "delta-EOP-bootstrapped",
                            "value": results["delta_eop_bootstrapped"],
                        }
                    )
                    all_results.append(
                        {
                            "model": original_model,
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "EOP-bootstrapped",
                            "value": results["eop_recon_bootstrapped"],
                        }
                    )
                    all_results.append(
                        {
                            "model": original_model,
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "EODD-bootstrapped",
                            "value": results["eodd_recon_bootstrapped"],
                        }
                    )
                    all_results.append(
                        {
                            "model": original_model,
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "EOP-std-err-bootstrapped",
                            "value": results["eop_recon_std_bootstrapped"],
                        }
                    )
                    all_results.append(
                        {
                            "model": original_model,
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "EODD-std-err-bootstrapped",
                            "value": results["eodd_recon_std_bootstrapped"],
                        }
                    )
                    all_results.append(
                        {
                            "model": "baseline",
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "EOP-bootstrapped",
                            "value": results["eop_class_bootstrapped"],
                        }
                    )
                    all_results.append(
                        {
                            "model": "baseline",
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "EODD-bootstrapped",
                            "value": results["eodd_class_bootstrapped"],
                        }
                    )
                    all_results.append(
                        {
                            "model": "baseline",
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "EOP-std-err-bootstrapped",
                            "value": results["eop_class_std_bootstrapped"],
                        }
                    )
                    all_results.append(
                        {
                            "model": "baseline",
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "EODD-std-err-bootstrapped",
                            "value": results["eodd_class_std_bootstrapped"],
                        }
                    )

            for _, (_, attribute_name) in sensitive_attributes.items():
                all_results.append(
                    {
                        "model": "baseline",
                        "interpreter": "average",
                        "attribute": attribute_name,
                        "metric": "EODD",
                        "value": np.mean(average_EODD_class[attribute_name]),
                    }
                )
                all_results.append(
                    {
                        "model": "baseline",
                        "interpreter": "average",
                        "attribute": attribute_name,
                        "metric": "EOP",
                        "value": np.mean(average_EOP_class[attribute_name]),
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "average",
                        "attribute": attribute_name,
                        "metric": "EODD",
                        "value": np.mean(average_EODD_recon[attribute_name]),
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "average",
                        "attribute": attribute_name,
                        "metric": "EOP",
                        "value": np.mean(average_EOP_recon[attribute_name]),
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "average",
                        "attribute": attribute_name,
                        "metric": "delta-EODD",
                        "value": np.mean(average_delta_EODD_recon[attribute_name]),
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "average",
                        "attribute": attribute_name,
                        "metric": "delta-EOP",
                        "value": np.mean(average_delta_EOP_recon[attribute_name]),
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "average",
                        "attribute": attribute_name,
                        "metric": "delta-EOP-p-value",
                        "value": np.mean(average_p_value_eop[attribute_name]),
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "average",
                        "attribute": attribute_name,
                        "metric": "delta-EODD-p-value",
                        "value": np.mean(average_p_value_eodd[attribute_name]),
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "average",
                        "attribute": attribute_name,
                        "metric": "delta-EODD-std-err",
                        "value": np.mean(average_std_eodd_delta[attribute_name]),
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "average",
                        "attribute": attribute_name,
                        "metric": "delta-EOP-std-err",
                        "value": np.mean(average_std_eop_delta[attribute_name]),
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "average",
                        "attribute": attribute_name,
                        "metric": "delta-EOP-bootstrapped",
                        "value": np.mean(
                            average_delta_EOP_bootstrapped[attribute_name]
                        ),
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "average",
                        "attribute": attribute_name,
                        "metric": "delta-EODD-bootstrapped",
                        "value": np.mean(
                            average_delta_EODD_bootstrapped[attribute_name]
                        ),
                    }
                )

                all_results.append(
                    {
                        "model": "baseline",
                        "interpreter": "average",
                        "attribute": attribute_name,
                        "metric": "EOP-bootstrapped",
                        "value": np.mean(
                            average_eop_class_bootstrapped[attribute_name]
                        ),
                    }
                )
                all_results.append(
                    {
                        "model": "baseline",
                        "interpreter": "average",
                        "attribute": attribute_name,
                        "metric": "EODD-bootstrapped",
                        "value": np.mean(
                            average_eodd_class_bootstrapped[attribute_name]
                        ),
                    }
                )
                all_results.append(
                    {
                        "model": "baseline",
                        "interpreter": "average",
                        "attribute": attribute_name,
                        "metric": "EOP-std-err-bootstrapped",
                        "value": np.mean(
                            average_eop_class_std_bootstrapped[attribute_name]
                        ),
                    }
                )
                all_results.append(
                    {
                        "model": "baseline",
                        "interpreter": "average",
                        "attribute": attribute_name,
                        "metric": "EODD-std-err-bootstrapped",
                        "value": np.mean(
                            average_eodd_class_std_bootstrapped[attribute_name]
                        ),
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "average",
                        "attribute": attribute_name,
                        "metric": "EOP-bootstrapped",
                        "value": np.mean(
                            average_eop_recon_bootstrapped[attribute_name]
                        ),
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "average",
                        "attribute": attribute_name,
                        "metric": "EODD-bootstrapped",
                        "value": np.mean(
                            average_eodd_recon_bootstrapped[attribute_name]
                        ),
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "average",
                        "attribute": attribute_name,
                        "metric": "EOP-std-err-bootstrapped",
                        "value": np.mean(
                            average_eop_recon_std_bootstrapped[attribute_name]
                        ),
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "average",
                        "attribute": attribute_name,
                        "metric": "EODD-std-err-bootstrapped",
                        "value": np.mean(
                            average_eodd_recon_std_bootstrapped[attribute_name]
                        ),
                    }
                )
    evaluation_results = pd.DataFrame(all_results)
    return evaluation_results


def bootstrap_psnr_difference(
    interpreter_results,
    attribute,
    attribute_values,
    interpreter_name,
    n_iterations=1000,
):
    """
    Perform a bootstrap test to compare a statistic (e.g., the mean) between two groups,
    even when the groups have different sample sizes.

    Parameters:
        group1, group2: Arrays or lists containing the data for each group.
        statistic: Function to compute the statistic of interest (default is np.mean).
        n_iterations: Number of bootstrap iterations (default is 10,000).

    Returns:
        observed_diff: The observed difference in the statistic between the two groups.
        p_value: The two-tailed bootstrap p-value.
    """
    psnr_values = []

    for i, val in enumerate(attribute_values):
        subgroup = interpreter_results[interpreter_results[attribute] == val]
        psnr_values.append((subgroup[f"{interpreter_name}"].mean(), i))

    min_psnr, min_index = min(psnr_values, key=lambda x: x[0])
    max_psnr, max_index = max(psnr_values, key=lambda x: x[0])

    # Calculate the observed difference
    observed_diff = max_psnr - min_psnr

    boot_diffs = []
    # Perform bootstrapping
    for i in range(n_iterations):
        boot_psnr_values = []

        boot_interpreter_results = interpreter_results.sample(
            n=len(interpreter_results), replace=True
        )
        for val in attribute_values:
            subgroup = boot_interpreter_results[
                boot_interpreter_results[attribute] == val
            ]
            boot_psnr_values.append(subgroup[f"{interpreter_name}"].mean())

        boot_diffs.append(boot_psnr_values[max_index] - boot_psnr_values[min_index])

    boot_diffs = np.array(boot_diffs)

    # Calculate the one-tailed p-value:
    # Proportion of bootstrap differences as or more extreme than the observed difference.
    if observed_diff >= 0:
        p_value = np.mean(boot_diffs <= 0)
    else:
        p_value = np.mean(boot_diffs >= 0)

    results = {
        "delta_psnr": observed_diff,
        "delta_psnr_p_value": p_value,
        "delta_psnr_std": np.std(boot_diffs),
        "delta_psnr_percent": (observed_diff / min_psnr) * 100,
    }
    return results


def psnr_difference_prediction(predictions):
    sensitive_attributes = {
        "sex": (["Male", "Female"], "gender"),
        "age_bin": (["O", "Y"], "age"),
        "race": (
            [
                "White",
                "Asian",
                "Other",
                "American Indian or Alaska Native",
                "Native Hawaiian or Other Pacific Islander",
                "Black",
            ],
            "ethnicity",
        ),
    }

    all_results = []
    for prediction in predictions:
        if prediction["fairness"]:
            original_model = prediction["model"]
            prediction_results = prediction["prediction_results"].copy()
            prediction_results["age_bin"] = prediction_results["age"].apply(
                lambda x: (
                    "Y" if float(str(x).split("(")[1].split(",")[0]) <= 62 else "O"
                )
            )

            for attribute, (
                attribute_values,
                attribute_name,
            ) in sensitive_attributes.items():

                results = bootstrap_psnr_difference(
                    prediction_results,
                    attribute,
                    attribute_values,
                    "psnr",
                )

                all_results.append(
                    {
                        "model": original_model,
                        "attribute": attribute_name,
                        "metric": "delta-psnr",
                        "value": results["delta_psnr"],
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "attribute": attribute_name,
                        "metric": "delta-psnr-p-value",
                        "value": results["delta_psnr_p_value"],
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "attribute": attribute_name,
                        "metric": "delta-psnr-std",
                        "value": results["delta_psnr_std"],
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "attribute": attribute_name,
                        "metric": "delta-psnr-percent",
                        "value": results["delta_psnr_percent"],
                    }
                )
    return pd.DataFrame(all_results)


def performance_prediction(predictions):
    # Create a list to store all rows and create DataFrame at the end
    all_results = []
    classifier_metrics = {
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
    image_metrics = {
        "psnr": "psnr",
        "ssim": "ssim",
        "nrmse": "nrmse",
        "lpips": "lpips",
    }

    for prediction in predictions:
        original_model = prediction["model"]
        photon_count = prediction["photon_count"]
        prediction_results = prediction["prediction_results"]

        for metric, metric_name in image_metrics.items():
            value = prediction_results[metric_name].mean()

            # Add result to list
            all_results.append(
                {
                    "photon_count": photon_count,
                    "model": original_model,
                    "metric": metric,
                    "value": value,
                }
            )

            for classifier, classifier_name in classifier_metrics.items():
                metric_prediction_results = prediction_results.copy()
                mask = (
                    prediction_results[classifier_name].notna()
                    & prediction_results[f"{classifier_name}_class"].notna()
                    & prediction_results[f"{classifier_name}_recon"].notna()
                )
                metric_prediction_results = metric_prediction_results[mask]
                value = metric_prediction_results[metric_name].mean()

                # Add result to list
                all_results.append(
                    {
                        "photon_count": photon_count,
                        "model": original_model,
                        "metric": f"{metric}-{classifier}",
                        "value": value,
                    }
                )

        baseline_auroc = []
        recon_auroc = []
        for metric, metric_name in classifier_metrics.items():

            # create mask for nan rows
            mask = (
                prediction_results[metric_name].notna()
                & prediction_results[f"{metric_name}_class"].notna()
                & prediction_results[f"{metric_name}_recon"].notna()
            )

            ground_truth = prediction_results[metric_name][mask]
            class_prediction = prediction_results[f"{metric_name}_class"][mask]
            recon_prediction = prediction_results[f"{metric_name}_recon"][mask]

            baseline_value = roc_auc_score(ground_truth, class_prediction)
            baseline_auroc.append(baseline_value)
            all_results.append(
                {
                    "photon_count": photon_count,
                    "model": "baseline",
                    "metric": metric,
                    "value": baseline_value,
                }
            )

            recon_value = roc_auc_score(ground_truth, recon_prediction)
            recon_auroc.append(recon_value)
            all_results.append(
                {
                    "photon_count": photon_count,
                    "model": original_model,
                    "metric": metric,
                    "value": recon_value,
                }
            )

        baseline_auroc = np.mean(baseline_auroc)
        all_results.append(
            {
                "photon_count": photon_count,
                "model": "baseline",
                "metric": "average",
                "value": baseline_auroc,
            }
        )
        recon_auroc = np.mean(recon_auroc)
        all_results.append(
            {
                "photon_count": photon_count,
                "model": original_model,
                "metric": "average",
                "value": recon_auroc,
            }
        )

    # Create final DataFrame from all results
    evaluation_results = pd.DataFrame(all_results)
    return evaluation_results


def evaluate_chex(config, results_dir, name):
    predictions = []

    for prediction in config["predictions"]:
        model = prediction["model"]
        photon_count = prediction["photon_count"]
        prediction_results = pd.read_csv(prediction["predictions"])
        fairness = prediction["fairness"]

        predictions.append(
            {
                "model": model,
                "photon_count": photon_count,
                "prediction_results": prediction_results,
                "fairness": fairness,
            }
        )

    print("Evaluating performance")
    performance_results = performance_prediction(predictions)
    performance_results.to_csv(
        results_dir / f"{name}_performance_results.csv", index=False
    )
    plot_chex_performance(performance_results, results_dir, name)
    print("Evaluating fairness")
    fairness_results = fairness_prediction(predictions)
    fairness_results.to_csv(results_dir / f"{name}_fairness_results.csv", index=False)
