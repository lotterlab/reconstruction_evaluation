import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from plot_ucsf import (
    plot_ucsf_performance,
)


def bootstrap_fairness_classifier(
    interpreter_results,
    attribute,
    attribute_values,
    interpreter_name,
    gt_name,
    n_iterations=1000,
):
    tpr_class = []
    fpr_class = []
    tpr_recon = []
    fpr_recon = []

    for val in attribute_values:
        subgroup = interpreter_results[interpreter_results[attribute] == val]
        if len(subgroup) == 0:
            continue
        tpr_class.append(get_tpr(subgroup[f"{interpreter_name}"], subgroup[gt_name]))
        fpr_class.append(get_fpr(subgroup[f"{interpreter_name}"], subgroup[gt_name]))
        tpr_recon.append(
            get_tpr(subgroup[f"{interpreter_name}_recon"], subgroup[gt_name])
        )
        fpr_recon.append(
            get_fpr(subgroup[f"{interpreter_name}_recon"], subgroup[gt_name])
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
                    subgroup[f"{interpreter_name}"],
                    subgroup[gt_name],
                )
            )
            boot_fpr_class.append(
                get_fpr(
                    subgroup[f"{interpreter_name}"],
                    subgroup[gt_name],
                )
            )
            boot_tpr_recon.append(
                get_tpr(
                    subgroup[f"{interpreter_name}_recon"],
                    subgroup[gt_name],
                )
            )
            boot_fpr_recon.append(
                get_fpr(
                    subgroup[f"{interpreter_name}_recon"],
                    subgroup[gt_name],
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
        "eodd_class_bootstrapped": np.mean(boot_baseline_eodd),
        "eop_class_bootstrapped": np.mean(boot_baseline_eop),
        "eodd_recon_bootstrapped": np.mean(boot_recon_eodd),
        "eop_recon_bootstrapped": np.mean(boot_recon_eop),
        "eodd_class_std_bootstrapped": np.std(boot_baseline_eodd),
        "eop_class_std_bootstrapped": np.std(boot_baseline_eop),
        "eodd_recon_std_bootstrapped": np.std(boot_recon_eodd),
        "eop_recon_std_bootstrapped": np.std(boot_recon_eop),
    }

    return results


def get_tpr(y_pred, y):
    # calculate true positive rate
    y_pred = y_pred.astype(int)
    y = y.astype(int)
    if y.sum() == 0:  # if no positive samples
        return 0.0
    tpr = y_pred[y == 1].sum() / y.sum()
    return tpr


def get_fpr(y_pred, y):
    # calculate false positive rate
    y_pred = y_pred.astype(int)
    y = y.astype(int)
    n_neg = len(y) - y.sum()
    if n_neg == 0:  # if no negative samples
        return 0.0
    fpr = y_pred[y == 0].sum() / n_neg
    return fpr


def calculate_threshold(y_pred, y_true):
    # calculate threshold for the metric
    fpr, sens, threshs = roc_curve(y_true, y_pred)
    spec = 1 - fpr
    return threshs[np.argmin(np.abs(spec - sens))]


def fairness_prediction_classifier(predictions):
    predictions = predictions.copy()  # Make a shallow copy of the list

    sensitive_attributes = {
        "sex": (["M", "F"], "gender"),
        "age_bin": (["O", "Y"], "age"),
    }

    interpreters = {
        "ttype": ("TTypeBCEClassifier", "diagnosis"),
        "tgrade": ("TGradeBCEClassifier", "cns"),
    }

    all_results = []
    for prediction in predictions:
        # Make a deep copy of the prediction dictionary
        prediction = {
            "model": prediction["model"],
            "acceleration": prediction["acceleration"],
            "prediction_results": prediction["prediction_results"].copy(),
            "fairness": prediction["fairness"],
        }
        if prediction["fairness"]:
            original_model = prediction["model"]
            prediction_results = prediction["prediction_results"]
            prediction_results["age_bin"] = prediction_results["age"].apply(
                lambda x: (
                    "Y" if float(str(x).split("(")[1].split(",")[0]) <= 58 else "O"
                )
            )
            prediction_results["sex"] = prediction_results["sex"].apply(
                lambda x: "M" if float(str(x).split("(")[1].split(")")[0]) == 1 else "F"
            )
            prediction_results["cns"] = prediction_results["cns"].apply(
                lambda x: float(str(x).split("(")[1].split(")")[0])
            )
            prediction_results["diagnosis"] = prediction_results["diagnosis"].apply(
                lambda x: float(str(x).split("(")[1].split(")")[0])
            )

            for interpreter, (interpreter_name, gt_name) in interpreters.items():
                # Get patient-level predictions and include sensitive attributes in the groupby
                patient_preds = prediction_results.groupby("patient_id").agg(
                    {
                        f"{interpreter_name}": "median",
                        f"{interpreter_name}_recon": "median",
                        gt_name: "first",
                        "sex": "first",
                        "age_bin": "first",
                    }
                )

                baseline_threshold = calculate_threshold(
                    patient_preds[f"{interpreter_name}"], patient_preds[gt_name]
                )
                recon_threshold = calculate_threshold(
                    patient_preds[f"{interpreter_name}_recon"],
                    patient_preds[gt_name],
                )

                patient_preds[f"{interpreter_name}"] = patient_preds[
                    f"{interpreter_name}"
                ].apply(lambda x: 1 if x >= baseline_threshold else 0)
                patient_preds[f"{interpreter_name}_recon"] = patient_preds[
                    f"{interpreter_name}_recon"
                ].apply(lambda x: 1 if x >= recon_threshold else 0)

                for attribute, (
                    attribute_values,
                    attribute_name,
                ) in sensitive_attributes.items():
                    results = bootstrap_fairness_classifier(
                        patient_preds,
                        attribute,
                        attribute_values,
                        interpreter_name,
                        gt_name,
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
                            "metric": "EODD-std-err-bootstrapped",
                            "value": results["eodd_class_std_bootstrapped"],
                        }
                    )
                    all_results.append(
                        {
                            "model": original_model,
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "EOP-std-err-bootstrapped",
                            "value": results["eop_class_std_bootstrapped"],
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
                            "metric": "EOP-bootstrapped",
                            "value": results["eop_recon_bootstrapped"],
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
                            "metric": "EOP-bootstrapped",
                            "value": results["eop_recon_bootstrapped"],
                        }
                    )
                    all_results.append(
                        {
                            "model": "baseline",
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "EODD-bootstrapped",
                            "value": results["eodd_recon_bootstrapped"],
                        }
                    )

    evaluation_results = pd.DataFrame(all_results)
    return evaluation_results


def bootstrap_fairness_segmentation(
    interpreter_results, attribute, attribute_values, n_iterations=1000
):

    dice_class = []
    dice_recon = []

    for val in attribute_values:
        subgroup = interpreter_results[interpreter_results[attribute] == val]
        dice_class.append(subgroup["UNet_dice"].mean())
        dice_recon.append(subgroup["UNet_recon_dice"].mean())

    delta_dice_class = max(dice_class) - min(dice_class)
    delta_dice_recon = max(dice_recon) - min(dice_recon)

    ser_class = (1 - min(dice_class)) / (1 - max(dice_class))
    ser_recon = (1 - min(dice_recon)) / (1 - max(dice_recon))

    observed_delta_delta_dice = delta_dice_recon - delta_dice_class
    observed_delta_ser = ser_recon - ser_class

    boot_delta_delta_dice = []
    boot_delta_ser = []

    delta_dice_class_bootstrapped = []
    delta_dice_recon_bootstrapped = []
    ser_class_bootstrapped = []
    ser_recon_bootstrapped = []

    for i in range(n_iterations):
        dice_class = []
        dice_recon = []

        boot_interpreter_results = interpreter_results.sample(
            n=len(interpreter_results), replace=True
        )

        for val in attribute_values:
            subgroup = boot_interpreter_results[
                boot_interpreter_results[attribute] == val
            ]
            dice_class.append(subgroup["UNet_dice"].mean())
            dice_recon.append(subgroup["UNet_recon_dice"].mean())

        boot_ser_class = (1 - min(dice_class)) / (1 - max(dice_class))
        boot_ser_recon = (1 - min(dice_recon)) / (1 - max(dice_recon))

        ser_class_bootstrapped.append(boot_ser_class)
        ser_recon_bootstrapped.append(boot_ser_recon)

        boot_delta_dice_class = max(dice_class) - min(dice_class)
        boot_delta_dice_recon = max(dice_recon) - min(dice_recon)

        delta_dice_class_bootstrapped.append(boot_delta_dice_class)
        delta_dice_recon_bootstrapped.append(boot_delta_dice_recon)

        boot_delta_delta_dice.append(boot_delta_dice_recon - boot_delta_dice_class)
        boot_delta_ser.append(boot_ser_recon - boot_ser_class)

    boot_delta_delta_dice = np.array(boot_delta_delta_dice)
    boot_delta_ser = np.array(boot_delta_ser)
    delta_dice_class_bootstrapped = np.array(delta_dice_class_bootstrapped)
    delta_dice_recon_bootstrapped = np.array(delta_dice_recon_bootstrapped)
    ser_class_bootstrapped = np.array(ser_class_bootstrapped)
    ser_recon_bootstrapped = np.array(ser_recon_bootstrapped)

    if observed_delta_delta_dice >= 0:
        p_value_delta_delta_dice = np.mean(boot_delta_delta_dice <= 0)
    else:
        p_value_delta_delta_dice = np.mean(boot_delta_delta_dice >= 0)

    if observed_delta_ser >= 0:
        p_value_delta_ser = np.mean(boot_delta_ser <= 0)
    else:
        p_value_delta_ser = np.mean(boot_delta_ser >= 0)

    std_delta_delta_dice = np.std(boot_delta_delta_dice)
    std_delta_ser = np.std(boot_delta_ser)

    results = {
        "delta_dice_class": delta_dice_class,
        "delta_dice_recon": delta_dice_recon,
        "ser_class": ser_class,
        "ser_recon": ser_recon,
        "delta_delta_dice": observed_delta_delta_dice,
        "delta_ser": observed_delta_ser,
        "delta_delta_dice_bootstrapped": boot_delta_delta_dice.mean(),
        "delta_ser_bootstrapped": boot_delta_ser.mean(),
        "delta_delta_dice_p_value": p_value_delta_delta_dice,
        "delta_ser_p_value": p_value_delta_ser,
        "delta_delta_dice_std": std_delta_delta_dice,
        "delta_ser_std": std_delta_ser,
        "delta_dice_class_bootstrapped": delta_dice_class_bootstrapped.mean(),
        "delta_dice_recon_bootstrapped": delta_dice_recon_bootstrapped.mean(),
        "ser_class_bootstrapped": ser_class_bootstrapped.mean(),
        "ser_recon_bootstrapped": ser_recon_bootstrapped.mean(),
        "delta_dice_class_std_bootstrapped": np.std(delta_dice_class_bootstrapped),
        "delta_dice_recon_std_bootstrapped": np.std(delta_dice_recon_bootstrapped),
        "ser_class_std_bootstrapped": np.std(ser_class_bootstrapped),
        "ser_recon_std_bootstrapped": np.std(ser_recon_bootstrapped),
    }
    return results


def fairness_prediction_segmentation(predictions):
    predictions = predictions.copy()  # Make a shallow copy of the list

    sensitive_attributes = {
        "sex": (["M", "F"], "gender"),
        "age_bin": (["O", "Y"], "age"),
    }

    all_results = []
    for prediction in predictions:
        # Make a deep copy of the prediction dictionary
        prediction = {
            "model": prediction["model"],
            "acceleration": prediction["acceleration"],
            "prediction_results": prediction["prediction_results"].copy(),
            "fairness": prediction["fairness"],
        }
        if prediction["fairness"]:
            original_model = prediction["model"]
            prediction_results = prediction["prediction_results"]
            prediction_results["age_bin"] = prediction_results["age"].apply(
                lambda x: (
                    "Y" if float(str(x).split("(")[1].split(",")[0]) <= 58 else "O"
                )
            )
            prediction_results["sex"] = prediction_results["sex"].apply(
                lambda x: "M" if float(str(x).split("(")[1].split(")")[0]) == 1 else "F"
            )

            # Get patient-level predictions and include sensitive attributes in the groupby
            patient_preds = prediction_results.groupby("patient_id").agg(
                {
                    "UNet_dice": "mean",
                    "UNet_recon_dice": "mean",
                    "sex": "first",
                    "age_bin": "first",
                }
            )

            for attribute, (
                attribute_values,
                attribute_name,
            ) in sensitive_attributes.items():

                results = bootstrap_fairness_segmentation(
                    patient_preds, attribute, attribute_values
                )

                all_results.append(
                    {
                        "model": "baseline",
                        "interpreter": "dice",
                        "attribute": attribute_name,
                        "metric": "SER",
                        "value": results["ser_class"],
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "dice",
                        "attribute": attribute_name,
                        "metric": "SER",
                        "value": results["ser_recon"],
                    }
                )
                all_results.append(
                    {
                        "model": "baseline",
                        "interpreter": "dice",
                        "attribute": attribute_name,
                        "metric": "delta-dice",
                        "value": results["delta_dice_class"],
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "dice",
                        "attribute": attribute_name,
                        "metric": "delta-dice",
                        "value": results["delta_dice_recon"],
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "dice",
                        "attribute": attribute_name,
                        "metric": "delta-SER",
                        "value": results["delta_ser"],
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "dice",
                        "attribute": attribute_name,
                        "metric": "delta-delta-dice",
                        "value": results["delta_delta_dice"],
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "dice",
                        "attribute": attribute_name,
                        "metric": "delta-SER-p-value",
                        "value": results["delta_ser_p_value"],
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "dice",
                        "attribute": attribute_name,
                        "metric": "delta-delta-dice-p-value",
                        "value": results["delta_delta_dice_p_value"],
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "dice",
                        "attribute": attribute_name,
                        "metric": "delta-delta-dice-std-err",
                        "value": results["delta_delta_dice_std"],
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "dice",
                        "attribute": attribute_name,
                        "metric": "delta-SER-std-err",
                        "value": results["delta_ser_std"],
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "dice",
                        "attribute": attribute_name,
                        "metric": "delta-delta-dice-bootstrapped",
                        "value": results["delta_delta_dice_bootstrapped"],
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "dice",
                        "attribute": attribute_name,
                        "metric": "delta-SER-bootstrapped",
                        "value": results["delta_ser_bootstrapped"],
                    }
                )
                all_results.append(
                    {
                        "model": "baseline",
                        "interpreter": "dice",
                        "attribute": attribute_name,
                        "metric": "delta-dice-bootstrapped",
                        "value": results["delta_dice_class_bootstrapped"],
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "dice",
                        "attribute": attribute_name,
                        "metric": "delta-dice-bootstrapped",
                        "value": results["delta_dice_recon_bootstrapped"],
                    }
                )
                all_results.append(
                    {
                        "model": "baseline",
                        "interpreter": "dice",
                        "attribute": attribute_name,
                        "metric": "SER-bootstrapped",
                        "value": results["ser_class_bootstrapped"],
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "dice",
                        "attribute": attribute_name,
                        "metric": "SER-bootstrapped",
                        "value": results["ser_recon_bootstrapped"],
                    }
                )
                all_results.append(
                    {
                        "model": "baseline",
                        "interpreter": "dice",
                        "attribute": attribute_name,
                        "metric": "SER-std-err-bootstrapped",
                        "value": results["ser_class_std_bootstrapped"],
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "dice",
                        "attribute": attribute_name,
                        "metric": "SER-std-err-bootstrapped",
                        "value": results["ser_recon_std_bootstrapped"],
                    }
                )
                all_results.append(
                    {
                        "model": "baseline",
                        "interpreter": "dice",
                        "attribute": attribute_name,
                        "metric": "delta-dice-std-err-bootstrapped",
                        "value": results["delta_dice_class_std_bootstrapped"],
                    }
                )
                all_results.append(
                    {
                        "model": original_model,
                        "interpreter": "dice",
                        "attribute": attribute_name,
                        "metric": "delta-dice-std-err-bootstrapped",
                        "value": results["delta_dice_recon_std_bootstrapped"],
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
        "sex": (["M", "F"], "gender"),
        "age_bin": (["O", "Y"], "age"),
    }

    all_results = []
    for prediction in predictions:
        if prediction["fairness"]:
            original_model = prediction["model"]
            prediction_results = prediction["prediction_results"].copy()
            prediction_results["age_bin"] = prediction_results["age"].apply(
                lambda x: (
                    "Y" if float(str(x).split("(")[1].split(",")[0]) <= 58 else "O"
                )
            )
            prediction_results["sex"] = prediction_results["sex"].apply(
                lambda x: "M" if float(str(x).split("(")[1].split(")")[0]) == 1 else "F"
            )
            prediction_results = prediction_results.groupby("patient_id").agg(
                {
                    "psnr": "median",
                    "sex": "first",
                    "age_bin": "first",
                }
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
    predictions = predictions.copy()
    # Create a list to store all rows and create DataFrame at the end
    all_results = []
    metrics = {
        "dice": "UNet_dice",
        "dice_recon": "UNet_recon_dice",
        "sum_dice": "UNet_sum",
        "sum_recon": "UNet_recon_sum",
        "psnr": "psnr",
        "ssim": "ssim",
        "nrmse": "nrmse",
        "lpips": "lpips",
        "ttype": "TTypeBCEClassifier",
        "ttype_recon": "TTypeBCEClassifier_recon",
        "tgrade": "TGradeBCEClassifier",
        "tgrade_recon": "TGradeBCEClassifier_recon",
    }

    for prediction in predictions:
        original_model = prediction["model"]
        acceleration = prediction["acceleration"]
        prediction_results = prediction["prediction_results"]

        for metric, metric_name in metrics.items():

            # For classification metrics, calculate AUROC
            if metric in ["ttype", "ttype_recon"]:
                # First aggregate predictions per patient using mean
                patient_preds = prediction_results.groupby("patient_id")[
                    metric_name
                ].median()
                # Get ground truth (one per patient)
                patient_gt = prediction_results.groupby("patient_id")[
                    "diagnosis"
                ].first()
                # Calculate AUROC
                value = roc_auc_score(patient_gt, patient_preds)

            elif metric in ["tgrade", "tgrade_recon"]:
                # First aggregate predictions per patient using mean
                patient_preds = prediction_results.groupby("patient_id")[
                    metric_name
                ].median()
                # Get ground truth (one per patient)
                patient_gt = prediction_results.groupby("patient_id")["cns"].first()
                # Calculate AUROC
                value = roc_auc_score(patient_gt, patient_preds)

            else:
                # For other metrics, keep the original aggregation
                patient_results = prediction_results.groupby("patient_id")[metric_name]
                patient_values = patient_results.mean()
                value = patient_values.mean()

            current_model = (
                "baseline" if metric in ["ttype", "tgrade", "dice"] else original_model
            )

            if "recon" in metric:
                metric = metric.split("_")[0]

            # Add result to list
            all_results.append(
                {
                    "acceleration": acceleration,
                    "model": current_model,
                    "metric": metric,
                    "value": value,
                }
            )

    # Create final DataFrame from all results
    evaluation_results = pd.DataFrame(all_results)
    return evaluation_results


def evaluate_ucsf(config, results_dir, name):
    predictions = []

    for prediction in config["predictions"]:
        model = prediction["model"]
        acceleration = prediction["acceleration"]
        prediction_results = pd.read_csv(prediction["predictions"])
        fairness = prediction["fairness"]

        predictions.append(
            {
                "model": model,
                "acceleration": acceleration,
                "prediction_results": prediction_results,
                "fairness": fairness,
            }
        )

    print("Evaluating performance")
    performance_results = performance_prediction(predictions)
    performance_results.to_csv(
        results_dir / f"{name}_performance_results.csv", index=False
    )
    plot_ucsf_performance(performance_results, results_dir, name)
    fairness_results_classifier = fairness_prediction_classifier(predictions)
    fairness_results_segmentation = fairness_prediction_segmentation(predictions)
    fairness_results = pd.concat(
        [fairness_results_classifier, fairness_results_segmentation]
    )
    fairness_results.to_csv(results_dir / f"{name}_fairness_results.csv", index=False)
