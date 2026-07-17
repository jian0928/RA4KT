import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt



def load_csv(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{path} not found"
        )

    return pd.read_csv(path)



def save_figure(
        output_path
):

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved figure: {output_path}"
    )



def plot_lambda_cons(
        input_file,
        output_dir
):

    """
    Sensitivity analysis of lambda_cons.
    """

    data = load_csv(
        input_file
    )


    plt.figure(
        figsize=(6,4)
    )


    plt.plot(
        data["lambda_cons"],
        data["AUC"],
        marker="o",
        label="AUC"
    )


    plt.plot(
        data["lambda_cons"],
        data["ACC"],
        marker="s",
        label="ACC"
    )


    plt.xlabel(
        r"$\lambda_{cons}$"
    )

    plt.ylabel(
        "Performance"
    )


    plt.legend()


    save_figure(
        os.path.join(
            output_dir,
            "lambda_cons_sensitivity.png"
        )
    )



def plot_lambda_coh(
        input_file,
        output_dir
):

    """
    Sensitivity analysis of lambda_coh.
    """

    data = load_csv(
        input_file
    )


    plt.figure(
        figsize=(6,4)
    )


    plt.plot(
        data["lambda_coh"],
        data["AUC"],
        marker="o",
        label="AUC"
    )


    plt.plot(
        data["lambda_coh"],
        data["ACC"],
        marker="s",
        label="ACC"
    )


    plt.xlabel(
        r"$\lambda_{coh}$"
    )


    plt.ylabel(
        "Performance"
    )


    plt.legend()


    save_figure(
        os.path.join(
            output_dir,
            "lambda_coh_sensitivity.png"
        )
    )



def plot_convergence(
        input_file,
        output_dir
):

    """
    Training convergence curve.
    """

    data = load_csv(
        input_file
    )


    plt.figure(
        figsize=(6,4)
    )


    plt.plot(
        data["epoch"],
        data["loss"],
        label="Training Loss"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.legend()


    save_figure(
        os.path.join(
            output_dir,
            "convergence_curve.png"
        )
    )



def plot_attribution_quality(
        input_file,
        output_dir
):

    """
    Attribution evaluation visualization.
    """

    data = load_csv(
        input_file
    )


    metrics = [
        "Validity",
        "Sparsity",
        "Coherence"
    ]


    values = [
        data[m].mean()
        for m in metrics
    ]


    plt.figure(
        figsize=(6,4)
    )


    plt.bar(
        metrics,
        values
    )


    plt.ylabel(
        "Score"
    )


    save_figure(
        os.path.join(
            output_dir,
            "attribution_quality.png"
        )
    )



def main():

    parser = argparse.ArgumentParser(
        description=
        "Generate RA4KT experimental figures"
    )


    parser.add_argument(
        "--result_dir",
        type=str,
        default="results"
    )


    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/figures"
    )


    args = parser.parse_args()



    os.makedirs(
        args.output_dir,
        exist_ok=True
    )


    tasks = [

        (
            "sensitivity_lambda_cons.csv",
            plot_lambda_cons
        ),

        (
            "sensitivity_lambda_coh.csv",
            plot_lambda_coh
        ),

        (
            "convergence.csv",
            plot_convergence
        ),

        (
            "attribution_quality.csv",
            plot_attribution_quality
        )

    ]


    for filename, func in tasks:

        input_path = os.path.join(
            args.result_dir,
            filename
        )


        if os.path.exists(input_path):

            func(
                input_path,
                args.output_dir
            )

        else:

            print(
                f"Skip {filename}: not found"
            )



if __name__ == "__main__":

    main()