import os
import argparse
import pandas as pd



def load_results(file_path):

    """
    Load experimental results.
    """

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Cannot find {file_path}"
        )


    return pd.read_csv(file_path)



def save_latex_table(
        dataframe,
        output_path
):

    """
    Save dataframe as LaTeX table.
    """

    latex = dataframe.to_latex(
        index=False,
        float_format="%.4f"
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(latex)



def generate_table(
        input_file,
        output_name,
        output_dir
):

    """
    Generate csv and latex table.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    df = load_results(
        input_file
    )


    # save csv

    csv_path = os.path.join(
        output_dir,
        output_name + ".csv"
    )


    df.to_csv(
        csv_path,
        index=False
    )


    # save latex

    tex_path = os.path.join(
        output_dir,
        output_name + ".tex"
    )


    save_latex_table(
        df,
        tex_path
    )


    print(
        f"Generated {output_name}"
    )



def main():

    parser = argparse.ArgumentParser(
        description=
        "Generate RA4KT experimental tables"
    )


    parser.add_argument(
        "--result_dir",
        type=str,
        default="results"
    )


    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/tables"
    )


    args = parser.parse_args()



    experiments = [

        (
            "main_results.csv",
            "main_results"
        ),

        (
            "ablation_results.csv",
            "ablation_results"
        ),

        (
            "attribution_results.csv",
            "attribution_results"
        )

    ]



    for filename, output_name in experiments:


        input_path = os.path.join(
            args.result_dir,
            filename
        )


        if os.path.exists(input_path):

            generate_table(

                input_path,

                output_name,

                args.output_dir

            )

        else:

            print(
                f"Skip {filename}: not found"
            )



if __name__ == "__main__":

    main()