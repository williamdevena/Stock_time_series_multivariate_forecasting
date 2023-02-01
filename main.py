"""
This module runs the entire project. By running this module all
the stages will be executed: data acquisition, data preparation,
data visualization and data inference.

- All the the raw data acquired will be stored locally in the "data"
folder, together with two files that contain descriptive statistics
of the data

- All the plots produced during the experiments will be saved in the
"plots" folder

- All the logs (including the final metrics) will be written in
the "project_log/assignment.log" file

"""

import logging
import os


from src import (costants, data_acquisition,
                 data_inference, data_preparation, data_storing,
                 data_visualization, logging_utilities, seeds)


def main():
    """
    Runs the entire project (data acquisition, data preparation,
    data visualization and data inference).

    - Plots folder: "./plots"

    - Data folder (contains also the inference and
    residual plots): "./data"

    - Log file (contains also results metrics at the end):
    "./project_log/assignment.log"

    Args: None

    Returns: None
    """

    # CREATING LOG FOLDER
    if not os.path.exists("./project_log"):
        os.mkdir("./project_log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler("project_log/assignment.log"),
            logging.StreamHandler()
        ]
    )

    # SET SEEDS
    seeds.set_seeds()

    # DATA ACQUISITION AND STORING
    data = data_acquisition.data_acquisition()

    # DATA STORING
    data_storing.data_storing(
        data=data, raw=True)

    # DATA PREPARATION STAGE 1 (DATA CLEANING)
    prepared_data = data_preparation.data_preparation_stage1(data)

    # DATA VISUALIZATION
    data_visualization.data_visualization(data=prepared_data)

    # DATA PREPARATION STAGE 2 (OUTLIERS, DROP COLUMNS, SCALING)
    final_df, df_test, final_df_only_close, df_test_only_close = data_preparation.data_preparation_stage2(
        data=prepared_data
    )

    # MULTIVARIATE INFERENCE
    logging_utilities.print_name_stage_project("MULTIVARIATE INFERENCE")
    data_inference.data_inference(
        mode='multivariate',
        final_df=final_df,
        df_test=df_test,
        plot_path=os.path.join(costants.PLOTS_FOLDER, "multivariate_inference"))

    # UNIVARIATE INFERENCE
    logging_utilities.print_name_stage_project("UNIVARIATE INFERENCE")
    data_inference.data_inference(
        mode='univariate',
        final_df=final_df_only_close,
        df_test=df_test_only_close,
        plot_path=os.path.join(costants.PLOTS_FOLDER, "univariate_inference"))


if __name__ == "__main__":
    main()
