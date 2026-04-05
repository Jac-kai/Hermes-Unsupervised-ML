# -------------------- Import Modules --------------------
import logging
import os

from Hermes.Hermes_Model_Menu_Helper import (
    collect_common_cluster_params,
    collect_model_cluster_kwargs,
    select_model_name,
)
from Hermes.Hermes_ML_UnSup_Engine import HermesEngine
from Hermes.Menu_Helper_Decorator import input_int, menu_wrapper

logger = logging.getLogger("Hermes")


# -------------------- Helper: select saved model file --------------------
def _select_saved_model_file(hermes: HermesEngine, model_name: str) -> str | None:
    """
    Display saved model files for a selected clustering model type and let the
    user choose one by menu number.

    This helper reads the available saved ``.joblib`` files for the specified
    clustering model type through ``HermesEngine._get_saved_model_files()``,
    displays the file names as a numbered terminal menu, and returns the full
    file path selected by the user.

    Parameters
    ----------
    hermes : HermesEngine
        Active Hermes engine instance used to retrieve saved model files.
    model_name : str
        Name of the clustering model type whose saved model files should be
        listed.

    Returns
    -------
    str or None
        Full file path of the selected saved model file.

        - ``str``: selected saved model file path
        - ``None``: when no saved files exist, the user cancels, or the selected
          menu number is invalid

    Workflow
    --------
    1. Retrieve saved model files for the given model type.
    2. If no files exist, print a warning and return ``None``.
    3. Build a numbered mapping from menu indices to file paths.
    4. Display saved file names using base file names only.
    5. Ask the user to choose one file by menu number.
    6. Validate the selection.
    7. Return the selected full file path.

    Notes
    -----
    - This helper only selects a saved file; it does not load the model itself.
    - File discovery is delegated to the Hermes engine.
    - Displayed file names use ``os.path.basename()`` for cleaner terminal
      output.
    """
    logger.info("Selecting saved model file for model_name=%s", model_name)
    saved_files = hermes._get_saved_model_files(model_name)

    if not saved_files:
        logger.warning("No saved model files found for model_name=%s", model_name)
        print("⚠️ No saved model files found ‼️")
        return None

    file_map = {i: path for i, path in enumerate(saved_files, 1)}

    print(f"\n----- 🔥 Saved {model_name} Models 🔥 -----")
    for i, path in file_map.items():
        print(f"📦 {i}. {os.path.basename(path)}")
    print("-" * 50)

    selected_num = input_int("🕯️ Select saved model", default=-1)
    if selected_num is None:
        logger.info(
            "Saved model file selection cancelled for model_name=%s", model_name
        )
        return None

    if selected_num not in file_map:
        logger.warning(
            "Saved model selection out of range: model_name=%s, selected_num=%s",
            model_name,
            selected_num,
        )
        print("⚠️ Saved model selection is out of range ‼️")
        return None

    logger.info(
        "Saved model file selected: model_name=%s, filepath=%s",
        model_name,
        file_map[selected_num],
    )
    return file_map[selected_num]


# -------------------- Helper: run clustering workflow --------------------
def _run_clustering_workflow(hermes: HermesEngine):
    """
    Run the full clustering workflow from model selection to result confirmation.

    This helper validates that source data, input core, and clustering feature
    data already exist, then guides the user through the main clustering
    execution flow:

    1. select a clustering model,
    2. select common preprocessing parameters,
    3. select model-specific clustering parameters,
    4. run clustering,
    5. review the current clustering summary,
    6. confirm, save, rerun, or leave.

    Parameters
    ----------
    hermes : HermesEngine
        Active Hermes engine instance used to access source data, clustering
        feature data, model execution, and model saving.

    Returns
    -------
    None
        Returns when the user confirms the clustering result, exits the
        workflow, or when required prerequisites are missing.

    Workflow
    --------
    1. Validate that source data has been loaded.
    2. Validate that the input core has been built.
    3. Validate that clustering feature data ``X`` has been built.
    4. Ask the user to select a clustering model.
    5. Collect common clustering preprocessing parameters.
    6. Collect model-specific clustering hyperparameters.
    7. Merge all clustering keyword arguments.
    8. Execute clustering through ``HermesEngine.cluster_model()``.
    9. Show the current clustering summary.
    10. Allow the user to confirm, save, rerun, or leave.

    Notes
    -----
    - This helper delegates actual clustering execution to the Hermes engine.
    - Saving the trained model after a successful run is optional and handled
      through ``HermesEngine.save_current_model()``.
    - Selecting rerun returns to the outer clustering loop and restarts the
      full parameter-selection workflow.
    """
    logger.info("Starting clustering workflow")

    if hermes.source_data is None:
        logger.warning("Clustering workflow blocked: source_data is None")
        print("⚠️ No source data available. Please load data first ‼️")
        return

    if hermes.input_core is None:
        logger.warning("Clustering workflow blocked: input_core is None")
        print("⚠️ InputCore has not been built. Please load data first ‼️")
        return

    if hermes.cluster_input_data is None:
        logger.warning("Clustering workflow blocked: cluster_input_data is None")
        print("⚠️ X has not been built yet. Please select input columns first ‼️")
        return

    while True:
        # ---------- Select clustering model ----------
        model_name = select_model_name(hermes)
        if model_name is None:
            logger.info("Clustering workflow cancelled during model selection")
            return

        # ---------- Collect common clustering params ----------
        common_params = collect_common_cluster_params(hermes)
        if common_params is None:
            logger.info(
                "Clustering workflow cancelled during common parameter selection"
            )
            return

        # ---------- Collect model-specific kwargs ----------
        model_kwargs = collect_model_cluster_kwargs(model_name)
        if model_kwargs is None:
            logger.info(
                "Clustering workflow cancelled during model-specific parameter selection: %s",
                model_name,
            )
            return

        # ---------- Merge all clustering kwargs ----------
        cluster_kwargs = {
            **common_params,
            **model_kwargs,
        }

        # ---------- Run clustering ----------
        result = hermes.cluster_model(
            model_name=model_name,
            **cluster_kwargs,
        )

        if result is None:
            print("⚠️ Failed to run clustering model ‼️")
            logger.warning(
                "Cluster model execution failed for model_name=%s", model_name
            )
            continue

        logger.info("Cluster model executed successfully: %s", model_name)
        print("🔥 Clustering model executed successfully.")
        hermes.show_current_model_summary()

        while True:
            confirm = input_int(
                "🕯️ (1) Confirm result | (2) Save model | (3) Rerun | (0) Back"
            )

            if confirm == 1:
                logger.info("Current clustering result confirmed: %s", model_name)
                print("🔥 Current clustering result confirmed.")
                return

            elif confirm == 2:
                logger.info(
                    "Saving current clustering model from run workflow: %s",
                    model_name,
                )
                hermes.save_current_model()

            elif confirm == 3:
                logger.info("Rerun requested for clustering workflow")
                break

            elif confirm == 0 or confirm is None:
                logger.info(
                    "Clustering workflow exited after model run: %s", model_name
                )
                return

            else:
                logger.warning(
                    "Invalid clustering workflow confirm selection: model_name=%s, confirm=%s",
                    model_name,
                    confirm,
                )
                print("⚠️ Invalid selection ‼️")


# -------------------- Helper: load trained model workflow --------------------
def _load_trained_model_workflow(hermes: HermesEngine):
    """
    Run the trained-model loading workflow for Hermes clustering models.

    This helper validates that source data, input core, and clustering feature
    data already exist, then guides the user through selecting a clustering
    model type, choosing a saved model file, and loading the trained model into
    the current Hermes engine session.

    Parameters
    ----------
    hermes : HermesEngine
        Active Hermes engine instance used to resolve saved model files and load
        the selected trained model.

    Returns
    -------
    None
        Returns when the trained model is loaded successfully, when the user
        exits the workflow, or when prerequisite workflow state is missing.

    Workflow
    --------
    1. Validate that source data has been loaded.
    2. Validate that the input core has been built.
    3. Validate that clustering feature data ``X`` has been built.
    4. Ask the user to select a clustering model type.
    5. Ask the user to select a saved model file for that type.
    6. Load the trained model through ``HermesEngine.load_trained_model()``.
    7. Display the current clustering summary after loading.

    Notes
    -----
    - This helper assumes that loading a trained clustering model still requires
      the current clustering feature data to exist.
    - Saved model selection is delegated to ``_select_saved_model_file()``.
    - After loading succeeds, the loaded model becomes the current active model
      in the Hermes engine.
    """
    logger.info("Starting trained model loading workflow")

    if hermes.source_data is None:
        logger.warning("Load trained model workflow blocked: source_data is None")
        print("⚠️ No source data available. Please load data first ‼️")
        return

    if hermes.input_core is None:
        logger.warning("Load trained model workflow blocked: input_core is None")
        print("⚠️ InputCore has not been built. Please load data first ‼️")
        return

    if hermes.cluster_input_data is None:
        logger.warning(
            "Load trained model workflow blocked: cluster_input_data is None"
        )
        print("⚠️ X has not been built yet. Please select input columns first ‼️")
        return

    model_name = select_model_name(hermes)
    if model_name is None:
        logger.info("Load trained model workflow cancelled during model selection")
        return

    filepath = _select_saved_model_file(hermes, model_name)
    if filepath is None:
        logger.info(
            "Load trained model workflow cancelled during file selection: %s",
            model_name,
        )
        return

    logger.info(
        "Loading trained clustering model: model_name=%s, filepath=%s",
        model_name,
        filepath,
    )
    loaded_model = hermes.load_trained_model(model_name=model_name, filepath=filepath)

    if loaded_model is None:
        print("⚠️ Failed to load trained model ‼️")
        logger.warning("Failed to load trained clustering model: %s", model_name)
        return

    logger.info(
        "Trained clustering model loaded successfully from workflow: model_name=%s, filepath=%s",
        model_name,
        filepath,
    )
    print("🔥 Trained clustering model loaded successfully.")
    hermes.show_current_model_summary()


# -------------------- Cluster Model Menu --------------------
@menu_wrapper("Cluster Model")
def cluster_model_menu(hermes: HermesEngine):
    """
    Open the clustering model management menu for Hermes.

    This menu acts as the main model-management entry point for clustering
    workflows. It provides a single terminal menu where the user can:

    1. run a clustering model,
    2. save the current clustering model,
    3. load a previously trained clustering model,
    4. show the current clustering model summary.

    Parameters
    ----------
    hermes : HermesEngine
        Active Hermes engine instance used to manage clustering models, trained
        model persistence, and clustering summaries.

    Returns
    -------
    None
        Returns when the user exits the cluster model menu.

    Workflow
    --------
    1. Display the cluster model management menu.
    2. Ask the user to select a service.
    3. Dispatch the selected service to the corresponding helper or engine
       method.
    4. Keep the menu open until the user chooses to go back.

    Menu Services
    -------------
    1. Run Clustering Model
        Open the full clustering workflow and execute a clustering model.
    2. Save Current Model
        Save the current clustering model through the Hermes engine.
    3. Load Trained Model
        Open the trained-model loading workflow.
    4. Show Current Model Summary
        Display the summary of the currently active clustering model.

    Notes
    -----
    - This menu only controls model-management flow and does not implement the
      clustering algorithm logic itself.
    - Clustering execution is delegated to ``_run_clustering_workflow()``.
    - Trained-model loading is delegated to ``_load_trained_model_workflow()``.
    - Model saving and summary display are delegated directly to the Hermes
      engine.
    """
    logger.info("Opening Cluster Model Menu")

    while True:
        print("\n━━━━━━━━🏮  Cluster Model Menu 🏮 ━━━━━━━")
        print("1. 🧠 Run Clustering Model")
        print("2. 💾 Save Current Model")
        print("3. 📥 Load Trained Model")
        print("4. 🪶 Show Current Model Summary")
        print("0. ↩️ Back")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        selected_service = input_int("🕯️ Select Services", default=-1)

        if selected_service is None or selected_service == 0:
            logger.info("Exiting Cluster Model Menu")
            return

        if selected_service == 1:
            logger.info("Cluster Model Menu selection: Run Clustering Model")
            _run_clustering_workflow(hermes)

        elif selected_service == 2:
            logger.info("Cluster Model Menu selection: Save Current Model")
            hermes.save_current_model()

        elif selected_service == 3:
            logger.info("Cluster Model Menu selection: Load Trained Model")
            _load_trained_model_workflow(hermes)

        elif selected_service == 4:
            logger.info("Cluster Model Menu selection: Show Current Model Summary")
            hermes.show_current_model_summary()

        else:
            logger.warning(
                "Invalid Cluster Model Menu selection: selected_service=%s",
                selected_service,
            )
            print("⚠️ Invalid selection ‼️")


# =================================================
