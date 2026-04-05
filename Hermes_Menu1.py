# -------------------- Import Modules --------------------
import logging

from Hermes.Hermes_ML_UnSup_Engine import HermesEngine
from Hermes.Menu_Helper_Decorator import (
    column_list,
    input_int,
    input_list,
    menu_wrapper,
)

logger = logging.getLogger("Hermes")


# -------------------- Loaded ML Data Menu --------------------
@menu_wrapper("Loaded ML Data")
def loaded_ml_data_menu(hermes: HermesEngine):
    """
    Interactively load a machine-learning dataset into the Hermes workflow.

    This menu displays the available folders in the current working place,
    allows the user to choose a folder and file by numeric index, and then
    delegates the actual dataset-loading process to
    ``HermesEngine.ml_dataset_search()``.

    If dataset loading succeeds, the Hermes engine stores the loaded source
    data and builds the downstream cores required for later unsupervised
    learning operations.

    Parameters
    ----------
    hermes : HermesEngine
        Active Hermes engine instance used to manage dataset loading and
        workflow state.

    Returns
    -------
    None
        Returns when dataset loading succeeds or when the user exits the menu.

    Workflow
    --------
    1. Display available working-place folders.
    2. Prompt the user to choose a folder number.
    3. Display files inside the selected folder.
    4. Prompt the user to choose a file number.
    5. Load the selected dataset through ``hermes.ml_dataset_search()``.
    6. Exit the menu if loading succeeds; otherwise repeat the process.

    Notes
    -----
    This menu is responsible only for interactive folder and file selection.
    The actual dataset-opening logic and dependent-core construction are handled
    by the Hermes engine.

    If the user cancels during folder selection or file selection, the menu
    exits immediately.

    Examples
    --------
    >>> loaded_ml_data_menu(hermes)
    """
    logger.info("Opening Loaded ML Data Menu")
    while True:
        # ---------- Show working-place folders ----------
        folders = hermes.hunter_core.working_place_searcher()
        logger.info("Retrieved working-place folders: folder_count=%s", len(folders))

        print(f"\n----- 🔥 Folder Lists 🔥-----\n{'-'*50}")
        for i, folder in folders.items():
            print(f"📂 {i}. {folder}")

        # ---------- Select folder ----------
        selected_folder_num = input_int("🕯️ Select folder")

        if selected_folder_num is None:
            logger.info("Loaded ML Data Menu cancelled during folder selection")
            return None

        logger.info("Selected folder number: %s", selected_folder_num)

        # ---------- Show files inside selected folder ----------
        files = hermes.hunter_core.files_searcher_from_folders(
            selected_folder_num=selected_folder_num,
        )
        logger.info(
            "Retrieved file list for folder_num=%s: file_count=%s",
            selected_folder_num,
            len(files),
        )

        print(f"\n----- 🔥 File Lists 🔥-----\n{'-'*50}")
        for i, file in files.items():
            print(f"📄 {i}. {file}")

        # ---------- Select file ----------
        selected_file_num = input_int("🕯️ Select file")

        if selected_file_num is None:
            logger.info("Loaded ML Data Menu cancelled during file selection")
            return None

        logger.info(
            "Selected file number: folder_num=%s, file_num=%s",
            selected_folder_num,
            selected_file_num,
        )

        # ---------- Load dataset to Zeus Engine ----------
        loaded_data = hermes.ml_dataset_search(
            selected_folder_num=selected_folder_num,
            selected_file_num=selected_file_num,
        )

        if loaded_data is None:
            logger.warning(
                "Failed to load ML dataset: folder_num=%s, file_num=%s",
                selected_folder_num,
                selected_file_num,
            )
            print("⚠️ Failed to load ML dataset ‼️")
            continue

        logger.info(
            "ML dataset loaded successfully: folder_num=%s, file_num=%s",
            selected_folder_num,
            selected_file_num,
        )
        print(f"🔥 ML dataset loaded successfully.\n{'-' * 100}")
        return


# -------------------- Select Feature Menu --------------------
@menu_wrapper("Select Input Columns")
def select_input_menu(hermes: HermesEngine):
    """
    Interactively select input columns for Hermes unsupervised-learning workflows.

    This menu displays all available columns in the currently loaded source
    dataset, allows the user to select one or more input columns by numeric
    index, and then asks the Hermes engine to build the corresponding ``X``
    dataset used for downstream unsupervised-learning tasks such as clustering.

    If the user provides no explicit input-column selection, the Hermes engine
    is instructed to build ``X`` automatically using all available columns.

    Parameters
    ----------
    hermes : HermesEngine
        Active Hermes engine instance used to manage source data, input-column
        selection, and clustering-input construction.

    Returns
    -------
    None
        Returns when the user confirms the current input selection, exits the
        menu, or when required source data is unavailable.

    Workflow
    --------
    1. Confirm that source data and input core are available.
    2. Display all columns in the current source dataset.
    3. Prompt the user to choose one or more input-column indices.
    4. Convert valid numeric selections into real column names.
    5. Pass selected input columns to ``hermes.select_input_data()``.
    6. Build the corresponding ``X`` dataset.
    7. Display the current input selection and X summary.
    8. Let the user confirm, reselect, or leave the menu.

    Input Rules
    -----------
    - The user must select columns using numeric indices shown in the column list.
    - Multiple selections are entered as a comma-separated list.
    - If the user presses Enter without input, Hermes uses all columns.
    - If the user enters the back command, the menu exits immediately.

    Notes
    -----
    This menu does not directly manipulate the input core or build ``X`` by
    itself. It only collects user input and delegates the actual workflow
    operations to the Hermes engine.

    Column validation, input-state updates, and X construction are handled by
    the engine and the underlying unsupervised input core.

    Examples
    --------
    >>> select_input_menu(hermes)
    """
    logger.info("Opening Select Input Menu")

    if hermes.source_data is None:
        logger.warning("Select Input Menu blocked: source_data is None")
        print("⚠️ No source data available. Please load data first ‼️")
        return

    if hermes.input_core is None:
        logger.warning("Select Input Menu blocked: input_core is None")
        print("⚠️ InputCore has not been built. Please load data first ‼️")
        return

    while True:
        # ---------- Show available columns ----------
        col_map = column_list(hermes.source_data)
        logger.info("Displayed selectable columns: column_count=%s", len(col_map))

        if not col_map:
            logger.warning("No columns available for input selection")
            print("⚠️ No columns available for selection ‼️")
            return

        # ---------- Select feature columns by index ----------
        input_columns = input_list("🕯️ Select input column index(es)")
        if input_columns == "__BACK__":
            logger.info("Select Input Menu cancelled during input-column selection")
            return

        if input_columns is None:
            logger.info("No input columns entered. Using all columns automatically")
            print("🔔 No input columns entered. Using all columns automatically.")
            x_data = hermes.select_input_data()

        else:
            if not all(str(item).isdigit() for item in input_columns):
                logger.warning(
                    "Invalid input-column selection: non-numeric indices=%s",
                    input_columns,
                )
                print("⚠️ INPUT COLUMN selections must all be numeric indices ‼️")
                continue

            # ---------- Exchanging str type to int type ----------
            input_indices = [int(item) for item in input_columns]

            if any(idx not in col_map for idx in input_indices):
                logger.warning(
                    "Input-column selection out of range: indices=%s", input_indices
                )
                print("⚠️ One or more INPUT indices are out of range ‼️")
                continue

            # ---------- Get column names from source data ----------
            selected_input_columns = [col_map[idx] for idx in input_indices]
            logger.info("Selected input columns: %s", selected_input_columns)

            # ---------- Convey to Hermes engine ----------
            x_data = hermes.select_input_data(
                input_columns=selected_input_columns,
            )

        if x_data is None:
            logger.warning("Failed to build clustering X data from selected inputs")
            print("⚠️ Failed to build X ‼️")
            continue

        # ---------- Show the sleected column info. ----------
        print("🔥 Input selection completed.")
        print("👓 Show the selected inputs.")
        hermes.show_current_input_selection()

        # ---------- Final confirmations ----------
        while True:
            confirm = input_int("🕯️ (1) Confirm selection | (2) Reselect | (0) Back")

            if confirm == 1:
                logger.info("Current input selection confirmed")
                print("🔥 Current selection confirmed.")
                return

            elif confirm == 2:
                hermes.reset_input_selection()
                logger.info("Input reselection requested")
                print("♻️ Input selection has been reset. Please select again.")
                break

            elif confirm == 0 or confirm is None:
                logger.info("Select Input Menu exited during confirmation step")
                return

            else:
                logger.warning(
                    "Invalid confirmation selection in Select Input Menu: confirm=%s",
                    confirm,
                )
                print("⚠️ Invalid selection ‼️")


# =================================================
