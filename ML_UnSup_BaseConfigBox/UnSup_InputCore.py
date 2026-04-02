# -------------------- Import Modules --------------------
import pandas as pd


# -------------------- Unsupervised Input Core --------------------
class UnSupInputCore:
    """
    Manage unsupervised-learning input columns and build the corresponding X dataset.

    `UnSupInputCore` is a lightweight input-management core designed for
    unsupervised-learning workflows such as clustering, dimensionality reduction,
    and anomaly detection. Unlike supervised feature-management components, this
    core does not handle target columns or build `y`. Its responsibility is limited
    to validating the source dataset, storing selected input columns, and building
    the corresponding `X` DataFrame used by downstream models or preprocessing
    pipelines.

    The core supports two common workflows:

    1. Explicit input selection:
    - set input columns with `set_input_columns()`
    - build the matching `X` dataset with `build_x_data()`

    2. Default full-column workflow:
    - call `build_x_data()` directly
    - if no input columns were previously selected, all columns in the source
        dataset are used

    Parameters
    ----------
    source_data : pandas.DataFrame
        Source dataset from which unsupervised-learning input columns are selected.

    Attributes
    ----------
    source_data : pandas.DataFrame
        Original dataset used as the source for input-column selection.
    input_columns : list[str] or None
        Selected column names used to build the unsupervised-learning input matrix.
        If `None`, `build_x_data()` will default to all columns in `source_data`.
    X : pandas.DataFrame or None
        Built input dataset containing only the selected input columns.

    Notes
    -----
    This core is intentionally focused on input-column management only. It does not
    perform preprocessing, scaling, encoding, train/test splitting, or model
    training. Those responsibilities should remain in upper workflow layers such as
    an engine, pipeline builder, or model-specific configuration class.
    """

    # -------------------- Initialization --------------------
    def __init__(self, source_data: pd.DataFrame):
        """
        Initialize the unsupervised input core with a source dataset.

        Parameters
        ----------
        source_data : pandas.DataFrame
            Source dataset used for input-column selection and X construction.

        Returns
        -------
        None
        """
        self.source_data = source_data  # Source data (From ML_Cleaned_Dataset folder)
        self.input_columns = None  # list[str] | None
        self.X = None  # pd.DataFrame | None

    # -------------------- Validation (helper) --------------------
    def _validation(self) -> bool:
        """
        Validate the current source dataset before input selection or X construction.

        This helper checks whether `source_data` exists, is a pandas DataFrame, and is
        not empty. It is used internally before operations that depend on the source
        dataset.

        Returns
        -------
        bool
            `True` if the source dataset is valid and ready for use, otherwise `False`.
        """
        if self.source_data is None:
            print("⚠️ No source data available ‼️")
            return False

        if not isinstance(self.source_data, pd.DataFrame):
            print("⚠️ Source data must be a pandas DataFrame ‼️")
            return False

        if self.source_data.empty:
            print("⚠️ Source data is empty ‼️")
            return False

        return True

    # -------------------- Reset input state --------------------
    def reset_input_state(self):
        """
        Reset the current input-selection state.

        This method clears both the stored input-column selection and the built `X`
        dataset so that a new unsupervised-learning input configuration can be created.

        Returns
        -------
        None
        """
        self.input_columns = None
        self.X = None
        print("☢️ Input state has been reset ☢️")

    # -------------------- Set input columns --------------------
    def set_input_columns(self, input_columns: list[str]):
        """
        Set and validate input columns for unsupervised learning.

        This method validates the provided input-column list, removes surrounding
        whitespace, checks for duplicates, and confirms that all selected columns exist
        in the source dataset. When validation succeeds, the cleaned column list is
        stored in `self.input_columns`.

        This method only records the selected input columns. It does not build `X`.
        Use `build_x_data()` to construct the actual input dataset.

        Parameters
        ----------
        input_columns : list[str]
            Column names to use as unsupervised-learning inputs.

        Returns
        -------
        list[str] or None
            Cleaned input-column list if successful, otherwise `None`.

        Validation Rules
        ----------------
        - `input_columns` must be a list
        - the list must not be empty
        - all items must be strings
        - blank column names are removed during cleaning
        - the cleaned result must contain at least one valid column
        - duplicate column names are not allowed
        - all selected columns must exist in `source_data`
        """
        if not self._validation():
            return None

        if not isinstance(input_columns, list):
            print("⚠️ Input columns must be provided as a list ‼️")
            return None

        if not input_columns:
            print("⚠️ Input columns must not be empty ‼️")
            return None

        if not all(isinstance(col, str) for col in input_columns):
            print("⚠️ All input column names must be strings ‼️")
            return None

        # ---------- Record input columns ----------
        cleaned_input_columns = [col.strip() for col in input_columns if col.strip()]

        # ---------- Check < 1 column and duplicates ----------
        if not cleaned_input_columns:
            print("⚠️ Input columns must contain at least one valid column name ‼️")
            return None

        if len(cleaned_input_columns) != len(set(cleaned_input_columns)):
            print("⚠️ Input columns contain duplicates ‼️")
            return None

        # ---------- Check input columns existed ----------
        invalid_columns = [
            col for col in cleaned_input_columns if col not in self.source_data.columns
        ]
        if invalid_columns:
            print(f"⚠️ Invalid input columns: {invalid_columns} ‼️")
            return None

        # ---------- Record to instance ----------
        self.input_columns = cleaned_input_columns
        print(f"🔥 Input columns set successfully: {self.input_columns}\n{'-'*100}")

        return self.input_columns

    # -------------------- Build X data --------------------
    def build_x_data(self):
        """
        Build the unsupervised-learning X dataset from selected input columns.

        If `input_columns` has already been set, this method builds `X` using those
        columns only. If no input columns have been selected yet, all columns in the
        source dataset are used by default.

        A copied DataFrame is stored in `self.X` to avoid accidental linkage to the
        original source dataset during later processing.

        Returns
        -------
        pandas.DataFrame or None
            Built X dataset if successful, otherwise `None`.

        Notes
        -----
        This method does not perform feature engineering, encoding, scaling, or missing-
        value handling. It only selects columns from the source dataset and stores the
        result as `X`.
        """
        if not self._validation():
            return None

        if self.input_columns is None:
            self.input_columns = list(self.source_data.columns)

        if not self.input_columns:
            print("⚠️ No input columns available to build X ‼️")
            return None

        # ---------- Get X by input columns from sourced data ----------
        self.X = self.source_data[self.input_columns].copy()

        print("🔥 X built successfully.")
        return self.X


# =================================================
