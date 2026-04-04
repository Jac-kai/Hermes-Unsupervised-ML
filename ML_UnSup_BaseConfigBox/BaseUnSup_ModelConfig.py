# -------------------- Import Modules --------------------
from __future__ import annotations

import re
from abc import ABC
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


# -------------------- Base Model Configure --------------------
class BaseClusterConfig(ABC):
    """
    BaseClusterConfig
    =================

    Abstract base class for unsupervised clustering pipeline management.

    This class provides shared infrastructure for clustering-oriented workflows
    that operate on feature matrix ``X`` only, without target labels ``Y``.
    It is intended to support clustering estimators such as:

    - ``KMeans``
    - ``DBSCAN``
    - ``AgglomerativeClustering``

    Core Responsibilities
    ---------------------
    - Store cleaned clustering input data.
    - Build a preprocessing ``ColumnTransformer`` for numeric and categorical features.
    - Optionally attach a scaler before the final clustering estimator.
    - Build and fit a unified sklearn ``Pipeline``.
    - Cache transformed feature matrix before the final clustering step.
    - Extract transformed feature names after preprocessing.
    - Capture common fitted clustering outputs such as:
    - ``labels_``
    - ``cluster_centers_``
    - ``inertia_``
    - Save and load fitted clustering pipelines.
    - Provide utility access to the final fitted clusterer.

    Design Scope
    ------------
    This base class focuses only on shared clustering infrastructure and pipeline
    construction.

    It intentionally does not perform:
    - clustering quality evaluation
    - clustering summary generation
    - cluster preview/report formatting
    - algorithm-specific diagnostics such as:
    - best-k search for KMeans
    - noise analysis for DBSCAN
    - dendrogram logic for AgglomerativeClustering

    These higher-level task behaviors are expected to be implemented in the
    mission layer.

    Typical Workflow
    ----------------
    A subclass or mission layer typically follows this sequence:

    1. Build a clustering estimator.
    2. Call ``fit_cluster_pipeline(...)``.
    3. Use captured outputs such as ``self.labels_``.
    4. Perform evaluation / summary / visualization in the mission layer.

    Attributes
    ----------
    cleaned_X_data : pd.DataFrame
        Cleaned feature matrix used for clustering.

    model_pipeline : sklearn.pipeline.Pipeline or None
        Fitted clustering pipeline.

    feature_names : list[str] or None
        Best-effort extracted transformed feature names after preprocessing.

    X_processed : np.ndarray or None
        Transformed feature matrix before the final clusterer step.

    _numeric_cols : list[str] or None
        Cached numeric column names used for feature-name fallback extraction.

    _categorical_cols : list[str] or None
        Cached categorical column names used for feature-name fallback extraction.

    input_model_type : str or None
        Stored model type label for later reporting.

    input_use_scaler : str or None
        Stored scaler selection used during fitting.

    labels_ : np.ndarray or None
        Cluster labels captured from the fitted estimator when available.

    cluster_centers_ : np.ndarray or None
        Cluster centers captured from the fitted estimator when available.

    inertia_ : float or None
        Inertia captured from the fitted estimator when available.

    Notes
    -----
    - This class assumes clustering is performed directly on the available input
    dataset rather than through supervised train/test target logic.
    - Not every clustering estimator exposes the same fitted attributes.
    Missing attributes are safely preserved as ``None``.
    - Feature-name extraction is best-effort and may fall back to cached column
    names when exact transformed names are unavailable.
    """

    step_name: str = "clusterer"

    # -------------------- Initialization --------------------
    def __init__(self, cleaned_X_data: pd.DataFrame):
        """
        Initialize the clustering base manager with cleaned feature data.

        Parameters
        ----------
        cleaned_X_data : pd.DataFrame
            Cleaned feature matrix used for clustering.

        Raises
        ------
        TypeError
            If ``cleaned_X_data`` is not a pandas DataFrame.

        ValueError
            If ``cleaned_X_data`` is empty.

        Side Effects
        ------------
        Initializes internal state used throughout the clustering workflow, including:
        - fitted pipeline storage
        - transformed feature cache
        - feature name cache
        - preprocessing column caches
        - run metadata
        - clustering output buffers

        Notes
        -----
        All clustering result attributes are initialized to ``None`` because no model
        has been fitted at object creation time.
        """
        if not isinstance(cleaned_X_data, pd.DataFrame):
            raise TypeError("⚠️  cleaned_X_data must be a pandas DataFrame ‼️")

        if cleaned_X_data.empty:
            raise ValueError("⚠️  cleaned_X_data is empty ‼️")

        self.cleaned_X_data = cleaned_X_data.copy()  # Copy original data

        # ---------- Record model preprocesses and pipeline ----------
        self.model_pipeline = None
        self.feature_names = None
        self.X_processed = None

        # ---------- Record Numeric- and non-numeric-type columns ----------
        self._numeric_cols = None
        self._categorical_cols = None

        # ---------- Record metadata ----------
        self.input_model_type = None
        self.input_use_scaler = None

        # ---------- Clustering outputs ----------
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None

    # ---------- basic identity helpers ----------
    @property
    def task(self) -> str:
        """
        Task type of this manager.

        Returns
        -------
        str
            Always returns ``"clustering"``.

        Notes
        -----
        This property is provided for consistent task identification across the
        project's model/mission infrastructure.
        """
        return "clustering"

    # ---------- Helper: build scalers ----------
    def _build_scaler(self, scaler_type: str = "standard"):
        """
        Build a scaler instance based on user selection.

        Parameters
        ----------
        scaler_type : str, default="standard"
            Name of the scaler to construct.

            Supported values:
            - ``"standard"`` or ``"std"`` -> ``StandardScaler``
            - ``"minmax"`` or ``"min_max"`` -> ``MinMaxScaler``
            - ``"robust"`` or ``"rbst"`` -> ``RobustScaler``
            - ``"none"``, ``"no"``, or ``"off"`` -> no scaler (returns ``None``)

        Returns
        -------
        object or None
            Instantiated sklearn scaler object if scaling is requested,
            otherwise ``None``.

        Raises
        ------
        ValueError
            If ``scaler_type`` is not one of the supported options.

        Notes
        -----
        Scaling is often important for clustering algorithms that rely on distance
        or geometry in feature space, such as:
        - ``KMeans``
        - ``DBSCAN``
        - ``AgglomerativeClustering``
        """
        scaler_type = scaler_type.lower().strip()

        if scaler_type in ["none", "no", "off"]:
            return None
        if scaler_type in ["standard", "std"]:
            return StandardScaler()
        if scaler_type in ["minmax", "min_max"]:
            return MinMaxScaler()
        if scaler_type in ["robust", "rbst"]:
            return RobustScaler()

        raise ValueError(
            "⚠️  scaler_type must be 'standard', 'minmax', 'robust', or 'none' ‼️"
        )

    # -------------------- Build preprocess --------------------
    def build_preprocessor(
        self,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        cat_encoder: str = "ohe",
    ) -> ColumnTransformer:
        """
        Build a preprocessing ``ColumnTransformer`` for clustering features.

        Parameters
        ----------
        categorical_cols : list[str] or None, default=None
            Column names to treat as categorical features.
            If ``None``, categorical columns are inferred from dtypes:
            - ``object``
            - ``category``
            - ``bool``

        numeric_cols : list[str] or None, default=None
            Column names to treat as numeric features.
            If ``None``, numeric columns are inferred as all remaining columns
            not included in ``categorical_cols``.

        cat_encoder : str, default="ohe"
            Encoding strategy for categorical features.

            Supported values:
            - ``"ohe"``, ``"onehot"``, ``"one_hot"``:
            use ``OneHotEncoder(handle_unknown="ignore")``
            - ``"ordinal"``, ``"ord"``, ``"ord_label"``:
            use ``OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)``

        Returns
        -------
        sklearn.compose.ColumnTransformer
            Preprocessing transformer used as the first step of the clustering pipeline.

        Raises
        ------
        ValueError
            If ``cat_encoder`` is not supported.

        Side Effects
        ------------
        Stores inferred or user-specified preprocessing column groups into:
        - ``self._numeric_cols``
        - ``self._categorical_cols``

        Notes
        -----
        Preprocessing policy used by this method:

        Numeric features
            - ``SimpleImputer(strategy="median")``

        Categorical features
            - ``SimpleImputer(strategy="most_frequent")``
            - followed by either One-Hot or Ordinal encoding

        This method builds the transformer configuration only. Actual fitting and
        data transformation occur later when the pipeline is fitted.
        """
        df = self.cleaned_X_data

        if categorical_cols is None:
            categorical_cols = df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()
        if numeric_cols is None:
            numeric_cols = [c for c in df.columns if c not in categorical_cols]

        self._numeric_cols = numeric_cols
        self._categorical_cols = categorical_cols

        # ---------- Numeric columns ----------
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )

        # ---------- Encoding non-numeric columns ----------
        cat_encoder = cat_encoder.lower().strip()
        if cat_encoder in ["ohe", "onehot", "one_hot"]:
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
        elif cat_encoder in ["ordinal", "ord", "ord_label"]:
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ord",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                    ),
                ]
            )
        else:
            raise ValueError("⚠️  cat_encoder must be 'ohe' or 'ordinal' ‼️")

        # ---------- Merge the preprocessed columns ----------
        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    # -------------------- Shared fit of clusterer --------------------
    def fit_cluster_pipeline(
        self,
        base_model: Any,
        scaler_type: str = "standard",
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        cat_encoder: str = "ohe",
        extra_steps: Optional[List[Tuple[str, Any]]] = None,
        cache_processed_X: bool = True,
    ) -> Pipeline:
        """
        Build and fit a shared clustering pipeline.

        Parameters
        ----------
        base_model : Any
            sklearn-style clustering estimator instance to place at the end of the
            pipeline, for example:
            - ``KMeans(...)``
            - ``DBSCAN(...)``
            - ``AgglomerativeClustering(...)``

        scaler_type : str, default="standard"
            Scaler selection passed to ``_build_scaler()``.

        categorical_cols : list[str] or None, default=None
            Explicit categorical column names to forward into ``build_preprocessor()``.

        numeric_cols : list[str] or None, default=None
            Explicit numeric column names to forward into ``build_preprocessor()``.

        cat_encoder : str, default="ohe"
            Categorical encoding strategy passed to ``build_preprocessor()``.

        extra_steps : list[tuple[str, Any]] or None, default=None
            Optional additional sklearn pipeline steps inserted after preprocessing
            and optional scaling, but before the final clustering estimator.

        cache_processed_X : bool, default=True
            Whether to cache the transformed feature matrix before the final clusterer
            into ``self.X_processed``.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Fitted clustering pipeline.

        Side Effects
        ------------
        Sets or updates:
        - ``self.model_pipeline``
        - ``self.X_processed``
        - ``self.feature_names``
        - ``self.input_use_scaler``
        - ``self.labels_``
        - ``self.cluster_centers_``
        - ``self.inertia_``

        Notes
        -----
        Typical resulting pipeline structure:

        ``Pipeline([
            ("preprocess", ...),
            ("scaler", ...),        # optional
            (...extra_steps...),    # optional
            ("clusterer", base_model)
        ])``

        This method is the main shared fit routine for clustering estimators in this
        base class.
        """
        # ---------- Preprocess ----------
        preprocess = self.build_preprocessor(
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            cat_encoder=cat_encoder,
        )

        # ---------- Step appneded ----------
        steps: List[Tuple[str, Any]] = [("preprocess", preprocess)]

        # ---------- Scaler ----------
        scaler = self._build_scaler(scaler_type)
        self.input_use_scaler = scaler_type
        if scaler is not None:
            steps.append(("scaler", scaler))

        if extra_steps:  # Other steps
            steps.extend(extra_steps)

        # ---------- Clusterer ----------
        steps.append((self.step_name, base_model))

        # ---------- Pipeline integration ----------
        pipe = Pipeline(steps=steps)
        pipe.fit(self.cleaned_X_data)

        self.model_pipeline = pipe

        # ---------- Cache transformed X before final clusterer ----------
        if cache_processed_X:
            self.X_processed = self._transform_before_clusterer(self.cleaned_X_data)

        # ---------- Extract feature names ----------
        self._extract_feature_names()

        # ---------- Cluster outputs ----------
        self._capture_cluster_outputs()

        return self.model_pipeline

    # -------------------- Capture cluster outputs --------------------
    def _capture_cluster_outputs(self):
        """
        Capture common fitted outputs from the final clustering estimator.

        Captured Attributes
        -------------------
        This helper updates the following instance attributes when available:
        - ``self.labels_``
        - ``self.cluster_centers_``
        - ``self.inertia_``

        Behavior
        --------
        1. Reset previously stored clustering outputs to ``None``.
        2. Retrieve the final clusterer from the fitted pipeline.
        3. Safely inspect supported fitted attributes on that estimator.
        4. Store available values into the current object.

        Notes
        -----
        Not all clustering estimators expose the same fitted attributes.

        Typical examples:
        - ``KMeans``:
        - ``labels_``
        - ``cluster_centers_``
        - ``inertia_``

        - ``DBSCAN``:
        - ``labels_``

        - ``AgglomerativeClustering``:
        - ``labels_``

        Resetting outputs before recapturing is important so that attributes from a
        previously fitted model do not remain incorrectly attached when the new model
        does not provide them.
        """
        # ---------- Reset outputs before recapturing ----------
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None

        # ---------- Guard: pipeline must exist ----------
        if self.model_pipeline is None:
            return

        # ---------- Get final clusterer from pipeline ----------
        clusterer = self.model_pipeline.named_steps.get(self.step_name, None)
        if clusterer is None:
            return

        # ---------- labels ----------
        if hasattr(clusterer, "labels_"):
            labels = getattr(clusterer, "labels_")
            if labels is not None:
                self.labels_ = np.asarray(labels)

        # ---------- Clustercenters ----------
        if hasattr(clusterer, "cluster_centers_"):
            centers = getattr(clusterer, "cluster_centers_")
            if centers is not None:
                self.cluster_centers_ = np.asarray(centers)

        # ---------- Inertia ----------
        if hasattr(clusterer, "inertia_"):
            inertia = getattr(clusterer, "inertia_")
            if inertia is not None:
                self.inertia_ = float(inertia)

    # ---------- Helper: transform before clustering ----------
    def _transform_before_clusterer(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Transform input data using all fitted pipeline steps before the final clusterer.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix to transform.

        Returns
        -------
        np.ndarray or None
            Transformed feature matrix immediately before the final clustering
            estimator, or ``None`` if no fitted pipeline is available.

        Notes
        -----
        This helper applies all pipeline steps up to, but not including, the final
        clusterer step. It is useful for:

        - clustering metric computation
        - plotting in transformed feature space
        - manual inspection of preprocessing results
        - downstream mission-layer analysis

        If the intermediate transformed data is sparse, it is converted to a dense
        NumPy array before returning.
        """
        if self.model_pipeline is None:
            return None

        Xt = X.copy()

        # ---------- Record pipeline's steps ----------
        for step_name, step_obj in self.model_pipeline.steps:
            if step_name == self.step_name:  # Stop at estimator
                break
            Xt = step_obj.transform(Xt)

        # ---------- Transform sparse matric to dense array ----------
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()

        return np.asarray(Xt)  # Numpy array format

    # ---------- Feature names ----------
    def _extract_feature_names(self):
        """
        Extract transformed feature names from the fitted preprocessing step.

        Strategy
        --------
        This method uses a two-level recovery strategy:

        1. Preferred path
        Use ``ColumnTransformer.get_feature_names_out()`` when available and reject
        obviously generic placeholder names such as ``x0``, ``x1``, etc.

        2. Fallback path
        Rebuild transformed names using cached numeric/categorical column groups.
        If a fitted ``OneHotEncoder`` is detected in the categorical branch,
        expanded OHE feature names are used when possible.

        Side Effects
        ------------
        Sets:
        - ``self.feature_names``

        Notes
        -----
        This is a best-effort extraction method. Exact transformed feature naming may
        not always be available, especially when custom transformers or more complex
        pipeline structures are introduced.
        """
        # ---------- Pipeline check ----------
        if self.model_pipeline is None:
            self.feature_names = None
            return

        # ---------- Proprocess check ----------
        pre = self.model_pipeline.named_steps.get("preprocess", None)
        if pre is None:
            self.feature_names = None
            return

        # ---------- Main extract feature name method ----------
        try:
            if hasattr(pre, "get_feature_names_out"):
                names = pre.get_feature_names_out()
                if names is not None:
                    names = list(names)

                    if all(
                        re.match(r"^[a-zA-Z]\d+$", str(n)) for n in names
                    ):  # Reject generic names like x0, x1...
                        raise ValueError("generic feature names")

                    self.feature_names = names
                    return

            raise ValueError("⚠️  No usable feature names ‼️")

        # ---------- Backup plan 1 ----------
        except Exception:
            num_cols = getattr(self, "_numeric_cols", None)
            cat_cols = getattr(self, "_categorical_cols", None)

            if num_cols is None or cat_cols is None:
                self.feature_names = None
                return

            names_out = list(num_cols)

            try:
                cat_pipe = pre.named_transformers_.get("cat", None)
                if (
                    cat_pipe is not None
                    and hasattr(cat_pipe, "named_steps")
                    and ("ohe" in cat_pipe.named_steps)
                ):
                    ohe = cat_pipe.named_steps["ohe"]
                    names_out.extend(ohe.get_feature_names_out(cat_cols).tolist())
                else:
                    names_out.extend(list(cat_cols))

                self.feature_names = names_out

            # ---------- Backup plan 2 ----------
            # Using self._numeric_cols and self._categorical_cols directly
            except Exception:
                self.feature_names = list(num_cols) + list(cat_cols)

    # ---------- Save model ----------
    def save_model_engine(self, file_path: str) -> str:
        """
        Save the fitted clustering pipeline to disk.

        Parameters
        ----------
        file_path : str
            Output file path, typically ending with ``.joblib``.

        Returns
        -------
        str
            The saved file path.

        Raises
        ------
        ValueError
            If no fitted pipeline is available.

        Notes
        -----
        Only the fitted pipeline is persisted. Other cached runtime attributes can be
        reconstructed later from the loaded pipeline when needed.
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️  No fitted pipeline to save ‼️")

        dump(self.model_pipeline, file_path)
        return file_path

    # ---------- Load model ----------
    def load_model_engine(self, file_path: str):
        """
        Load a fitted clustering pipeline from disk and rebuild core runtime state.

        This method restores the fitted sklearn pipeline from a saved ``joblib`` file,
        rebuilds the transformed feature matrix before the final clusterer, extracts
        transformed feature names, and recaptures common fitted clustering outputs such
        as labels, cluster centers, and inertia.

        If the current object also provides a mission-layer runtime rebuild helper,
        that helper is called so derived summaries and previews can be reconstructed
        after loading.

        Parameters
        ----------
        file_path : str
            Path to the saved ``.joblib`` clustering pipeline file.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Loaded fitted clustering pipeline.

        Raises
        ------
        ValueError
            If ``cleaned_X_data`` is unavailable when rebuilding transformed data.

        Notes
        -----
        This method restores the fitted pipeline itself. Mission-layer cached results
        such as summary dictionaries, profile tables, and preview tables are expected
        to be rebuilt after loading rather than stored as the primary saved object.
        """
        loaded_pipeline = load(file_path)
        self.model_pipeline = loaded_pipeline

        if self.cleaned_X_data is None:
            raise ValueError(
                "⚠️ cleaned_X_data is required to rebuild transformed clustering state ‼️"
            )

        # ---------- Rebuild transformed X before final clusterer ----------
        self.X_processed = self._transform_before_clusterer(self.cleaned_X_data)

        # ---------- Re-extract transformed feature names ----------
        self._extract_feature_names()

        # ---------- Re-capture fitted cluster outputs ----------
        self._capture_cluster_outputs()

        # ---------- Rebuild mission-layer runtime state when supported ----------
        if hasattr(self, "_rebuild_cluster_runtime_state"):
            self._rebuild_cluster_runtime_state()

        return self.model_pipeline

    # -------------------- Utility --------------------
    def clone_clusterer_engine(self):
        """
        Clone the final clustering estimator from the fitted pipeline.

        Returns
        -------
        Any
            A cloned sklearn-style clustering estimator with the same parameters as the
            fitted final clusterer, but without fitted state.

        Raises
        ------
        ValueError
            If no fitted pipeline exists.
        ValueError
            If the final clusterer step is missing from the pipeline.

        Notes
        -----
        This uses ``sklearn.base.clone()``, which copies estimator configuration but
        does not preserve learned fitted attributes such as:
        - ``labels_``
        - ``cluster_centers_``
        - ``inertia_``
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️  No fitted pipeline found ‼️")

        clusterer = self.model_pipeline.named_steps.get(self.step_name, None)
        if clusterer is None:
            raise ValueError("⚠️  No clusterer step found in pipeline ‼️")

        return clone(clusterer)


# =================================================
