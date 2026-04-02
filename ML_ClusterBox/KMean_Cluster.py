# -------------------- Import Modules --------------------
import os
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

from Hermes.ML_UnSup_MissionBox.ClusteringBaseConfig_Missioner import (
    BaseClustering_Missioner,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLOT_DIR = os.path.join(BASE_DIR, "ML_Report/Cluster_Plot")
os.makedirs(PLOT_DIR, exist_ok=True)


# -------------------- KMeans Cluster --------------------
class KMeans_Cluster(BaseClustering_Missioner):
    """
    KMeans clustering mission layer.

    This class extends ``BaseClustering_Missioner`` and provides the
    KMeans-specific training and diagnostic workflow used in the project's
    unsupervised clustering module.

    Supported Responsibilities
    --------------------------
    - Build a ``KMeans`` estimator with user-specified clustering hyperparameters
    - Fit the shared preprocessing + clustering pipeline through
    ``fit_cluster_pipeline()``
    - Evaluate fitted KMeans clustering results using common internal metrics
    - Return a compact clustering summary dictionary
    - Perform elbow-method analysis by fitting multiple KMeans models across
    different ``k`` values
    - Plot and optionally save the KMeans elbow curve

    Design Notes
    ------------
    This class acts as the KMeans-specific execution layer on top of the shared
    clustering base/mission infrastructure.

    Compared with the base clustering mission layer, this class focuses on logic
    that is specific to KMeans, especially:

    - KMeans estimator construction
    - inertia-based elbow analysis
    - repeated refitting across multiple cluster counts

    The elbow-method helper is intentionally implemented here rather than in the
    base clustering class because ``inertia_`` is a KMeans-specific fitted
    attribute and is not available for algorithms such as DBSCAN or
    AgglomerativeClustering.
    """

    def kmeans_model_engine(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
        init="k-means++",
        n_init: int | str = "auto",
        scaler_type: str = "standard",
        categorical_cols=None,
        numeric_cols=None,
        cat_encoder: str = "ohe",
        extra_steps: Optional[List[Tuple[str, Any]]] = None,
    ):
        """
        Build, fit, evaluate, and summarize a KMeans clustering pipeline.

        This method is the main KMeans training entrypoint for the current mission
        layer. It creates a ``KMeans`` estimator, fits it together with the shared
        preprocessing pipeline, evaluates the clustering result using common internal
        metrics, and returns a compact summary dictionary.

        Parameters
        ----------
        n_clusters : int, default=3
            Number of clusters to form.

            This is the KMeans ``k`` value that controls how many cluster centers
            the estimator will learn.

        random_state : int, default=42
            Random seed used by KMeans centroid initialization.

            Supplying a fixed value improves reproducibility across repeated runs.

        init : str or array-like, default="k-means++"
            Initialization method used by KMeans to choose starting cluster centers.

            Common options include:
            - ``"k-means++"``
            - ``"random"``

            In advanced usage, an explicit array of initial cluster centers may
            also be provided if supported by the underlying sklearn estimator.

            Notes
            -----
            Initialization strategy can affect convergence behavior and final
            clustering quality, especially when the data contains overlapping
            groups or less stable cluster structure.

        n_init : int or str, default="auto"
            Number of KMeans initializations.

            This value is forwarded directly to ``sklearn.cluster.KMeans`` and
            controls how many centroid initializations are attempted before the
            best solution is retained.

        scaler_type : str, default="standard"
            Scaler option passed to the shared clustering pipeline.

            Common choices depend on the implementation of ``_build_scaler()``
            in the base class, for example:
            - ``"standard"``
            - ``"minmax"``
            - ``"robust"``
            - ``"none"``

        categorical_cols : list[str] or None, default=None
            Explicit categorical feature columns used by the preprocessing pipeline.

            If ``None``, categorical columns are inferred by the base preprocessing
            logic.

        numeric_cols : list[str] or None, default=None
            Explicit numeric feature columns used by the preprocessing pipeline.

            If ``None``, numeric columns are inferred by the base preprocessing
            logic.

        cat_encoder : str, default="ohe"
            Categorical encoding strategy used during preprocessing.

            Typical supported values depend on the base class implementation, such as:
            - ``"ohe"``
            - ``"ordinal"``

        extra_steps : list[tuple[str, Any]] or None, default=None
            Optional extra sklearn pipeline steps inserted after preprocessing and
            optional scaling, but before the final KMeans estimator.

            This is useful for inserting steps such as:
            - PCA
            - custom transformers
            - dimensionality reduction utilities

        Returns
        -------
        Dict[str, Any]
            A compact clustering summary dictionary generated by
            ``cluster_summary_engine()``.

            The returned summary typically includes:
            - model name
            - task type
            - scaler setting
            - sample count
            - feature counts
            - number of clusters found
            - silhouette score
            - Calinski-Harabasz score
            - Davies-Bouldin score
            - inertia
            - cluster-center availability

        Side Effects
        ------------
        Updates clustering state stored on the current object, including:
        - ``self.input_model_type``
        - ``self.model_pipeline``
        - ``self.X_processed``
        - ``self.feature_names``
        - ``self.labels_``
        - ``self.cluster_centers_``
        - ``self.inertia_``
        - ``self.evaluation_result``

        Notes
        -----
        This method performs a full KMeans workflow in one call:

        1. Build KMeans estimator
        2. Fit preprocessing + clustering pipeline
        3. Evaluate clustering result
        4. Return clustering summary

        Because KMeans exposes fitted ``cluster_centers_`` and ``inertia_``,
        these outputs are captured automatically by the shared base configuration
        layer after fitting.
        """
        self.input_model_type = "KMeans"

        # ---------- KMeans model setting ----------
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            init=init,
        )

        # ---------- KMeans fit ----------
        self.fit_cluster_pipeline(
            base_model=model,
            scaler_type=scaler_type,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            cat_encoder=cat_encoder,
            extra_steps=extra_steps,
        )

        # ---------- KMeans evaluation ----------
        self.clustering_evaluation_engine()

        return self.cluster_summary_engine()

    # -------------------- KMeans Elbow plot --------------------
    def elbow_plot_engine(
        self,
        k_range=range(2, 11),
        random_state: int = 42,
        n_init: int | str = "auto",
        scaler_type: str = "standard",
        categorical_cols=None,
        numeric_cols=None,
        cat_encoder: str = "ohe",
        extra_steps: Optional[List[Tuple[str, Any]]] = None,
        figsize: tuple = (8, 5),
        save_fig: bool = False,
        fig_name: str = "kmeans_elbow_plot.png",
    ):
        """
        Run KMeans elbow analysis and plot inertia across multiple cluster counts.

        This method repeatedly fits KMeans models over a candidate range of
        ``k`` values, records the corresponding ``inertia`` from each fitted model,
        builds a tabular summary of the results, and plots the classical elbow curve.

        Intended Purpose
        ----------------
        The elbow method is a KMeans-specific diagnostic technique used to help
        choose a reasonable number of clusters.

        For each candidate ``k``:
        - a new KMeans model is fitted under the same preprocessing conditions
        - the fitted model's ``inertia`` is recorded
        - the final plot shows how inertia changes as ``k`` increases

        A visible bend or flattening in the curve is often used as a heuristic
        reference for selecting an appropriate cluster count.

        Parameters
        ----------
        k_range : iterable, default=range(2, 11)
            Candidate cluster counts to evaluate.

            Each value in ``k_range`` is used as a separate KMeans ``n_clusters``
            setting during repeated fitting.

            Examples:
            - ``range(2, 11)``
            - ``[2, 3, 4, 5, 6]``

        random_state : int, default=42
            Random seed passed to every temporary KMeans model created during
            elbow analysis.

            Keeping this fixed improves reproducibility of the elbow curve.

        n_init : int or str, default="auto"
            Number of KMeans initializations used for each candidate ``k``.

            This value is forwarded directly to ``sklearn.cluster.KMeans``.

        scaler_type : str, default="standard"
            Scaler option forwarded to ``fit_cluster_pipeline()``.

            The elbow curve should ideally be generated under the same preprocessing
            conditions as the final KMeans model so that all inertia values are
            comparable in the same transformed feature space.

        categorical_cols : list[str] or None, default=None
            Explicit categorical columns used in preprocessing.

        numeric_cols : list[str] or None, default=None
            Explicit numeric columns used in preprocessing.

        cat_encoder : str, default="ohe"
            Categorical encoding strategy used during preprocessing.

        extra_steps : list[tuple[str, Any]] or None, default=None
            Optional extra sklearn pipeline steps inserted before the final KMeans
            estimator for every candidate ``k``.

            This allows elbow analysis to remain consistent with the final model
            pipeline, including optional steps such as PCA.

        figsize : tuple, default=(8, 5)
            Figure size for the elbow plot.

        save_fig : bool, default=False
            Whether to save the elbow plot to disk.

        fig_name : str, default="kmeans_elbow_plot.png"
            Output file name when ``save_fig=True``.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the evaluated cluster counts and their associated
            inertia values.

            Columns
            -------
            k : int
                Candidate cluster count used for a temporary KMeans fit.

            inertia : float
                Inertia obtained from the fitted KMeans model for that ``k``.

        Raises
        ------
        ValueError
            If ``k_range`` is empty.

        ValueError
            If any candidate ``k`` is smaller than 1.

        Side Effects
        ------------
        Produces a matplotlib elbow plot and optionally saves it to the KMeans elbow
        plot directory.

        Implementation Detail
        ---------------------
        This method temporarily refits the current object multiple times using
        different KMeans estimators. Because repeated calls to
        ``fit_cluster_pipeline()`` would otherwise overwrite the object's existing
        fitted state, the method first records the current state and restores it
        after elbow analysis completes.

        The restored state includes:
        - fitted pipeline
        - transformed feature cache
        - feature names
        - labels
        - cluster centers
        - inertia
        - evaluation results

        This restoration logic ensures that elbow analysis behaves like a diagnostic
        tool and does not destroy previously fitted clustering results on the
        current object.

        Notes
        -----
        - Elbow analysis is specific to KMeans because it relies on ``inertia``,
        which is not a common fitted attribute across all clustering algorithms.
        - Inertia generally decreases as ``k`` increases, so the curve should be
        interpreted by inspecting the change in slope rather than simply choosing
        the smallest inertia.
        - For meaningful comparison, all candidate ``k`` values should be evaluated
        under the same preprocessing, scaling, and optional transformation settings.
        """
        k_list = list(k_range)

        if not k_list:
            raise ValueError("⚠️  k_range cannot be empty ‼️")

        if any(k < 1 for k in k_list):
            raise ValueError("⚠️  All K values in k_range must be >= 1 ‼️")

        inertia_records = []

        # ---------- Record current fitted state ----------
        original_model_pipeline = self.model_pipeline
        original_X_processed = self.X_processed
        original_feature_names = self.feature_names
        original_labels = self.labels_
        original_cluster_centers = self.cluster_centers_
        original_inertia = self.inertia_
        original_evaluation_result = self.evaluation_result

        try:
            for k in k_list:
                model = KMeans(
                    n_clusters=k,
                    random_state=random_state,
                    n_init=n_init,
                )

                self.fit_cluster_pipeline(
                    base_model=model,
                    scaler_type=scaler_type,
                    categorical_cols=categorical_cols,
                    numeric_cols=numeric_cols,
                    cat_encoder=cat_encoder,
                    extra_steps=extra_steps,
                )

                inertia_records.append(
                    {
                        "k": k,
                        "inertia": self.inertia_,
                    }
                )

        finally:
            # ---------- Restore original fitted state ----------
            self.model_pipeline = original_model_pipeline
            self.X_processed = original_X_processed
            self.feature_names = original_feature_names
            self.labels_ = original_labels
            self.cluster_centers_ = original_cluster_centers
            self.inertia_ = original_inertia
            self.evaluation_result = original_evaluation_result

        elbow_df = pd.DataFrame(inertia_records)

        plt.figure(figsize=figsize)
        plt.plot(elbow_df["k"], elbow_df["inertia"], marker="o")
        plt.title("KMeans Elbow Plot")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.xticks(elbow_df["k"])
        plt.tight_layout()

        if save_fig:
            save_path = os.path.join(PLOT_DIR, fig_name)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()
        plt.close()

        return elbow_df


# =================================================
