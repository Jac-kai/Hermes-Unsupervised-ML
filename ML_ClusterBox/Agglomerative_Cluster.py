# -------------------- Import Modules --------------------
import os
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

from Hermes.ML_UnSup_MissionBox.ClusteringBaseConfig_Missioner import (
    BaseClustering_Missioner,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLOT_DIR = os.path.join(BASE_DIR, "ML_Report/Clusterer_Plot")
os.makedirs(PLOT_DIR, exist_ok=True)


# -------------------- Agglomerative Cluster --------------------
class Agglomerative_Cluster(BaseClustering_Missioner):
    """
    Agglomerative hierarchical clustering mission layer.

    This class extends ``BaseClustering_Missioner`` and provides the
    AgglomerativeClustering-specific execution and diagnostic workflow used in
    the project's unsupervised clustering module.

    Supported Responsibilities
    --------------------------
    - Build an ``AgglomerativeClustering`` estimator with user-specified
    hierarchical clustering parameters
    - Fit the shared preprocessing + clustering pipeline through
    ``fit_cluster_pipeline()``
    - Evaluate fitted agglomerative clustering results using common internal
    clustering metrics
    - Return a compact clustering summary dictionary
    - Visualize hierarchical merge structure through a dendrogram built from the
    transformed feature space

    Design Notes
    ------------
    This class specializes the shared clustering mission layer for
    agglomerative hierarchical clustering behavior.

    Compared with KMeans and DBSCAN, AgglomerativeClustering has a different
    structural interpretation:

    - it does not optimize cluster centers like KMeans
    - it does not identify density-connected regions like DBSCAN
    - it progressively merges samples or groups into larger clusters according to
    a chosen linkage rule

    Because of this hierarchical structure, dendrogram visualization is a natural
    algorithm-specific diagnostic tool and is included in this class.

    Notes
    -----
    - This implementation uses sklearn's ``AgglomerativeClustering`` for model
    fitting.
    - Dendrogram plotting is handled separately using
    ``scipy.cluster.hierarchy`` because scipy provides a direct and standard
    interface for hierarchical tree visualization.
    - For the most interpretable comparison between fitted model behavior and
    dendrogram structure, the dendrogram linkage method should usually match
    the linkage configuration used during model fitting.
    """

    def agglomerative_model_engine(
        self,
        n_clusters: int = 3,
        linkage: str = "ward",
        metric: str = "euclidean",
        scaler_type: str = "standard",
        categorical_cols=None,
        numeric_cols=None,
        cat_encoder: str = "ohe",
        extra_steps: Optional[List[Tuple[str, Any]]] = None,
    ):
        """
        Build, fit, evaluate, and summarize an AgglomerativeClustering pipeline.

        This method is the main agglomerative hierarchical clustering training
        entrypoint for the current mission layer. It creates an
        ``AgglomerativeClustering`` estimator, fits it together with the shared
        preprocessing pipeline, evaluates the clustering result using common
        internal metrics, and returns a compact summary dictionary.

        Parameters
        ----------
        n_clusters : int, default=3
            Number of clusters to form.

            This determines the final number of clusters produced by the fitted
            agglomerative model when using the standard fixed-cluster-count mode.

        linkage : str, default="ward"
            Linkage strategy used by ``AgglomerativeClustering`` to decide how
            clusters are merged.

            Common options include:
            - ``"ward"``
            - ``"single"``
            - ``"complete"``
            - ``"average"``

            Notes
            -----
            Different linkage rules can produce noticeably different hierarchical
            structures and final clustering outputs.

        metric : str, default="euclidean"
            Distance metric used by the agglomerative clustering estimator.

            Common examples include:
            - ``"euclidean"``
            - ``"manhattan"``
            - ``"cosine"``
            - ``"precomputed"``

            Notes
            -----
            When ``linkage="ward"``, the metric must remain compatible with ward
            linkage behavior, which is typically Euclidean-based.

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
            optional scaling, but before the final agglomerative estimator.

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
            - cluster-center availability
            - inertia availability

        Side Effects
        ------------
        Updates clustering state stored on the current object, including:
        - ``self.input_model_type``
        - ``self.model_pipeline``
        - ``self.X_processed``
        - ``self.feature_names``
        - ``self.labels_``
        - ``self.cluster_centers_`` (typically unavailable for agglomerative clustering)
        - ``self.inertia_`` (typically unavailable for agglomerative clustering)
        - ``self.evaluation_result``

        Notes
        -----
        This method performs a full agglomerative clustering workflow in one call:

        1. Build AgglomerativeClustering estimator
        2. Fit preprocessing + clustering pipeline
        3. Evaluate clustering result
        4. Return clustering summary

        Because AgglomerativeClustering generally does not expose fitted
        ``cluster_centers_`` or ``inertia_`` like KMeans, those shared clustering
        outputs usually remain unavailable in this workflow.
        """
        self.input_model_type = "AgglomerativeClustering"

        # ---------- AgglomerativeClustering setting ----------
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric,
        )

        # ---------- AgglomerativeClustering fit ----------
        self.fit_cluster_pipeline(
            base_model=model,
            scaler_type=scaler_type,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            cat_encoder=cat_encoder,
            extra_steps=extra_steps,
        )

        # ---------- AgglomerativeClustering evaluation ----------
        self.clustering_evaluation_engine()

        return self.cluster_summary_engine()

    # -------------------- Agglomerative dendrogram plot --------------------
    def dendrogram_plot_engine(
        self,
        method: str = "ward",
        truncate_mode: Optional[str] = None,
        p: int = 12,
        orientation: str = "top",
        show_leaf_counts: bool = True,
        distance_sort: str | bool = False,
        labels: Optional[list] = None,
        figsize: tuple = (10, 6),
        save_fig: bool = False,
        fig_name: str = "agglomerative_dendrogram.png",
    ):
        """
        Plot a hierarchical clustering dendrogram using the transformed feature space.

        Intended Purpose
        ----------------
        This method visualizes how samples are progressively merged in hierarchical
        clustering. It is especially useful for:

        - inspecting hierarchical merge structure
        - understanding relative merge distances
        - visually exploring possible cluster-cut locations
        - comparing different linkage strategies
        - supporting interpretation alongside fitted AgglomerativeClustering output

        Unlike flat clustering result plots, a dendrogram displays the full merge
        history rather than only the final cluster assignments.

        Parameters
        ----------
        method : str, default="ward"
            Linkage method used by ``scipy.cluster.hierarchy.linkage()``.

            Common options include:
            - ``"ward"``
            - ``"single"``
            - ``"complete"``
            - ``"average"``

            Notes
            -----
            For the most interpretable comparison with the fitted sklearn
            ``AgglomerativeClustering`` model, this method should usually match the
            linkage setting used during model fitting when possible.

        truncate_mode : str or None, default=None
            Dendrogram truncation mode passed to ``scipy.cluster.hierarchy.dendrogram()``.

            Common options include:
            - ``None`` : show the full dendrogram
            - ``"lastp"`` : show only the last ``p`` merged clusters
            - ``"level"`` : show only ``p`` hierarchy levels

            Truncation is especially helpful when the number of samples is large and
            the full dendrogram becomes visually crowded.

        p : int, default=12
            Truncation parameter used when ``truncate_mode`` is not ``None``.

            The meaning of ``p`` depends on the chosen truncation mode, for example:
            - with ``"lastp"``, it controls how many final merged groups are displayed
            - with ``"level"``, it controls how many hierarchical levels are shown

        orientation : str, default="top"
            Orientation of the dendrogram.

            Common options include:
            - ``"top"``
            - ``"bottom"``
            - ``"left"``
            - ``"right"``

            This parameter affects only visual layout and does not change the
            underlying hierarchical structure.

        show_leaf_counts : bool, default=True
            Whether to display the number of original observations contained in
            non-singleton leaf nodes.

            This is particularly useful when a truncated dendrogram is shown.

        distance_sort : str or bool, default=False
            Controls whether child branches are sorted by linkage distance.

            Common values include:
            - ``False`` : no distance-based sorting
            - ``"ascending"``
            - ``"descending"``

            This affects only branch display order and does not alter the hierarchy
            itself.

        labels : list or None, default=None
            Optional custom labels for dendrogram leaf nodes.

            If provided, the length should align with the number of samples used in
            the transformed feature space. This can be helpful when using meaningful
            row identifiers instead of default numeric sample positions.

        figsize : tuple, default=(10, 6)
            Figure size.

        save_fig : bool, default=False
            Whether to save the figure.

        fig_name : str, default="agglomerative_dendrogram.png"
            Output file name when ``save_fig=True``.

        Returns
        -------
        Any
            The linkage matrix produced by ``scipy.cluster.hierarchy.linkage()``.

            This matrix encodes the full hierarchical merge history and can be reused
            for further custom analysis outside the plotting method.

        Raises
        ------
        ValueError
            If transformed feature matrix is unavailable.

        ValueError
            If transformed feature matrix is not 2D.

        ValueError
            If fewer than 2 samples are available.

        Implementation Notes
        --------------------
        The dendrogram is constructed from ``self.X_processed`` rather than directly
        from the raw input data so that the hierarchical structure is computed in the
        same transformed feature space used by the clustering workflow.

        This means the plotted hierarchy reflects any preprocessing already applied,
        such as:
        - imputation
        - categorical encoding
        - scaling
        - optional dimensionality reduction steps inserted through ``extra_steps``

        The method first builds a scipy linkage matrix from the transformed data and
        then passes that matrix into ``dendrogram()`` for visualization.

        Notes
        -----
        - This plot is primarily a hierarchical structure visualization and does not
        rely on the final cluster labels stored in ``self.labels_``.
        - The y-axis represents merge distance, so branches that join at larger
        heights indicate merges between groups that were farther apart.
        - A dendrogram can be used informally to inspect a possible number of clusters
        by imagining a horizontal cut across the tree.
        - For very large datasets, truncation is usually recommended to keep the plot
        interpretable.
        """
        if self.X_processed is None:
            raise ValueError(
                "⚠️  X_processed not found. Run fit_cluster_pipeline() first ‼️"
            )

        X_plot = np.asarray(self.X_processed)

        if X_plot.ndim != 2:
            raise ValueError("⚠️  X_processed must be a 2D array for plotting ‼️")

        if X_plot.shape[0] < 2:
            raise ValueError("⚠️  At least 2 samples are required for dendrogram ‼️")

        # ---------- Build linkage matrix from transformed feature space ----------
        Z = linkage(X_plot, method=method)

        # ---------- Plot dendrogram ----------
        plt.figure(figsize=figsize)
        dendrogram(
            Z,
            truncate_mode=truncate_mode,
            p=p if truncate_mode is not None else 0,
            orientation=orientation,
            show_leaf_counts=show_leaf_counts,
            distance_sort=distance_sort,
            labels=labels,
        )
        plt.title(f"Agglomerative Dendrogram ({method})")
        plt.xlabel("Sample Index / Cluster")
        plt.ylabel("Merge Distance")
        plt.tight_layout()

        if save_fig:
            save_path = os.path.join(PLOT_DIR, fig_name)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()
        plt.close()

        return Z


# =================================================
