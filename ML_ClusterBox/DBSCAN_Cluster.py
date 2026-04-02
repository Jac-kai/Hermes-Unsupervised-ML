# -------------------- Import Modules --------------------
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from Hermes.ML_UnSup_MissionBox.ClusteringBaseConfig_Missioner import (
    BaseClustering_Missioner,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLOT_DIR = os.path.join(BASE_DIR, "ML_Report/Clusterer_Plot")
os.makedirs(PLOT_DIR, exist_ok=True)


# -------------------- DBSCAN Cluster --------------------
class DBSCAN_Cluster(BaseClustering_Missioner):
    """
    DBSCAN clustering mission layer.

    This class extends ``BaseClustering_Missioner`` and provides the
    DBSCAN-specific execution and diagnostic workflow used in the project's
    unsupervised clustering module.

    Supported Responsibilities
    --------------------------
    - Build a ``DBSCAN`` estimator with user-specified density parameters
    - Fit the shared preprocessing + clustering pipeline through
    ``fit_cluster_pipeline()``
    - Evaluate fitted DBSCAN results using common internal clustering metrics
    - Return a compact clustering summary dictionary
    - Provide DBSCAN-specific noise summary utilities
    - Provide a k-distance diagnostic plot to help choose a suitable ``eps`` value

    Design Notes
    ------------
    This class specializes the shared clustering mission layer for
    density-based clustering behavior.

    Compared with KMeans and Agglomerative clustering, DBSCAN has several
    distinct characteristics:

    - cluster count is not explicitly specified in advance
    - noise points may be assigned the label ``-1``
    - cluster quality often depends strongly on ``eps`` and ``min_samples``
    - distance-based diagnostics are especially important during tuning

    Because of these characteristics, this mission layer includes additional
    DBSCAN-oriented helper logic such as noise summarization and k-distance
    plotting.

    Notes
    -----
    - DBSCAN may label some points as noise using ``-1``.
    - When evaluating DBSCAN results, internal metrics are often more meaningful
    when noise points are excluded.
    - The k-distance diagnostic utility is intended as a parameter-selection aid
    and does not replace formal model fitting.
    """

    def dbscan_model_engine(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        scaler_type: str = "standard",
        categorical_cols=None,
        numeric_cols=None,
        cat_encoder: str = "ohe",
        extra_steps: Optional[List[Tuple[str, Any]]] = None,
    ):
        """
        Build, fit, evaluate, and summarize a DBSCAN clustering pipeline.

        This method is the main DBSCAN training entrypoint for the current mission
        layer. It creates a ``DBSCAN`` estimator, fits it together with the shared
        preprocessing pipeline, evaluates the clustering result using common
        internal metrics, and returns a compact summary dictionary.

        Parameters
        ----------
        eps : float, default=0.5
            Maximum neighborhood radius used by DBSCAN.

            Two samples are considered neighbors when their distance is within this
            radius according to the selected metric.

            This parameter strongly affects:
            - cluster density sensitivity
            - number of points labeled as noise
            - number of clusters discovered

        min_samples : int, default=5
            Minimum number of samples required in the neighborhood of a point
            for that point to be treated as a core point.

            Larger values generally make DBSCAN more conservative and may increase
            the number of samples labeled as noise.

        metric : str, default="euclidean"
            Distance metric used by DBSCAN.

            Common examples include:
            - ``"euclidean"``
            - ``"manhattan"``
            - ``"cosine"``
            - ``"precomputed"``

            Notes
            -----
            The interpretation of ``eps`` depends directly on the chosen metric.

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
            optional scaling, but before the final DBSCAN estimator.

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
            - noise count
            - noise ratio
            - silhouette score
            - Calinski-Harabasz score
            - Davies-Bouldin score

        Side Effects
        ------------
        Updates clustering state stored on the current object, including:
        - ``self.input_model_type``
        - ``self.model_pipeline``
        - ``self.X_processed``
        - ``self.feature_names``
        - ``self.labels_``
        - ``self.cluster_centers_`` (typically unavailable for DBSCAN)
        - ``self.inertia_`` (typically unavailable for DBSCAN)
        - ``self.evaluation_result``

        Notes
        -----
        This method performs a full DBSCAN workflow in one call:

        1. Build DBSCAN estimator
        2. Fit preprocessing + clustering pipeline
        3. Evaluate clustering result
        4. Return clustering summary

        Evaluation is performed with ``ignore_noise=True`` so that points labeled
        ``-1`` are excluded from internal metric computation by default. This is
        commonly preferred for DBSCAN because noise points are not part of any
        formed cluster.
        """
        self.input_model_type = "DBSCAN"

        # ---------- DBSCAN setting ----------
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
        )

        # ---------- DBSCAN fit ----------
        self.fit_cluster_pipeline(
            base_model=model,
            scaler_type=scaler_type,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            cat_encoder=cat_encoder,
            extra_steps=extra_steps,
        )

        # ---------- DBSCAN evaluation ----------
        self.clustering_evaluation_engine(ignore_noise=True)

        return self.cluster_summary_engine()

    # -------------------- DBSCAN noise --------------------
    def dbscan_noise_summary_engine(self) -> Dict[str, Any]:
        """
        Summarize DBSCAN noise statistics from fitted cluster labels.

        This helper computes the number and proportion of samples labeled as
        noise by DBSCAN.

        In DBSCAN, noise points are assigned the label ``-1``. This method
        extracts those counts directly from the fitted ``labels_`` stored on
        the current object.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing DBSCAN noise statistics.

            Keys
            ----
            noise_count : int
                Number of samples labeled as noise (``-1``).

            noise_ratio : float
                Ratio of noise samples relative to the total number of samples.

        Raises
        ------
        ValueError
            If cluster labels are unavailable.

        Notes
        -----
        This method is specific to DBSCAN-style clustering behavior because the
        concept of explicit noise labeling is a defining characteristic of DBSCAN.

        A high noise ratio may suggest that:
        - ``eps`` is too small
        - ``min_samples`` is too large
        - the chosen distance metric is too restrictive
        - the transformed feature space is not suitable for the current DBSCAN
        parameter setting
        """
        if self.labels_ is None:
            raise ValueError("⚠️  Cluster labels not found. Run DBSCAN first ‼️")

        labels = np.asarray(self.labels_)
        noise_count = int(np.sum(labels == -1))
        noise_ratio = float(noise_count / len(labels)) if len(labels) > 0 else 0.0

        return {
            "noise_count": noise_count,
            "noise_ratio": noise_ratio,
        }

    # -------------------- DBSCAN k-distance plot --------------------
    def k_distance_plot_engine(
        self,
        k: Optional[int] = None,
        ignore_zero_distance: bool = True,
        figsize: tuple = (8, 5),
        save_fig: bool = False,
        fig_name: str = "dbscan_k_distance_plot.png",
    ) -> pd.DataFrame:
        """
        Plot sorted k-nearest-neighbor distances to help choose DBSCAN ``eps``.

        Intended Purpose
        ----------------
        This diagnostic method computes the distance from each sample to its
        k-th nearest neighbor in the transformed feature space and plots those
        distances in ascending order.

        In common DBSCAN practice, the point where the sorted k-distance curve
        starts rising more sharply is often used as a heuristic reference for
        choosing a reasonable ``eps`` value.

        Parameters
        ----------
        k : int or None, default=None
            Neighbor rank used for distance extraction.

            Behavior
            --------
            - If an integer is provided, that exact k-th nearest-neighbor distance
            is used.
            - If ``None``, the method first tries to use the fitted DBSCAN model's
            ``min_samples`` value when available.
            - If ``None`` and no fitted DBSCAN estimator is available, the method
            defaults to ``5``.

            Recommended Practice
            --------------------
            In DBSCAN workflows, ``k`` is commonly chosen equal to
            ``min_samples``.

        ignore_zero_distance : bool, default=True
            Whether to drop zero distances before plotting.

            Notes
            -----
            Zero distances may appear when:
            - duplicate rows exist
            - multiple samples overlap exactly in transformed feature space

            Keeping them is not incorrect, but removing them can make the
            elbow-like rise easier to inspect visually.

        figsize : tuple, default=(8, 5)
            Figure size.

        save_fig : bool, default=False
            Whether to save the figure into the clustering plot directory.

        fig_name : str, default="dbscan_k_distance_plot.png"
            Output file name when ``save_fig=True``.

        Returns
        -------
        pd.DataFrame
            DataFrame containing sorted sample order and corresponding k-distance
            values.

            Columns
            -------
            sample_rank : int
                Sorted position after ascending distance sort.

            k_distance : float
                Distance from a sample to its k-th nearest neighbor.

        Raises
        ------
        ValueError
            If transformed feature matrix is unavailable.

        ValueError
            If transformed feature matrix is not 2D.

        ValueError
            If fewer than 2 samples are available.

        ValueError
            If ``k`` is smaller than 1.

        ValueError
            If ``k`` is greater than or equal to the number of samples.

        ValueError
            If all distances are removed after zero-distance filtering.

        Implementation Notes
        --------------------
        The computation is performed on ``self.X_processed`` so that the distance
        structure matches the same transformed feature space used by clustering.

        For nearest-neighbor computation, the method queries ``k + 1`` neighbors
        rather than ``k`` because the nearest neighbor of each point is usually
        the point itself with distance 0. Therefore:

        - ``distances[:, 0]`` corresponds to self-distance
        - ``distances[:, k]`` corresponds to the distance to the k-th actual
        neighbor excluding the sample itself

        The resulting distances are then sorted in ascending order to form the
        diagnostic k-distance curve.

        Notes
        -----
        - This method does not require DBSCAN to be fully refit as long as
        ``self.X_processed`` already exists.
        - If a fitted DBSCAN estimator is available, its ``min_samples`` value can
        be used automatically as the default ``k``.
        - This plot is a tuning aid for ``eps`` selection and is not itself a
        clustering result visualization.
        """
        if self.X_processed is None:
            raise ValueError(
                "⚠️  X_processed not found. Run fit_cluster_pipeline() first ‼️"
            )

        X_plot = np.asarray(self.X_processed)

        if X_plot.ndim != 2:
            raise ValueError("⚠️  X_processed must be a 2D array for plotting ‼️")

        n_samples = X_plot.shape[0]
        if n_samples < 2:
            raise ValueError(
                "⚠️  At least 2 samples are required for k-distance plot ‼️"
            )

        # ---------- Resolve default k ----------
        if k is None:
            k = 5

            if self.model_pipeline is not None:
                clusterer = self.model_pipeline.named_steps.get(self.step_name, None)
                if clusterer is not None and isinstance(clusterer, DBSCAN):
                    k = getattr(clusterer, "min_samples", 5)

        if k < 1:
            raise ValueError("⚠️  k must be >= 1 ‼️")

        if k >= n_samples:
            raise ValueError("⚠️  k must be smaller than the number of samples ‼️")

        # ---------- Nearest-neighbor distance computation ----------
        # Set k+1 neighbors because the closest neighbor of each point is itself
        # with distance 0. Therefore:
        # - neighbors[:, 0] = self-distance
        # - neighbors[:, k] = distance to the k-th actual neighbor
        nn_obj = NearestNeighbors(n_neighbors=k + 1)
        nn_obj.fit(X_plot)

        distances, _ = nn_obj.kneighbors(X_plot)

        # Use the k-th actual neighbor distance after self-distance column
        k_distances = distances[:, k]

        # ---------- Optional zero-distance filtering ----------
        if ignore_zero_distance:
            k_distances = k_distances[k_distances > 0]

        if len(k_distances) == 0:
            raise ValueError(
                "⚠️  No valid k-distances remain after zero-distance filtering ‼️"
            )

        # ---------- Sort ascending for elbow-like inspection ----------
        k_distances_sorted = np.sort(k_distances)

        result_df = pd.DataFrame(
            {
                "sample_rank": np.arange(1, len(k_distances_sorted) + 1),
                "k_distance": k_distances_sorted,
            }
        )

        # ---------- Plot ----------
        plt.figure(figsize=figsize)
        plt.plot(result_df["sample_rank"], result_df["k_distance"])
        plt.title(f"DBSCAN k-Distance Plot (k={k})")
        plt.xlabel("Sorted Sample Rank")
        plt.ylabel(f"Distance to {k}-th Nearest Neighbor")
        plt.tight_layout()

        if save_fig:
            save_path = os.path.join(PLOT_DIR, fig_name)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()
        plt.close()

        return result_df


# =================================================
