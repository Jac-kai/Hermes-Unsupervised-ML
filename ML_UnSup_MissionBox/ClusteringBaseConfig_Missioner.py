# -------------------- Import Modules --------------------
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

from Hermes.ML_UnSup_BaseConfigBox.BaseUnSup_ModelConfig import BaseClusterConfig

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLOT_DIR = os.path.join(BASE_DIR, "ML_Report/Clusterer_Plot")
os.makedirs(PLOT_DIR, exist_ok=True)


# -------------------- Base Clustering Missioner --------------------
class BaseClustering_Missioner(BaseClusterConfig):
    """
    Base mission layer for clustering tasks.

    This class extends BaseClusterConfig and provides clustering-task logic
    shared across clustering algorithms such as:

    - KMeans
    - DBSCAN
    - AgglomerativeClustering

    Responsibilities
    ----------------
    - Evaluate fitted clustering results using internal metrics
    - Build compact clustering summary dictionaries
    - Preview clustering labels
    - Handle noise-aware evaluation logic for DBSCAN-style outputs

    Notes
    -----
    This mission layer assumes:
    - the pipeline has already been fitted
    - clustering outputs have already been captured into:
      self.labels_, self.cluster_centers_, self.inertia_
    """

    # -------------------- Initialization --------------------
    def __init__(self, cleaned_X_data: pd.DataFrame):
        super().__init__(cleaned_X_data)  # Initialization
        self.evaluation_result: Optional[Dict[str, Any]] = None
        self.prediction_preview: Optional[pd.DataFrame] = None
        self.clustered_data_preview: Optional[pd.DataFrame] = None
        self.cluster_size_summary: Optional[pd.DataFrame] = None
        self.cluster_profile_summary: Optional[pd.DataFrame] = None
        self.cluster_summary: Optional[Dict[str, Any]] = None

    # -------------------- Rebuild cluster state for loading engine --------------------
    def _rebuild_cluster_runtime_state(self):
        """
        Rebuild clustering runtime summaries and previews after loading.

        This helper regenerates mission-layer clustering outputs from the restored
        fitted pipeline and captured clustering attributes so that a loaded model
        behaves the same as a freshly trained model.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the fitted pipeline is unavailable.
        ValueError
            If cluster labels are unavailable.
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️ No fitted clustering pipeline found ‼️")

        if self.labels_ is None:
            raise ValueError("⚠️ Cluster labels not found after loading ‼️")

        # ---------- Rebuild evaluation ----------
        try:
            if self.input_model_type == "DBSCAN":
                self.clustering_evaluation_engine(ignore_noise=True)
            else:
                self.clustering_evaluation_engine(ignore_noise=False)
        except Exception as e:
            print(f"⚠️ Failed to rebuild clustering evaluation: {e} ‼️")

        # ---------- Rebuild summary ----------
        try:
            self.cluster_summary_engine()
        except Exception as e:
            print(f"⚠️ Failed to rebuild cluster summary: {e} ‼️")

        # ---------- Rebuild labeled data preview ----------
        try:
            self.cluster_labels_in_data_engine()
        except Exception as e:
            print(f"⚠️ Failed to rebuild clustered data preview: {e} ‼️")

        # ---------- Rebuild size summary ----------
        try:
            self.cluster_size_summary_engine()
        except Exception as e:
            print(f"⚠️ Failed to rebuild cluster size summary: {e} ‼️")

        # ---------- Rebuild profile summary ----------
        try:
            if self.input_model_type == "DBSCAN":
                self.cluster_profile_summary_engine(ignore_noise=True)
            else:
                self.cluster_profile_summary_engine(ignore_noise=False)
        except Exception as e:
            print(f"⚠️ Failed to rebuild cluster profile summary: {e} ‼️")

    # -------------------- Helper --------------------
    def _get_ready_data(
        self,
        ignore_noise: bool = True,
    ):
        """
        Prepare transformed X and labels for clustering metric calculation.

        Parameters
        ----------
        ignore_noise : bool, default=True
            Whether to exclude noise points with label -1.

        Returns
        -------
        tuple
            (X_metric, y_metric, noise_count, noise_ratio, n_clusters_found)

        Raises
        ------
        ValueError
            If transformed X or labels are unavailable.
        """
        if self.X_processed is None:
            raise ValueError(
                "⚠️  X_processed not found. Run fit_cluster_pipeline() first ‼️"
            )

        if self.labels_ is None:
            raise ValueError(
                "⚠️  Cluster labels not found. Run fit_cluster_pipeline() first ‼️"
            )

        # ---------- Numpy array exchanging ----------
        X_eval = np.asarray(self.X_processed)
        y_eval = np.asarray(self.labels_)

        if X_eval.ndim != 2:
            raise ValueError("⚠️  X_processed must be 2D for metric evaluation ‼️")

        # ---------- Noise calculation ----------
        noise_count = int(np.sum(y_eval == -1))
        noise_ratio = float(noise_count / len(y_eval)) if len(y_eval) > 0 else 0.0

        # ---------- Noise inspection ----------
        if ignore_noise:
            mask = y_eval != -1
            X_metric = X_eval[mask]
            y_metric = y_eval[mask]
        else:
            X_metric = X_eval
            y_metric = y_eval

        # ---------- Excluding noise ----------
        valid_clusters = (
            np.unique(y_eval[y_eval != -1]) if np.any(y_eval != -1) else np.array([])
        )
        n_clusters_found = int(len(valid_clusters))

        return X_metric, y_metric, noise_count, noise_ratio, n_clusters_found

    # -------------------- Clustering original dataset and label check --------------------
    def _check_cluster_ready(self):
        """
        Validate that clustering outputs are available.

        Raises
        ------
        ValueError
            If clustering labels or transformed feature matrix are unavailable.
        """
        if self.X_processed is None:
            raise ValueError("⚠️  X_processed not found. Run cluster_fit() first ‼️")

        if self.labels_ is None:
            raise ValueError("⚠️  Cluster labels not found. Run cluster_fit() first ‼️")

    # -------------------- Clustering evaluation --------------------
    def clustering_evaluation_engine(
        self,
        ignore_noise: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate fitted clustering results using internal clustering metrics.

        Parameters
        ----------
        ignore_noise : bool, default=True
            Whether to exclude noise points (-1) when computing metrics.

        Returns
        -------
        Dict[str, Any]
            Evaluation result dictionary.
        """
        X_metric, y_metric, noise_count, noise_ratio, n_clusters_found = (
            self._get_ready_data(ignore_noise=ignore_noise)
        )

        silhouette_val = None
        ch_val = None
        db_val = None

        if len(y_metric) >= 2 and len(np.unique(y_metric)) >= 2:
            try:
                silhouette_val = float(silhouette_score(X_metric, y_metric))
            except Exception:
                silhouette_val = None

            try:
                ch_val = float(calinski_harabasz_score(X_metric, y_metric))
            except Exception:
                ch_val = None

            try:
                db_val = float(davies_bouldin_score(X_metric, y_metric))
            except Exception:
                db_val = None

        result = {
            "n_samples": int(self.cleaned_X_data.shape[0]),
            "n_original_features": int(self.cleaned_X_data.shape[1]),
            "n_features_after_transform": (
                None if self.X_processed is None else int(self.X_processed.shape[1])
            ),
            "n_clusters_found": n_clusters_found,
            "noise_count": noise_count,
            "noise_ratio": noise_ratio,
            "silhouette_score": silhouette_val,
            "calinski_harabasz_score": ch_val,
            "davies_bouldin_score": db_val,
            "inertia": self.inertia_,
            "has_cluster_centers": self.cluster_centers_ is not None,
        }
        self.evaluation_result = result

        return result

    # -------------------- Summary --------------------
    def cluster_summary_engine(self) -> Dict[str, Any]:
        """
        Build a compact clustering summary dictionary.

        This method collects core fitted-model metadata and evaluation outputs into
        a single summary dictionary for downstream display, saving, or reporting.

        Returns
        -------
        Dict[str, Any]
            Summary dictionary containing model name, task type, scaler setting,
            sample count, feature counts, discovered cluster count, noise statistics,
            internal evaluation metrics, inertia, cluster-center availability, and
            transformed feature-name count.

        Raises
        ------
        ValueError
            If no fitted clustering pipeline is available.

        Notes
        -----
        If ``self.evaluation_result`` has not been built yet, this method calls
        ``clustering_evaluation_engine()`` first.
        """
        if self.model_pipeline is None:
            raise ValueError("⚠️  No fitted clustering pipeline found ‼️")

        if self.evaluation_result is None:
            self.clustering_evaluation_engine()

        clusterer = self.model_pipeline.named_steps.get(self.step_name, None)
        model_name = (
            clusterer.__class__.__name__ if clusterer is not None else "Unknown"
        )

        summary = {
            "model": model_name,
            "task": self.task,
            "scaler": self.input_use_scaler,
            "n_samples": self.evaluation_result["n_samples"],
            "n_original_features": self.evaluation_result["n_original_features"],
            "n_features_after_transform": self.evaluation_result[
                "n_features_after_transform"
            ],
            "n_clusters_found": self.evaluation_result["n_clusters_found"],
            "noise_count": self.evaluation_result["noise_count"],
            "noise_ratio": self.evaluation_result["noise_ratio"],
            "silhouette_score": self.evaluation_result["silhouette_score"],
            "calinski_harabasz_score": self.evaluation_result[
                "calinski_harabasz_score"
            ],
            "davies_bouldin_score": self.evaluation_result["davies_bouldin_score"],
            "inertia": self.evaluation_result["inertia"],
            "has_cluster_centers": self.evaluation_result["has_cluster_centers"],
            "feature_names_len": (
                None if self.feature_names is None else len(self.feature_names)
            ),
        }

        self.cluster_summary = summary
        return summary

    # -------------------- Label preview --------------------
    def cluster_label_preview_engine(self, n: int = 10) -> pd.DataFrame:
        """
        Preview cluster labels with original row indices.

        Parameters
        ----------
        n : int, default=10
            Number of rows to preview.

        Returns
        -------
        pd.DataFrame
            Preview DataFrame with original index and cluster labels.
        """
        if self.labels_ is None:
            raise ValueError("⚠️  Cluster labels not found. Run cluster_fit() first ‼️")

        preview = pd.DataFrame(
            {
                "original_index": self.cleaned_X_data.index,
                "cluster_label": self.labels_,
            }
        )

        self.prediction_preview = preview.head(n).T
        return self.prediction_preview

    # -------------------- Label and data after clustering --------------------
    def cluster_labels_in_data_engine(self) -> pd.DataFrame:
        """
        Attach cluster labels to the original cleaned input data.

        Returns
        -------
        pd.DataFrame
            Original cleaned_X_data with an additional `cluster_label` column.
        """
        if self.labels_ is None:
            raise ValueError("⚠️  Cluster labels not found. Run cluster_fit() first ‼️")

        df = self.cleaned_X_data.copy()
        df["cluster_label"] = self.labels_
        self.clustered_data_preview = df

        return df

    # -------------------- Clustering size summary --------------------
    def cluster_size_summary_engine(self) -> pd.DataFrame:
        """
        Summarize the number of samples and proportion in each cluster.

        Returns
        -------
        pd.DataFrame
            Summary table with cluster label, sample count, and ratio.
        """
        if self.labels_ is None:
            raise ValueError("⚠️  Cluster labels not found. Run cluster_fit() first ‼️")

        labels_series = pd.Series(self.labels_, name="cluster_label")
        summary = (
            labels_series.value_counts(dropna=False)
            .sort_index()
            .rename_axis("cluster_label")
            .reset_index(name="count")
        )
        summary["ratio"] = summary["count"] / summary["count"].sum()
        self.cluster_size_summary = summary

        return summary

    # -------------------- Clustering profile summary --------------------
    def cluster_profile_summary_engine(
        self,
        numeric_only: bool = True,
        ignore_noise: bool = False,
    ) -> pd.DataFrame:
        """
        Summarize mean feature values for each cluster.

        Parameters
        ----------
        numeric_only : bool, default=True
            Whether to aggregate only numeric columns.
        ignore_noise : bool, default=False
            Whether to exclude noise samples with label ``-1`` before aggregation.

        Returns
        -------
        pd.DataFrame
            Cluster-level feature mean summary table.

        Raises
        ------
        ValueError
            If clustering labels are not available.
        ValueError
            If no rows remain after optional noise filtering.
        ValueError
            If no numeric columns are available for aggregation.
        """
        if self.labels_ is None:
            raise ValueError("⚠️  Cluster labels not found. Run cluster_fit() first ‼️")

        df = self.cleaned_X_data.copy()
        df["cluster_label"] = self.labels_

        if ignore_noise:
            df = df[df["cluster_label"] != -1]

        if df.empty:
            raise ValueError("⚠️  No data available after noise filtering ‼️")

        if numeric_only:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != "cluster_label"]

            if not numeric_cols:
                raise ValueError(
                    "⚠️  No numeric columns available for cluster profile summary ‼️"
                )

            profile = df.groupby("cluster_label")[numeric_cols].mean()
        else:
            profile = df.groupby("cluster_label").mean(numeric_only=True)

        self.cluster_profile_summary = profile
        return profile

    # -------------------- Clustering size barplot --------------------
    def cluster_size_barplot_engine(
        self,
        figsize: tuple = (8, 5),
        save_fig: bool = False,
        fig_name: str = "cluster_size_barplot.png",
    ):
        """
        Plot the sample count of each cluster as a bar chart.

        This method builds a cluster-size summary table through
        ``cluster_size_summary_engine()`` and visualizes the sample count of each
        cluster label using a matplotlib bar chart.

        Parameters
        ----------
        figsize : tuple, default=(8, 5)
            Figure size passed to ``plt.figure()``.
        save_fig : bool, default=False
            Whether to save the generated figure to ``PLOT_DIR``.
        fig_name : str, default="cluster_size_barplot.png"
            Output file name used when ``save_fig=True``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If cluster labels are unavailable and the size summary cannot be built.

        Notes
        -----
        The x-axis shows cluster labels converted to string form, and the y-axis
        shows the number of samples assigned to each cluster.
        """
        summary = self.cluster_size_summary_engine()

        plt.figure(figsize=figsize)
        plt.bar(summary["cluster_label"].astype(str), summary["count"])
        plt.title("Cluster Size Distribution")
        plt.xlabel("Cluster Label")
        plt.ylabel("Count")
        plt.tight_layout()

        if save_fig:
            save_path = os.path.join(PLOT_DIR, fig_name)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    # -------------------- Clustering scatter plot --------------------
    def cluster_scatter_plot_engine(
        self,
        feature_idx_1: int = 0,
        feature_idx_2: int = 1,
        figsize: tuple = (8, 6),
        save_fig: bool = False,
        fig_name: str = "cluster_scatter_plot.png",
    ):
        """
        Plot a 2D scatter chart using two transformed feature dimensions.

        This method visualizes clustering results in the transformed feature space
        stored in ``self.X_processed``. Each point is colored by its cluster label,
        and noise points (label ``-1``) are shown separately when present.

        Parameters
        ----------
        feature_idx_1 : int, default=0
            Column index in ``self.X_processed`` used for the x-axis.
        feature_idx_2 : int, default=1
            Column index in ``self.X_processed`` used for the y-axis.
        figsize : tuple, default=(8, 6)
            Figure size passed to ``plt.figure()``.
        save_fig : bool, default=False
            Whether to save the generated figure to ``PLOT_DIR``.
        fig_name : str, default="cluster_scatter_plot.png"
            Output file name used when ``save_fig=True``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If clustering outputs are unavailable.
        ValueError
            If ``self.X_processed`` has fewer than 2 feature dimensions.
        ValueError
            If ``feature_idx_1`` and ``feature_idx_2`` are the same.
        ValueError
            If either feature index is outside the valid column range.

        Notes
        -----
        This plot uses the already transformed feature matrix and does not apply PCA
        or any additional dimensionality reduction.
        """
        self._check_cluster_ready()

        X_plot = np.asarray(self.X_processed)
        y_plot = np.asarray(self.labels_)

        if X_plot.shape[1] < 2:
            raise ValueError(
                "⚠️  X_processed has fewer than 2 features. Use PCA plot instead ‼️"
            )

        if feature_idx_1 == feature_idx_2:
            raise ValueError("⚠️  feature_idx_1 and feature_idx_2 must be different ‼️")

        if feature_idx_1 >= X_plot.shape[1] or feature_idx_2 >= X_plot.shape[1]:
            raise ValueError("⚠️  Feature index out of range for X_processed ‼️")

        plt.figure(figsize=figsize)

        unique_labels = np.unique(y_plot)
        for label in unique_labels:
            mask = y_plot == label
            legend_name = f"Cluster {label}" if label != -1 else "Noise (-1)"
            plt.scatter(
                X_plot[mask, feature_idx_1],
                X_plot[mask, feature_idx_2],
                alpha=0.7,
                label=legend_name,
            )

        plt.title("Cluster Scatter Plot")
        plt.xlabel(f"Processed Feature {feature_idx_1}")
        plt.ylabel(f"Processed Feature {feature_idx_2}")
        plt.legend()
        plt.tight_layout()

        if save_fig:
            save_path = os.path.join(PLOT_DIR, fig_name)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    # -------------------- Clustering PCA plot --------------------
    def cluster_pca_plot_engine(
        self,
        n_components: int = 2,
        ignore_noise: bool = False,
        figsize: tuple = (8, 6),
        save_fig: bool = False,
        fig_name: str = "cluster_pca_plot.png",
    ):
        """
        Visualize clustering results in 2D or 3D projected space.

        Plotting Logic
        --------------
        This method first uses ``self.X_processed`` as the plotting source.

        - If ``self.X_processed`` already has exactly the requested number of dimensions,
        it is plotted directly.
        - If ``self.X_processed`` has more dimensions than requested, PCA is applied
        only for visualization.
        - If ``self.X_processed`` has fewer dimensions than requested, an error is raised.

        This design avoids unnecessary repeated PCA when dimensionality reduction
        has already been included earlier in the fitted pipeline via ``extra_steps``.

        Parameters
        ----------
        n_components : int, default=2
            Target plotting dimension. Supported values are:
            - 2 for 2D scatter plot
            - 3 for 3D scatter plot

        ignore_noise : bool, default=False
            Whether to exclude noise samples with cluster label ``-1`` before plotting.

        figsize : tuple, default=(8, 6)
            Figure size.

        save_fig : bool, default=False
            Whether to save the figure.

        fig_name : str, default="cluster_pca_plot.png"
            Output file name when ``save_fig=True``.

        Raises
        ------
        ValueError
            If clustering outputs are unavailable.
        ValueError
            If ``n_components`` is not 2 or 3.
        ValueError
            If no samples remain after optional noise filtering.
        ValueError
            If ``self.X_processed`` has fewer dimensions than ``n_components``.
        """
        self._check_cluster_ready()

        if n_components not in [2, 3]:
            raise ValueError("⚠️  n_components must be 2 or 3 ‼️")

        X_plot = np.asarray(self.X_processed)
        y_plot = np.asarray(self.labels_)

        if ignore_noise:
            mask = y_plot != -1
            X_plot = X_plot[mask]
            y_plot = y_plot[mask]

        if len(X_plot) == 0:
            raise ValueError("⚠️  No samples available for plotting ‼️")

        if X_plot.ndim != 2:
            raise ValueError("⚠️  X_processed must be a 2D array for plotting ‼️")

        current_dim = X_plot.shape[1]

        if current_dim < n_components:
            raise ValueError(
                f"⚠️  X_processed has only {current_dim} dimension(s); "
                f"cannot plot {n_components}D view ‼️"
            )

        # ---------- Use current transformed space directly if already matched ----------
        if current_dim == n_components:
            X_vis = X_plot
            title_suffix = "Direct View"
            axis_prefix = "Dim"

        # ---------- Apply PCA only when dimensionality is higher than requested ----------
        else:
            pca = PCA(n_components=n_components)
            X_vis = pca.fit_transform(X_plot)
            title_suffix = "PCA Projection"
            axis_prefix = "PC"

        # ---------- 2D plot ----------
        if n_components == 2:
            plt.figure(figsize=figsize)

            for label in np.unique(y_plot):
                mask = y_plot == label
                legend_name = f"Cluster {label}" if label != -1 else "Noise (-1)"
                plt.scatter(
                    X_vis[mask, 0],
                    X_vis[mask, 1],
                    alpha=0.7,
                    label=legend_name,
                )

            plt.title(f"Cluster Plot (2D, {title_suffix})")
            plt.xlabel(f"{axis_prefix}1")
            plt.ylabel(f"{axis_prefix}2")
            plt.legend()
            plt.tight_layout()

            if save_fig:
                save_path = os.path.join(PLOT_DIR, fig_name)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()

        # ---------- 3D plot ----------
        else:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

            for label in np.unique(y_plot):
                mask = y_plot == label
                legend_name = f"Cluster {label}" if label != -1 else "Noise (-1)"
                ax.scatter(
                    X_vis[mask, 0],
                    X_vis[mask, 1],
                    X_vis[mask, 2],
                    alpha=0.7,
                    label=legend_name,
                )

            ax.set_title(f"Cluster Plot (3D, {title_suffix})")
            ax.set_xlabel(f"{axis_prefix}1")
            ax.set_ylabel(f"{axis_prefix}2")
            ax.set_zlabel(f"{axis_prefix}3")
            ax.legend()
            plt.tight_layout()

            if save_fig:
                save_path = os.path.join(PLOT_DIR, fig_name)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.show()
            plt.close(fig)

    # -------------------- Silhouette plot --------------------
    def silhouette_plot_engine(
        self,
        ignore_noise: bool = True,
        figsize: tuple = (8, 6),
        save_fig: bool = False,
        fig_name: str = "silhouette_plot.png",
    ):
        """
        Plot silhouette values for each sample grouped by cluster.

        This method computes per-sample silhouette coefficients from the processed
        feature matrix and fitted cluster labels, then visualizes the distribution
        of silhouette values for each cluster as a stacked horizontal silhouette plot.

        Parameters
        ----------
        ignore_noise : bool, default=True
            Whether to exclude samples labeled ``-1`` before silhouette calculation.
        figsize : tuple, default=(8, 6)
            Figure size passed to ``plt.figure()``.
        save_fig : bool, default=False
            Whether to save the generated figure to ``PLOT_DIR``.
        fig_name : str, default="silhouette_plot.png"
            Output file name used when ``save_fig=True``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If transformed data or cluster labels are unavailable.
        ValueError
            If fewer than 2 valid clusters remain for silhouette analysis.

        Notes
        -----
        A vertical dashed line is drawn at the average silhouette score to help
        compare per-cluster distributions against the global clustering quality.
        """
        X_metric, y_metric, _, _, _ = self._get_ready_data(ignore_noise=ignore_noise)

        unique_clusters = np.unique(y_metric)
        if len(unique_clusters) < 2:
            raise ValueError(
                "⚠️  Silhouette plot requires at least 2 valid clusters ‼️"
            )

        sil_values = silhouette_samples(X_metric, y_metric)
        sil_avg = silhouette_score(X_metric, y_metric)

        plt.figure(figsize=figsize)

        y_lower = 10
        for cluster_id in unique_clusters:
            cluster_sil_values = sil_values[y_metric == cluster_id]
            cluster_sil_values.sort()

            size_cluster_i = cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i

            plt.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_sil_values,
                alpha=0.7,
            )
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster_id))
            y_lower = y_upper + 10

        plt.axvline(x=sil_avg, linestyle="--")
        plt.title("Silhouette Plot")
        plt.xlabel("Silhouette Coefficient")
        plt.ylabel("Cluster")
        plt.tight_layout()

        if save_fig:
            save_path = os.path.join(PLOT_DIR, fig_name)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    # -------------------- Clustering profile heatmap --------------------
    def cluster_profile_heatmap_engine(
        self,
        numeric_only: bool = True,
        ignore_noise: bool = False,
        figsize: tuple = (10, 6),
        save_fig: bool = False,
        fig_name: str = "cluster_profile_heatmap.png",
    ):
        """
        Plot cluster mean feature profiles as a heatmap.

        This method summarizes cluster-level mean feature values using
        ``cluster_profile_summary_engine()`` and visualizes the resulting table
        as a seaborn heatmap. Each row represents a cluster label and each column
        represents an aggregated feature.

        Parameters
        ----------
        numeric_only : bool, default=True
            Whether to aggregate only numeric columns when building the cluster
            profile summary.

        ignore_noise : bool, default=False
            Whether to exclude samples labeled ``-1`` before computing the cluster
            profile summary. This is mainly useful for clustering methods that may
            assign noise labels, such as DBSCAN.

        figsize : tuple, default=(10, 6)
            Figure size in inches, passed to ``plt.figure()``.

        save_fig : bool, default=False
            Whether to save the generated figure to ``PLOT_DIR``.

        fig_name : str, default="cluster_profile_heatmap.png"
            Output file name used when ``save_fig=True``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If clustering labels are unavailable.
        ValueError
            If no rows remain after optional noise filtering.
        ValueError
            If no numeric columns are available for aggregation.

        Notes
        -----
        The heatmap uses fixed display settings for annotation, numeric formatting,
        and colormap so that plot outputs remain visually consistent across runs.
        """
        profile = self.cluster_profile_summary_engine(
            numeric_only=numeric_only,
            ignore_noise=ignore_noise,
        )

        plt.figure(figsize=figsize)
        sns.heatmap(
            profile,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar=True,
        )
        plt.title("Cluster Profile Heatmap")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_fig:
            save_path = os.path.join(PLOT_DIR, fig_name)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()
        plt.close()


# =================================================
