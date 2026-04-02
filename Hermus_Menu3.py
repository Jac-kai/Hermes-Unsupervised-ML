# -------------------- Import Modules --------------------
import logging
from pprint import pprint

from Hermes.Hermus_ML_UnSup_Engine import HermesEngine
from Hermes.Menu_Config import EVALUATION_MENU_OPTIONS
from Hermes.Menu_Helper_Decorator import input_int, menu_wrapper

logger = logging.getLogger("Hermes")


# -------------------- Helper: ask save figure --------------------
def _select_save_fig_option() -> bool | None:
    """
    Ask whether generated figures should be saved.

    This helper displays a small numbered menu for figure-saving behavior and
    converts the user's numeric selection into a Boolean value.

    Parameters
    ----------
    None

    Returns
    -------
    bool or None
        - ``True``: save generated figures
        - ``False``: do not save generated figures
        - ``None``: user goes back or the selection is invalid

    Workflow
    --------
    1. Display the save-figure option menu.
    2. Read the selected menu number.
    3. Validate the selected number.
    4. Return the mapped Boolean value.

    Notes
    -----
    The default selection is ``2``, which maps to ``False``.

    Examples
    --------
    >>> save_fig = _select_save_fig_option()
    >>> print(save_fig)
    """
    logger.info("Selecting save-figure option")

    save_menu = {
        1: True,
        2: False,
    }

    print("\n----- 💾 Save Figure -----")
    for num, value in save_menu.items():
        print(f"📌 {num}. {value}")
    print("-" * 50)

    selected_num = input_int("🕯️ Select option", default=2)
    if selected_num is None:
        logger.info("Save-figure option selection cancelled")
        return None

    if selected_num not in save_menu:
        logger.warning("Save-figure option out of range: selected_num=%s", selected_num)
        print("⚠️ Selection is out of range ‼️")
        return None

    logger.info("Save-figure option selected: save_fig=%s", save_menu[selected_num])
    return save_menu[selected_num]


# -------------------- Helper: ask ignore noise --------------------
def _select_ignore_noise_option(default: int = 1) -> bool | None:
    """
    Ask whether cluster-noise samples should be ignored in evaluation plots.

    This helper is mainly used for clustering methods such as DBSCAN where
    noise points may exist. It displays a numbered yes/no-style menu and
    converts the user's numeric selection into a Boolean value.

    Parameters
    ----------
    default : int, default=1
        Default menu index passed to ``input_int`` when the user presses Enter.

    Returns
    -------
    bool or None
        - ``True``: ignore noise samples
        - ``False``: include noise samples
        - ``None``: user goes back or the selection is invalid

    Workflow
    --------
    1. Display the ignore-noise option menu.
    2. Read the selected menu number.
    3. Validate the selected number.
    4. Return the mapped Boolean value.

    Notes
    -----
    The numeric menu is mapped as:
    - ``1`` -> ``True``
    - ``2`` -> ``False``

    Examples
    --------
    >>> ignore_noise = _select_ignore_noise_option(default=2)
    >>> print(ignore_noise)
    """
    logger.info("Selecting ignore-noise option with default=%s", default)

    noise_menu = {
        1: True,
        2: False,
    }

    print("\n----- 🎧 Ignore Noise -----")
    for num, value in noise_menu.items():
        print(f"📌 {num}. {value}")
    print("-" * 50)

    selected_num = input_int("🕯️ Select option", default=default)
    if selected_num is None:
        logger.info("Ignore-noise option selection cancelled")
        return None

    if selected_num not in noise_menu:
        logger.warning(
            "Ignore-noise option out of range: selected_num=%s", selected_num
        )
        print("⚠️ Selection is out of range ‼️")
        return None

    logger.info(
        "Ignore-noise option selected: ignore_noise=%s", noise_menu[selected_num]
    )
    return noise_menu[selected_num]


# -------------------- Helper: ask PCA components --------------------
def _select_pca_components() -> int | None:
    """
    Ask for the number of PCA components used in cluster PCA visualization.

    This helper displays a small numbered menu for selecting the dimensionality
    of PCA-based cluster plots and returns the chosen component count.

    Parameters
    ----------
    None

    Returns
    -------
    int or None
        - ``2``: use 2 principal components
        - ``3``: use 3 principal components
        - ``None``: user goes back or the selection is invalid

    Workflow
    --------
    1. Display the PCA-component option menu.
    2. Read the selected menu number.
    3. Validate the selected number.
    4. Return the mapped PCA component count.

    Notes
    -----
    The default selection is ``1``, which maps to ``2`` components.

    Examples
    --------
    >>> n_components = _select_pca_components()
    >>> print(n_components)
    """
    logger.info("Selecting PCA components")

    component_menu = {
        1: 2,
        2: 3,
    }

    print("\n----- 🧠 PCA Components -----")
    print("🔔 Only 2D or 3D PCA projection is supported.")
    print("🔔 This does not mean the number of X variables.")
    for num, value in component_menu.items():
        print(f"📌 {num}. {value}")
    print("-" * 50)

    selected_num = input_int("🕯️ Select option", default=1)
    if selected_num is None:
        logger.info("PCA component selection cancelled")
        return None

    if selected_num not in component_menu:
        logger.warning(
            "PCA component selection out of range: selected_num=%s", selected_num
        )
        print("⚠️ Selection is out of range ‼️")
        return None

    logger.info(
        "PCA components selected: n_components=%s", component_menu[selected_num]
    )
    return component_menu[selected_num]


# -------------------- Helper: Show cluster --------------------
def _show_cluster_feature_map(hermes: HermesEngine) -> dict[int, str]:
    """
    Display the current clustering feature index map and return it.

    This helper reads the feature names from ``hermes.cluster_input_data``,
    builds a 1-based numeric mapping, prints the mapping for the user, and
    returns it for later feature-index selection in visualization methods.

    Parameters
    ----------
    hermes : HermesEngine
        Active Hermes engine instance containing the built clustering input
        dataset.

    Returns
    -------
    dict[int, str]
        Dictionary mapping 1-based feature indices to actual clustering input
        column names.

    Workflow
    --------
    1. Read feature names from ``hermes.cluster_input_data.columns``.
    2. Build a 1-based feature index mapping.
    3. Print the feature index map.
    4. Return the mapping.

    Notes
    -----
    This helper assumes that ``hermes.cluster_input_data`` already exists and
    is a column-based tabular object such as a pandas ``DataFrame``.

    Examples
    --------
    >>> feature_map = _show_cluster_feature_map(hermes)
    >>> print(feature_map)
    """
    logger.info("Showing cluster feature index map")

    feature_names = list(hermes.cluster_input_data.columns)
    feature_map = {i: name for i, name in enumerate(feature_names, 1)}
    logger.info("Cluster feature map prepared: feature_count=%s", len(feature_map))

    print("\n----- 🔎 Cluster Feature Index Map -----")
    for idx, name in feature_map.items():
        print(f"📌 {idx}. {name}")
    print("-" * 50)

    return feature_map


# -------------------- Helper: select k for DBSCAN k-distance plot --------------------
def _select_dbscan_k_distance_k(hermes: HermesEngine) -> int | None:
    """
    Ask for the ``k`` value used in the DBSCAN K-distance plot.

    This helper suggests a recommended ``k`` value based on the current DBSCAN
    model's ``min_samples`` attribute when available. If no current model or
    ``min_samples`` value is available, the helper falls back to ``5``.

    Parameters
    ----------
    hermes : HermesEngine
        Active Hermes engine instance. The helper optionally reads
        ``hermes.current_model.min_samples`` to determine a recommended default.

    Returns
    -------
    int or None
        - positive integer ``k`` value selected by the user
        - ``None``: user goes back or enters an invalid value

    Workflow
    --------
    1. Try to read ``min_samples`` from the current model.
    2. If unavailable, use ``5`` as the fallback recommendation.
    3. Display the recommended ``k`` value.
    4. Ask the user to enter a ``k`` value.
    5. Validate that ``k >= 1``.
    6. Return the validated ``k`` value.

    Notes
    -----
    In DBSCAN workflows, using ``min_samples`` as the K-distance plot ``k``
    value is a common practical starting point.

    Examples
    --------
    >>> k_value = _select_dbscan_k_distance_k(hermes)
    >>> print(k_value)
    """
    logger.info("Selecting DBSCAN K-distance k value")

    recommended_k = None

    if getattr(hermes, "current_model", None) is not None:
        recommended_k = getattr(hermes.current_model, "min_samples", None)

    if recommended_k is None:
        recommended_k = 5

    logger.info("DBSCAN K-distance recommended k resolved: %s", recommended_k)

    print("\n----- 📏 DBSCAN K-Distance k -----")
    print(f"🔥 Current DBSCAN min_samples : {recommended_k}")
    print(f"💡 Recommended k             : {recommended_k}")
    print("-" * 50)

    selected_k = input_int(
        "🕯️ Enter k for K-Distance",
        default=recommended_k,
    )
    if selected_k is None:
        logger.info("DBSCAN K-distance k selection cancelled")
        return None

    if selected_k < 1:
        logger.warning("Invalid DBSCAN K-distance k: %s", selected_k)
        print("⚠️ k must be at least 1 ‼️")
        return None

    return selected_k


# -------------------- Evaluation Menu --------------------
@menu_wrapper("Cluster Evaluation")
def evaluation_menu(hermes: HermesEngine):
    """
    Display and run cluster-evaluation tools for the current clustering model.

    This interactive menu provides access to summary views, evaluation outputs,
    previews, and visualization tools for the currently active clustering model
    stored in the Hermes engine.

    The menu supports result inspection features such as cluster-summary display,
    evaluation-result display, clustered-data previews, cluster-size summaries,
    and multiple plotting utilities including scatter plots, PCA plots,
    silhouette plots, heatmaps, elbow plots, K-distance plots, and dendrograms.

    Parameters
    ----------
    hermes : HermesEngine
        Active Hermes engine instance containing the current clustering model
        and its associated clustering results.

    Returns
    -------
    None
        Returns when the user exits the evaluation menu or when no current model
        is available.

    Workflow
    --------
    1. Validate that a current clustering model exists.
    2. Display the evaluation service menu.
    3. Read the user's selected service number.
    4. Dispatch the request to the matching summary, preview, or plotting tool.
    5. Continue looping until the user exits the menu.

    Menu Features
    -------------
    The menu may provide access to the following tools, depending on the current
    model implementation:

    - current model summary
    - cluster evaluation result
    - cluster-label preview
    - clustered-data preview
    - cluster-size summary
    - cluster-size bar plot
    - cluster scatter plot
    - cluster PCA plot
    - silhouette plot
    - cluster profile heatmap
    - KMeans elbow plot
    - DBSCAN K-distance plot
    - Agglomerative dendrogram plot

    Notes
    -----
    - Most evaluation actions are delegated to
      ``hermes.run_current_model_method()``.
    - Plotting options may ask for additional user input such as feature
      indices, PCA component count, whether to ignore noise, whether to save the
      figure, or a K-distance ``k`` value.
    - If no current model exists, the menu prints a warning and exits
      immediately.
    - Invalid menu selections are handled by warning messages instead of raising
      exceptions.

    Examples
    --------
    >>> evaluation_menu(hermes)
    """
    logger.info("Opening Cluster Evaluation Menu")

    if hermes.current_model is None:
        logger.warning("Cluster Evaluation Menu blocked: current_model is None")
        print("⚠️ No current model available. Please run clustering first ‼️")
        return

    while True:
        print("\n----- 🔥 Cluster Evaluation Menu 🔥 -----")
        for num, label in EVALUATION_MENU_OPTIONS.items():
            print(f"📘 {num}. {label}")
        print("-" * 50)

        selected_service = input_int("🕯️ Select service")
        if selected_service is None or selected_service == 0:
            logger.info("Exiting Cluster Evaluation Menu")
            return

        # ---------- Show current model summary ----------
        if selected_service == 1:
            logger.info("Evaluation service selected: show current model summary")
            hermes.show_current_model_summary()

        # ---------- Show cluster evaluation ----------
        elif selected_service == 2:
            logger.info("Evaluation service selected: show cluster evaluation")
            evaluation = hermes.get_cluster_evaluation()
            if evaluation is None:
                logger.warning("No cluster evaluation available")
                print("⚠️ No cluster evaluation available ‼️")
            else:
                pprint(evaluation)

        # ---------- Show cluster label preview ----------
        elif selected_service == 3:
            logger.info("Evaluation service selected: cluster label preview")
            preview = hermes.run_current_model_method(
                "cluster_label_preview_engine",
                n=10,
            )
            if preview is not None:
                print(preview)

        # ---------- Show clustered data preview ----------
        elif selected_service == 4:
            logger.info("Evaluation service selected: clustered data preview")
            preview = hermes.run_current_model_method("cluster_labels_in_data_engine")
            if preview is not None:
                print(preview.head(10))

        # ---------- Show cluster size summary ----------
        elif selected_service == 5:
            logger.info("Evaluation service selected: cluster size summary")
            summary = hermes.run_current_model_method("cluster_size_summary_engine")
            if summary is not None:
                print(summary)

        # ---------- Plot cluster size barplot ----------
        elif selected_service == 6:
            logger.info("Evaluation service selected: cluster size barplot")
            save_fig = _select_save_fig_option()
            if save_fig is None:
                logger.info(
                    "Cluster size barplot cancelled during save-figure selection"
                )
                continue

            hermes.run_current_model_method(
                "cluster_size_barplot_engine",
                save_fig=save_fig,
            )

        # ---------- Plot cluster scatter ----------
        elif selected_service == 7:
            logger.info("Evaluation service selected: cluster scatter plot")
            feature_map = _show_cluster_feature_map(hermes)

            feature_idx_1 = input_int("🕯️ Enter first feature index", default=0)
            if feature_idx_1 is None:
                logger.info(
                    "Cluster scatter plot cancelled during first feature selection"
                )
                continue

            if feature_idx_1 not in feature_map:
                logger.warning(
                    "First scatter feature index out of range: %s", feature_idx_1
                )
                print("⚠️ First feature index is out of range ‼️")
                continue

            feature_idx_2 = input_int("🕯️ Enter second feature index", default=1)
            if feature_idx_2 is None:
                logger.info(
                    "Cluster scatter plot cancelled during second feature selection"
                )
                continue

            if feature_idx_2 not in feature_map:
                logger.warning(
                    "Second scatter feature index out of range: %s", feature_idx_2
                )
                print("⚠️ Second feature index is out of range ‼️")
                continue

            if feature_idx_1 == feature_idx_2:
                logger.warning(
                    "Scatter plot feature indices cannot be the same: idx1=%s, idx2=%s",
                    feature_idx_1,
                    feature_idx_2,
                )
                print("⚠️ Two feature indexes cannot be the same ‼️")
                continue

            save_fig = _select_save_fig_option()
            if save_fig is None:
                continue

            logger.info(
                "Running cluster scatter plot: feature_idx_1=%s, feature_idx_2=%s, save_fig=%s",
                feature_idx_1,
                feature_idx_2,
                save_fig,
            )

            hermes.run_current_model_method(
                "cluster_scatter_plot_engine",
                feature_idx_1=feature_idx_1,
                feature_idx_2=feature_idx_2,
                save_fig=save_fig,
            )

        # ---------- Plot cluster PCA ----------
        elif selected_service == 8:
            logger.info("Evaluation service selected: cluster PCA plot")
            n_components = _select_pca_components()
            if n_components is None:
                logger.info("Cluster PCA plot cancelled during PCA component selection")
                continue

            ignore_noise = _select_ignore_noise_option(default=2)
            if ignore_noise is None:
                logger.info("Cluster PCA plot cancelled during ignore-noise selection")
                continue

            save_fig = _select_save_fig_option()
            if save_fig is None:
                continue

            logger.info(
                "Running cluster PCA plot: n_components=%s, ignore_noise=%s, save_fig=%s",
                n_components,
                ignore_noise,
                save_fig,
            )

            hermes.run_current_model_method(
                "cluster_pca_plot_engine",
                n_components=n_components,
                ignore_noise=ignore_noise,
                save_fig=save_fig,
            )

        # ---------- Plot silhouette ----------
        elif selected_service == 9:
            logger.info("Evaluation service selected: silhouette plot")
            ignore_noise = _select_ignore_noise_option(default=1)
            if ignore_noise is None:
                logger.info("Silhouette plot cancelled during ignore-noise selection")
                continue

            save_fig = _select_save_fig_option()
            if save_fig is None:
                logger.info("Silhouette plot cancelled during save-figure selection")
                continue

            logger.info(
                "Running silhouette plot: ignore_noise=%s, save_fig=%s",
                ignore_noise,
                save_fig,
            )

            hermes.run_current_model_method(
                "silhouette_plot_engine",
                ignore_noise=ignore_noise,
                save_fig=save_fig,
            )

        # ---------- Plot cluster profile heatmap ----------
        elif selected_service == 10:
            logger.info("Evaluation service selected: cluster profile heatmap")
            ignore_noise = _select_ignore_noise_option(default=2)
            if ignore_noise is None:
                logger.info(
                    "Cluster profile heatmap cancelled during ignore-noise selection"
                )
                continue

            save_fig = _select_save_fig_option()
            if save_fig is None:
                logger.info(
                    "Cluster profile heatmap cancelled during save-figure selection"
                )
                continue

            logger.info(
                "Running cluster profile heatmap: ignore_noise=%s, save_fig=%s",
                ignore_noise,
                save_fig,
            )

            hermes.run_current_model_method(
                "cluster_profile_heatmap_engine",
                ignore_noise=ignore_noise,
                save_fig=save_fig,
            )

        # ---------- KMeans Elbow Plot ----------
        elif selected_service == 11:
            logger.info("Evaluation service selected: KMeans elbow plot")
            k_start = input_int("🕯️ Enter k start", default=2)
            if k_start is None:
                logger.info("KMeans elbow plot cancelled during k_start selection")
                continue

            k_end = input_int("🕯️ Enter k end", default=10)
            if k_end is None:
                logger.info("KMeans elbow plot cancelled during k_end selection")
                continue

            save_fig = _select_save_fig_option()
            if save_fig is None:
                logger.info("KMeans elbow plot cancelled during save-figure selection")
                continue

            if k_start < 1 or k_end < 1 or k_start > k_end:
                logger.warning(
                    "Invalid KMeans elbow range: k_start=%s, k_end=%s",
                    k_start,
                    k_end,
                )
                print("⚠️ Invalid k range ‼️")
                continue

            logger.info(
                "Running KMeans elbow plot: k_start=%s, k_end=%s, save_fig=%s",
                k_start,
                k_end,
                save_fig,
            )

            hermes.run_current_model_method(
                "elbow_plot_engine",
                k_range=range(k_start, k_end + 1),
                save_fig=save_fig,
            )

        # ---------- DBSCAN K-Distance Plot ----------
        elif selected_service == 12:
            logger.info("Evaluation service selected: DBSCAN K-distance plot")
            k_value = _select_dbscan_k_distance_k(hermes)
            if k_value is None:
                logger.info("DBSCAN K-distance plot cancelled during k selection")
                continue

            save_fig = _select_save_fig_option()
            if save_fig is None:
                logger.info(
                    "DBSCAN K-distance plot cancelled during save-figure selection"
                )
                continue

            logger.info(
                "Running DBSCAN K-distance plot: k=%s, save_fig=%s",
                k_value,
                save_fig,
            )

            hermes.run_current_model_method(
                "k_distance_plot_engine",
                k=k_value,
                save_fig=save_fig,
            )

        # ---------- Agglomerative Dendrogram Plot ----------
        elif selected_service == 13:
            logger.info("Evaluation service selected: agglomerative dendrogram plot")
            save_fig = _select_save_fig_option()
            if save_fig is None:
                logger.info(
                    "Agglomerative dendrogram plot cancelled during save-figure selection"
                )
                continue

            logger.info("Running agglomerative dendrogram plot: save_fig=%s", save_fig)
            hermes.run_current_model_method(
                "dendrogram_plot_engine",
                save_fig=save_fig,
            )

        else:
            logger.warning(
                "Invalid Cluster Evaluation Menu selection: selected_service=%s",
                selected_service,
            )
            print("⚠️ Invalid selection ‼️")


# =================================================
