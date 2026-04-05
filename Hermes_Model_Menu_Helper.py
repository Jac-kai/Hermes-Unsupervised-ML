# -------------------- Import Modules --------------------
from Hermes.Hermes_ML_UnSup_Engine import HermesEngine
from Hermes.Menu_Config import COMMON_CLUSTER_PARAM_CONFIG, MODEL_CLUSTER_PARAM_CONFIG
from Hermes.Menu_Helper_Decorator import input_int


# -------------------- Helper: select model name --------------------
def select_model_name(hermes: HermesEngine) -> str | None:
    """
    Display available clustering models and let the user choose one by menu number.

    This helper retrieves the list of available unsupervised clustering models
    from the active ``HermesEngine`` instance, shows them as a numbered terminal
    menu, and returns the selected model name.

    If no clustering models are available, if the user cancels the selection,
    or if the selected number is outside the displayed range, the function
    returns ``None``.

    Parameters
    ----------
    hermes : HermesEngine
        Active Hermes engine instance used to retrieve available clustering
        model names.

    Returns
    -------
    str or None
        - ``str``: selected clustering model name
        - ``None``: when no model is available, the user goes back, or the
          selected menu number is invalid

    Workflow
    --------
    1. Retrieve available clustering model names from the engine.
    2. If no models are found, print a warning and return ``None``.
    3. Build a 1-based mapping between menu numbers and model names.
    4. Display the model selection menu.
    5. Read the selected menu number through ``input_int``.
    6. Validate the selected number.
    7. Return the corresponding model name.

    Notes
    -----
    This helper is intended for CLI-based clustering workflows where users
    select a model by displayed number instead of typing the model name
    manually.

    Examples
    --------
    >>> model_name = select_model_name(hermes)
    >>> print(model_name)
    """
    model_names = hermes.get_available_models()

    if not model_names:
        print("⚠️ No available clustering models found ‼️")
        return None

    model_map = {i: name for i, name in enumerate(model_names, 1)}

    print("\n----- 🔥 Available Clustering Models 🔥 -----")
    for i, name in model_map.items():
        print(f"🧠 {i}. {name}")
    print("-" * 50)

    selected_num = input_int("🕯️ Select model", default=-1)
    if selected_num is None:
        return None

    if selected_num not in model_map:
        print("⚠️ Model selection is out of range ‼️")
        return None

    return model_map[selected_num]


# -------------------- Helper: select from options --------------------
def select_from_options(
    label: str,
    options: dict[int, object],
    default: int | None = None,
):
    """
    Display a numbered option menu and return the value selected by the user.

    This helper prints a labeled list of selectable options for terminal-based
    workflows. Each option is displayed using its numeric key, and the user
    selects one by entering the corresponding number. The function then returns
    the mapped option value rather than the numeric key itself.

    If the user cancels the selection or enters a number outside the valid
    option range, the function returns ``None``.

    Parameters
    ----------
    label : str
        Display label shown above the option menu. This usually describes the
        parameter or setting currently being selected.
    options : dict[int, object]
        Dictionary mapping integer menu numbers to actual selectable values.
    default : int or None, default=None
        Default numeric menu key passed to ``input_int`` when the user presses
        Enter without typing a value.

    Returns
    -------
    object or None
        - selected option value from ``options``
        - ``None``: when the user goes back or selects an invalid menu number

    Workflow
    --------
    1. Print the selection label.
    2. Display all numbered options.
    3. Read the user's selected number through ``input_int``.
    4. If the user cancels, return ``None``.
    5. Validate whether the selected number exists in ``options``.
    6. Return the mapped option value.

    Notes
    -----
    The return type depends on the values stored in the ``options`` dictionary.
    For example, returned values may be strings, integers, floats, booleans,
    tuples, or any other configured object.

    Examples
    --------
    >>> metric = select_from_options(
    ...     label="Distance Metric",
    ...     options={1: "euclidean", 2: "manhattan"},
    ...     default=1,
    ... )
    >>> print(metric)
    """
    print(f"\n----- {label} -----")
    for num, value in options.items():
        print(f"📌 {num}. {value}")
    print("-" * 50)

    selected_num = input_int("🕯️ Select option", default=default)
    if selected_num is None:
        return None

    if selected_num not in options:
        print("⚠️ Selection is out of range ‼️")
        return None

    return options[selected_num]


# -------------------- Helper: should skip dependent param --------------------
def should_skip_param(param_config: dict, current_params: dict) -> bool:
    """
    Determine whether a parameter should be skipped based on dependency rules.

    This helper checks whether the current parameter has a dependency defined in
    its configuration. If a dependency exists, the function compares the already
    selected parameter value in ``current_params`` against the expected value
    declared in ``param_config["depends_on"]``.

    The parameter is skipped when the dependency condition is not satisfied.

    Parameters
    ----------
    param_config : dict
        Configuration dictionary for the current parameter. It may contain a
        ``"depends_on"`` entry in the form ``(param_name, expected_value)``.
    current_params : dict
        Dictionary containing parameter values already selected earlier in the
        same configuration workflow.

    Returns
    -------
    bool
        - ``True``: current parameter should be skipped
        - ``False``: current parameter should be asked normally

    Dependency Rules
    ----------------
    - If ``"depends_on"`` is missing or empty, the parameter is not skipped.
    - If the expected dependency value is a tuple, the current value must be one
      of the tuple values; otherwise the parameter is skipped.
    - If the expected dependency value is a single value, the current value must
      match it exactly; otherwise the parameter is skipped.

    Notes
    -----
    This helper is useful for dynamic parameter menus where some model
    hyperparameters only apply under specific settings chosen earlier.

    Examples
    --------
    >>> param_config = {"name": "p", "depends_on": ("metric", ("minkowski",))}
    >>> current_params = {"metric": "euclidean"}
    >>> should_skip_param(param_config, current_params)
    True
    """
    depends_on = param_config.get("depends_on")
    if not depends_on:
        return False

    dep_name, dep_expected_value = depends_on
    current_value = current_params.get(dep_name)

    if isinstance(dep_expected_value, tuple):
        return current_value not in dep_expected_value

    return current_value != dep_expected_value


# -------------------- Helper: collect common clustering params --------------------
def collect_common_cluster_params(hermes: HermesEngine) -> dict | None:
    """
    Collect common clustering preprocessing parameters from the user.

    This helper gathers shared preprocessing settings used before fitting a
    clustering model. It first checks whether clustering input data exists in
    the active ``HermesEngine`` instance. Then it inspects the input data for
    categorical columns and only asks for a categorical encoding method when
    such columns are present.

    Regardless of whether categorical columns exist, the function always asks
    the user to select a scaler type.

    Parameters
    ----------
    hermes : HermesEngine
        Active Hermes engine instance containing clustering input data in
        ``hermes.cluster_input_data``.

    Returns
    -------
    dict or None
        Dictionary of collected common clustering parameters.

        Possible keys include:
        - ``"cat_encoder"``: selected categorical encoding method
        - ``"scaler_type"``: selected scaler type

        Returns ``None`` when:
        - clustering input data is unavailable
        - the user cancels any required selection step

    Workflow
    --------
    1. Check whether clustering input data exists.
    2. Detect categorical columns from object, category, and bool dtypes.
    3. If categorical columns are found:
       - display the detected columns
       - ask the user to choose a categorical encoder
       - store the selected value in ``params["cat_encoder"]``
    4. If no categorical columns are found:
       - print an informational skip message
    5. Ask the user to choose a scaler type.
    6. Store the selected scaler in ``params["scaler_type"]``.
    7. Return the collected parameter dictionary.

    Notes
    -----
    This function intentionally skips categorical encoder selection when the
    dataset contains only numeric columns. This avoids unnecessary prompts in
    fully numeric clustering workflows.

    Examples
    --------
    >>> common_params = collect_common_cluster_params(hermes)
    >>> print(common_params)
    """
    if hermes.cluster_input_data is None:
        print("⚠️ No clustering input data available ‼️")
        return None

    params = {}

    categorical_cols = hermes.cluster_input_data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # ---------- Ask cat_encoder only when categorical columns exist ----------
    if categorical_cols:
        print("\n----- 🔎 Detected Categorical Columns -----")
        for col in categorical_cols:
            print(f"📌 {col}")
        print("-" * 50)

        config = COMMON_CLUSTER_PARAM_CONFIG["cat_encoder"]
        selected_value = select_from_options(
            label=config["label"],
            options=config["options"],
            default=config["default"],
        )
        if selected_value is None:
            return None

        params["cat_encoder"] = selected_value
    else:
        print("💡 No categorical columns detected. Skip categorical encoder selection.")

    # ---------- Always ask scaler_type ----------
    config = COMMON_CLUSTER_PARAM_CONFIG["scaler_type"]
    selected_value = select_from_options(
        label=config["label"],
        options=config["options"],
        default=config["default"],
    )
    if selected_value is None:
        return None

    params["scaler_type"] = selected_value

    return params


# -------------------- Helper: collect model-specific clustering kwargs --------------------
def collect_model_cluster_kwargs(model_name: str) -> dict | None:
    """
    Collect model-specific clustering hyperparameters from the user.

    This helper reads the parameter configuration of the specified clustering
    model from ``MODEL_CLUSTER_PARAM_CONFIG`` and interactively collects the
    corresponding keyword arguments for model initialization or fitting.

    Parameters may be conditionally skipped when their dependency rules are not
    satisfied. For DBSCAN, an additional guidance message is shown before asking
    for the ``eps`` parameter so the user has a rough starting reference.

    Parameters
    ----------
    model_name : str
        Name of the clustering model whose parameter configuration should be
        loaded from ``MODEL_CLUSTER_PARAM_CONFIG``.

    Returns
    -------
    dict or None
        Dictionary of collected model-specific keyword arguments.

        - keys are parameter names defined in the model configuration
        - values are the selected option values returned from the menu

        Returns ``None`` when:
        - the user cancels any parameter selection
        - a required selection step does not complete successfully

    Workflow
    --------
    1. Retrieve the parameter configuration list for the given model.
    2. Initialize an empty kwargs dictionary.
    3. Iterate through each parameter configuration entry.
    4. Skip parameters whose dependency conditions are not satisfied.
    5. For DBSCAN ``eps``, display a rough starting guideline before selection.
    6. Ask the user to select a value for the current parameter.
    7. Store the selected value in ``kwargs`` using the configured parameter name.
    8. Return the completed kwargs dictionary.

    Notes
    -----
    - If a model name is not found in ``MODEL_CLUSTER_PARAM_CONFIG``, the helper
      uses an empty list and returns an empty dictionary.
    - Dependency handling is delegated to ``should_skip_param``.
    - Option selection is delegated to ``select_from_options``.

    Examples
    --------
    >>> kwargs = collect_model_cluster_kwargs("KMeans")
    >>> print(kwargs)
    """
    param_list = MODEL_CLUSTER_PARAM_CONFIG.get(model_name, [])
    kwargs = {}

    for param_config in param_list:
        if should_skip_param(param_config, kwargs):
            continue

        if model_name == "DBSCAN" and param_config["name"] == "eps":
            print(
                "💡 DBSCAN first run usually starts with a rough eps guess."
                "💡 After fitting, use K-Distance Plot to refine eps."
                "💡 Common starting choices: 0.3 / 0.5 / 0.7 / 1.0"
            )
            print("-" * 50)

        selected_value = select_from_options(
            label=param_config["label"],
            options=param_config["options"],
            default=param_config.get("default"),
        )

        if selected_value is None:
            return None

        kwargs[param_config["name"]] = selected_value

    return kwargs


# =================================================
