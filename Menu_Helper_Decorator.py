# -------------------- Modules Import --------------------
from functools import wraps


# -------------------- Menu Wrapper --------------------
def menu_wrapper(menu_name: str):
    """
    Create a decorator that wraps terminal menu functions with a standard header,
    execution message, and basic exception handling.

    This decorator is intended for interactive menu-based workflows. It prints a
    formatted menu title before the wrapped function runs, then prints an execution
    message after successful completion. If an exception occurs during execution,
    the decorator catches it, displays an error message, and returns ``None``
    instead of letting the exception stop the whole menu system.

    Parameters
    ----------
    menu_name : str
        Display name of the menu or action. This name is shown in the header,
        success message, and error message.

    Returns
    -------
    callable
        A decorator function that can be applied to another callable.

    Notes
    -----
    The wrapped function keeps its original metadata because ``functools.wraps``
    is applied.

    Examples
    --------
    >>> @menu_wrapper("Load Data")
    ... def load_data_menu():
    ...     print("Loading...")
    ...
    >>> load_data_menu()
    """

    def decorator(func):
        """
        Wrap a target function with a standardized terminal menu presentation layer.

        This inner decorator receives the target function and returns a wrapped
        version that adds formatted output and exception handling.

        Parameters
        ----------
        func : callable
            Target function to decorate.

        Returns
        -------
        callable
            Wrapped function with menu header printing and safe execution behavior.
        """

        @wraps(func)
        def wrapped(*args, **kwargs):
            print(f"---------- 🔥  {menu_name} 🔥  ----------")
            try:
                result = func(*args, **kwargs)
                print(f"🕯️ 🕯️ 🕯️  Executing... {menu_name} 🕯️ 🕯️ 🕯️\n")
                return result
            except Exception as e:
                print(f"⚠️ [ERROR] {menu_name} failed: {e} ‼️")
                return None

        return wrapped

    return decorator


# -------------------- input_int --------------------
def input_int(prompt: str, default: int | None = None) -> int | None:
    """
    Read an integer value from terminal input.

    This helper prompts the user to enter a numeric value for menu-style terminal
    workflows. If the user enters ``0``, the function treats it as a back/cancel
    action and returns ``None``. If the user presses Enter without typing anything,
    the provided default value is returned. If conversion to integer fails, the
    function prints a warning message and also returns the default value.

    Parameters
    ----------
    prompt : str
        Prompt message shown to the user before reading input.
    default : int or None, default=None
        Value returned when the user provides empty input or enters an invalid
        non-integer value.

    Returns
    -------
    int or None
        - ``int``: valid integer entered by the user
        - ``default``: when input is empty or invalid
        - ``None``: when the user enters ``0`` to go back

    Examples
    --------
    >>> value = input_int("Enter fold number", default=5)
    >>> print(value)
    """
    try:
        value = input(prompt + " (Number only | 0 to ↩️  BACK) ⚡ ").strip()

        if value == "0":
            return None

        return int(value) if value else default

    except ValueError:
        print(f"⚠️ Invalid input, using default {default} ‼️")
        return default


# -------------------- input_yesno --------------------
def input_yesno(prompt: str, default: bool = False) -> bool | None:
    """
    Read a yes/no response from terminal input and convert it to a Boolean value.

    This helper repeatedly prompts the user until a valid yes/no response is
    entered. Accepted yes values are ``"y"`` and ``"yes"``, and accepted no
    values are ``"n"`` and ``"no"``. If the user presses Enter without typing
    anything, the function returns the provided default value. If the user enters
    ``0``, the function treats it as a back/cancel action and returns ``None``.

    Parameters
    ----------
    prompt : str
        Prompt message shown to the user before reading input.
    default : bool, default=False
        Boolean value returned when the user presses Enter without typing any
        response.

    Returns
    -------
    bool or None
        - ``True``: user entered ``y`` or ``yes``
        - ``False``: user entered ``n`` or ``no``
        - ``default``: user pressed Enter without input
        - ``None``: user entered ``0`` to go back

    Notes
    -----
    The function loops until valid input is received, unless the user chooses to
    go back.

    Examples
    --------
    >>> save_file = input_yesno("Save result?", default=True)
    >>> print(save_file)
    """
    while True:
        value = (
            input(prompt + " (y or yes/n or no | 0 to ↩️  BACK) ⚡ ").strip().lower()
        )

        if value == "0":
            return None
        if value == "":
            return default
        if value in ["y", "yes"]:
            return True
        if value in ["n", "no"]:
            return False

        print("⚠️ Invalid input, please enter y/yes or n/no ‼️")


# -------------------- input_list --------------------
def input_list(prompt: str) -> list[str] | str | None:
    """
    Read a comma-separated list of text values from terminal input.

    This helper is used in terminal workflows where the user needs to provide one
    or multiple values in a single input field. The entered text is split by
    commas, stripped of surrounding whitespace, and returned as a list of strings.
    If the user presses Enter without typing anything, the function returns
    ``None``. If the user enters ``0``, the function returns the special sentinel
    value ``"__BACK__"`` to indicate a back/cancel action.

    Parameters
    ----------
    prompt : str
        Prompt message shown to the user before reading input.

    Returns
    -------
    list[str] or str or None
        - ``list[str]``: parsed non-empty items from comma-separated input
        - ``"__BACK__"``: user entered ``0`` to go back
        - ``None``: user pressed Enter without input or an unexpected error occurred

    Notes
    -----
    Empty items created by repeated commas are ignored.

    Examples
    --------
    >>> cols = input_list("Enter selected columns")
    >>> print(cols)
    """
    try:
        value = input(
            f"{prompt} (comma-separated) (ENTER to skip | 0 to ↩️  BACK) ⚡ "
        ).strip()

        if value == "0":
            return "__BACK__"

        return [v.strip() for v in value.split(",") if v.strip()] if value else None

    except Exception:
        print("⚠️ Failed to read list input, returning None ‼️")
        return None


# -------------------- index_list --------------------
def index_list(data: object) -> dict[int, object]:
    """
    Display the index values of a data object and build a numbered index mapping.

    This helper is mainly designed for pandas-like objects that expose an
    ``index`` attribute. It validates whether the input data exists, whether it
    has an index, and whether the index is non-empty. If valid, it prints the
    index values as a numbered menu and returns a dictionary mapping display
    numbers to actual index values.

    Parameters
    ----------
    data : object
        Target data object expected to provide an ``index`` attribute, such as a
        pandas ``DataFrame`` or ``Series``.

    Returns
    -------
    dict[int, object]
        Dictionary mapping 1-based display numbers to actual index values.

        - Returns an empty dictionary when:
          - ``data`` is ``None``
          - ``data`` has no ``index`` attribute
          - the index is empty
          - an unexpected error occurs

    Notes
    -----
    This function is intended for menu-based selection workflows where users need
    to choose rows or labels by displayed number instead of typing raw index
    values directly.

    Examples
    --------
    >>> idx_map = index_list(df)
    >>> print(idx_map)
    """
    try:
        if data is None:
            print("⚠️ No data available ‼️")
            return {}

        if not hasattr(data, "index"):
            print("⚠️ Target data has no index attribute ‼️")
            return {}

        if len(data.index) == 0:
            print("⚠️ No index found in target data ‼️")
            return {}

        idx_map = {i: idx for i, idx in enumerate(data.index, 1)}

        print("🍁----- Index List -----🍁")
        for i, idx in idx_map.items():
            print(f"🐝 {i}. {idx}")
        print("-" * 40)

        return idx_map

    except Exception as e:
        print(f"⚠️ Failed to display index list: {e} ‼️")
        return {}


# -------------------- column_list --------------------
def column_list(data: object) -> dict[int, str]:
    """
    Display the column names of a tabular data object and build a numbered column mapping.

    This helper is intended for pandas-like tabular objects that expose
    ``columns`` and ``dtypes`` attributes. It validates whether the input data
    exists, whether it has columns, and whether the column set is non-empty. If
    valid, it prints each column as a numbered item together with its data type,
    then returns a dictionary mapping display numbers to actual column names.

    Parameters
    ----------
    data : object
        Target tabular data object expected to provide ``columns`` and ``dtypes``
        attributes, such as a pandas ``DataFrame``.

    Returns
    -------
    dict[int, str]
        Dictionary mapping 1-based display numbers to actual column names.

        - Returns an empty dictionary when:
          - ``data`` is ``None``
          - ``data`` has no ``columns`` attribute
          - the column set is empty
          - an unexpected error occurs

    Notes
    -----
    This function is useful in terminal-based data workflows where users select
    columns by displayed number instead of manually typing full column names.

    Examples
    --------
    >>> col_map = column_list(df)
    >>> print(col_map)
    """
    try:
        if data is None:
            print("⚠️ No data available ‼️")
            return {}

        if not hasattr(data, "columns"):
            print("⚠️ Target data has no columns attribute ‼️")
            return {}

        if len(data.columns) == 0:
            print("⚠️ No columns found in target data ‼️")
            return {}

        col_map = {i: col for i, col in enumerate(data.columns, 1)}
        col_type_map = {col: str(dtype) for col, dtype in data.dtypes.items()}

        print(f"🍁----- Column List -----🍁")
        for i, col in col_map.items():
            print(f"🐝 {i}. {col} ({col_type_map[col]})")
        print("-" * 40)

        return col_map

    except Exception as e:
        print(f"⚠️ Failed to display column list: {e} ‼️")
        return {}


# -------------------- Helper: input text --------------------
def input_text_value(prompt: str) -> str | None:
    """
    Read a free-text value from terminal input.

    This helper allows the user to manually type a text value in terminal-based
    workflows. If the user enters ``0``, the function treats it as a back/cancel
    action and returns ``None``. Otherwise, the raw stripped text is returned.

    Parameters
    ----------
    prompt : str
        Prompt message shown to the user before reading input.

    Returns
    -------
    str or None
        - ``str``: user-entered text after stripping leading and trailing whitespace
        - ``None``: user entered ``0`` to go back

    Examples
    --------
    >>> name = input_text_value("Enter feature name")
    >>> print(name)
    """
    value = input(f"{prompt} (typing manually) (0 to ↩️  BACK) ⚡ ").strip()

    if value == "0":
        return None

    return value


# -----------------------------------------
