# -------------------- Modules Import --------------------
import logging
import time

from Hermes.Hermes_Logging import hermes_init_logging
from Hermes.Hermus_Menu1 import loaded_ml_data_menu, select_input_menu
from Hermes.Hermus_Menu2 import cluster_model_menu
from Hermes.Hermus_Menu3 import evaluation_menu
from Hermes.Hermus_ML_UnSup_Engine import HermesEngine
from Hermes.Menu_Helper_Decorator import input_int

logger = logging.getLogger("Hermes")


# -------------------- cornus_control --------------------
def hermes_control():
    """
    Run the main terminal control loop for the Hermes unsupervised-learning system.

    This function creates a single ``HermesEngine`` instance and launches the
    top-level interactive menu for the Hermes workflow. Through this menu, the
    user can:

    1. load a dataset,
    2. select input columns and build the clustering X dataset,
    3. choose and run a clustering model,
    4. open clustering evaluation tools.

    The function keeps the same engine instance alive across menu selections so
    that loaded data, selected input columns, built clustering input data,
    trained clustering models, and evaluation results can be reused throughout
    the session.

    Workflow
    --------
    1. Create a ``HermesEngine`` instance.
    2. Define the main Hermes menu entries.
    3. Continuously display the main menu.
    4. Read the user's numeric menu selection.
    5. Execute the matched menu function using the same engine instance.
    6. Exit the loop when the user chooses to leave the system.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function runs the interactive Hermes terminal menu and returns only
        when the user exits the system.

    Notes
    -----
    - The same ``HermesEngine`` instance is reused for the entire session.
    - Entering the back command in the main menu exits the Hermes system.
    - Invalid selections are handled by warning messages instead of raising
      exceptions.
    - Menu functions are expected to accept a single ``HermesEngine`` argument.

    Examples
    --------
    >>> hermes_control()
    """
    logger.info("Starting Hermes main control loop")

    hermes_engine = HermesEngine()
    logger.info("HermesEngine instance created")

    menu = [
        (1, "📨 Upload Data", loaded_ml_data_menu),
        (2, "🔎 Input X dataset", select_input_menu),
        (3, "🧠 Cluster selection", cluster_model_menu),
        (4, "🪶 Evaluation", evaluation_menu),
        (0, "🍂 Leave System", None),
    ]
    menu_width = 35

    while True:
        print("🏮  Hermes Main Menu 🏮 ".center(menu_width, "━"))

        for opt, action, _ in menu:
            print(f"{opt}. {action:<{menu_width-6}}")
        print("━" * menu_width)

        choice = input_int(f"🕯️  Select Services (🔅 {time.asctime()})⚡ ", default=-1)
        logger.info("Hermes main menu selection received: %s", choice)

        if choice is None:
            logger.info("Hermes main menu exited by user")
            print("🎶🎶🎶 Leaving Hermes Engine... Goodbye 🍁 Zack King")
            break

        if choice == -1:
            logger.warning("Invalid Hermes main menu selection: %s", choice)
            print("⚠️ Invalid selection ‼️")
            continue

        for opt, label, func in menu:
            if choice == opt and func:
                logger.info("Opening Hermes menu: %s - %s", choice, label)
                func(hermes_engine)
                logger.info("Returned from Hermes menu: %s - %s", choice, label)
                break

            if choice == 0 and opt == 0:
                logger.info("User selected to leave Hermes system")
                print("🎶🎶🎶 Leaving Hermes Engine... Goodbye 🍁 Zack King")
                return
        else:
            logger.warning("Unmatched Hermes main menu selection: %s", choice)
            print("⚠️ Invalid selection ‼️")


# -------------------- Execute --------------------
if __name__ == "__main__":
    logger = hermes_init_logging()
    logger.info("Executing Hermes application entry point")
    hermes_control()


# -----------------------------------------
