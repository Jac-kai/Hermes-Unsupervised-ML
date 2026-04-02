# -------------------- Imported Modules -------------------
import logging
import os


# -------------------- Logging Setup --------------------
def hermes_init_logging() -> logging.Logger:
    """
    Initialize and return the Hermes project logger.

    This function configures the shared ``Hermes`` logger for the project,
    ensures that the Hermes log folder exists, and attaches both a file
    handler and a console stream handler when they have not already been
    added.

    The file log is written to ``Hermes_Logs/Hermes_Log.log`` under the
    current project directory. Repeated calls avoid duplicating handlers by
    checking existing logger handlers before adding new ones.

    Returns
    -------
    logging.Logger
        Configured logger instance named ``"Hermes"``.

    Side Effects
    ------------
    - Creates the ``Hermes_Logs`` folder if it does not already exist.
    - Creates or appends to the log file ``Hermes_Log.log``.
    - Adds a ``logging.FileHandler`` for file logging if missing.
    - Adds a ``logging.StreamHandler`` for console output if missing.
    - Logs an initialization message containing the resolved log-file path.

    Notes
    -----
    The logger level is set to ``logging.INFO`` and propagation is disabled
    so Hermes log messages are handled only by the configured Hermes logger
    handlers unless the implementation is changed later.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    log_folder = os.path.join(project_root, "Hermes_Logs")
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, "Hermes_Log.log")

    logger = logging.getLogger("Hermes")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", "") == log_file
        for h in logger.handlers
    ):
        fh = logging.FileHandler(log_file, encoding="utf-8-sig")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    ):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    logger.info("Logging initialized to: %s", log_file)
    return logger


# -------------------- Execute --------------------
if __name__ == "__main__":
    logger = hermes_init_logging()


# --------------------------------------------------------
