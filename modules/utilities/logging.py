# type hinting
from logging import Logger

# python module imports
import logging
from datetime import datetime
from modules.utilities.file_helpers import safe_path
import os
import shutil


def init_logger(logger_name: str="openfoam_run", debug: bool=False) -> Logger:
    """
    Initializes a logger that writes in the console and into info and error log files.
    """

    # create logger
    now = datetime.now().strftime("%d.%m.%Y-%H.%M.%S")
    logfilename = os.path.join("logs", f"{logger_name}_{str(now)}.log")
    logfilename_errors = os.path.join("logs", f"{logger_name}_{str(now)}.ERRORS.log")
    logfilename = safe_path(logfilename)
    logfilename_errors = safe_path(logfilename_errors)
    logger = logging.getLogger(logger_name)

    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # create folder for logs
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # create console handler and set level to error
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create file handler for full debug messages
    fh = logging.FileHandler(logfilename)
    fh.setLevel(logging.DEBUG)

    # create file handler for warnings and errors
    fh_errors = logging.FileHandler(logfilename_errors)
    fh_errors.setLevel(logging.WARNING)

    # create formatter
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(funcName)s - line %(lineno)3d --- %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    fh_errors.setFormatter(formatter)

    # add handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.addHandler(fh_errors)

    return logger


def close_logger(logger_name: str="openfoam_run") -> None:
    logger = logging.getLogger(logger_name)
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


def copy_log_file_to(destination: str, logger_name: str="openfoam_run", replace_logger_name_with_str: str=None) -> None:
    """
    Copies the current log files to a specified directory.
    """

    logger = logging.getLogger(logger_name)

    logger = logging.getLogger(logger_name)
    for handler in logger.handlers:
        try:
            filename = handler.baseFilename
            if replace_logger_name_with_str is not None:
                full_destination_path = shutil.copy(filename, destination)
                new_filename = full_destination_path.replace(logger_name, replace_logger_name_with_str)
                os.rename(full_destination_path, new_filename)
            else:
                shutil.copy(filename, destination)

        except:
            pass
