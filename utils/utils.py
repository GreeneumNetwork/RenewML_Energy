from typing import Union
import logging
import os

def get_logger(
    script_name: str,
    log_file: Union[str, None] = None,
    stream_handler: bool = True,
) -> logging.getLogger:

    """Initiate the logger to log the progress into a file.

    Args:
    -----
        script_name (str): Name of the scripts outputting the logs.
        log_file (str): Name of the log file.
        stream_handler (bool, optional): Whether or not to show logs in the
            console. Defaults to True.

    Returns:
    --------
        logging.getLogger: Logger object.
    """
    logger = logging.getLogger(name=script_name)
    logger.setLevel(logging.INFO)

    if log_file is not None:
        # create handlers
        f_handler = logging.FileHandler(os.path.join(log_file), mode="w+")
        # create formatters and add it to the handlers
        f_format = logging.Formatter(
            "%(asctime)s:%(name)s->%(funcName)s: %(levelname)s:%(message)s"
        )
        f_handler.setFormatter(f_format)
        # add handlers to the logger
        logger.addHandler(f_handler)

    # display the logs in console
    if stream_handler:
        s_handler = logging.StreamHandler()
        s_format = logging.Formatter("%(levelname)s: %(name)s->%(funcName)s: %(message)s")
        s_handler.setFormatter(s_format)
        logger.addHandler(s_handler)

    return logger