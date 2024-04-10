

import subprocess
import traceback

from modules.utilities.logging import logging

logger = logging.getLogger("openfoam_case")

def run_command(command_text : str) -> str:
    """
    use subprocess.Popen to run command and return output to string
    """
    # if "\n" in command_text:
    #     command_text = command_text.split("\n")
    #     command_text = [_c.strip() for _c in command_text]

    # else:
    # command_text = command_text.split(" ")
    try:
        proc = subprocess.Popen(command_text, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output_raw, errors_raw = proc.communicate()
        if len(errors_raw) > 1:
            logger.error(errors_raw.decode("ascii"))
        output = output_raw.decode("ascii")
    except Exception as e:
        traceback_text = traceback.format_exc()
        output = None
        logger.error(f"execution of command {command_text} failed:")
        logger.error(traceback_text)
    
    return output