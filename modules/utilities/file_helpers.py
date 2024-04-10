


def modify_file(source_file, destination_file, placeholders, replacements, logger = None):
    """
    opens a file, reads the content, replace placeholders, saves new file
    """
    if logger == None:
        def _log(_message): 
            print(_message)
    else:
        def _log(_message): 
            logger.info(_message)

    with open(source_file, "r") as f_source:
        file_text = f_source.read()
    f_source.close()

    with open(destination_file, "w") as f:
        for placeholder, replacement in zip(placeholders, replacements):
            file_text = file_text.replace(placeholder, str(replacement))
        f.write(file_text)
    f.close()
    _log("created decomposeParDict file for parallel run")



def safe_path(path : str) -> str:
    """
    makes safe path
    """
    

    return path