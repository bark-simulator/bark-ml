import os
from pathlib import Path

def list_files_in_dir(dir_path: str, file_ending: str = '') -> list:
    """Lists all files in the given directory ending with the given ending.

    Args:
        dir_path (str): The path to the directory in which to search for files
        file_ending (str, optional): The file ending to filter the found files. Defaults to ''.

    Raises:
        NotADirectoryError: If the given path is not a directory

    Returns:
        list: The files in the directory, filtered by the ending. 
    """
    entries = Path(os.path.expanduser(dir_path))
    files = [str(entry.absolute()) for entry in entries.iterdir() if entry.is_file()]
    files = [f for f in files if str(f).endswith(file_ending)]
    return files


def list_dirs_in_dir(dir_path: str, include_git: bool = False) -> list:
    """Lists all directories in the given directory.

    Args:
        dir_path (str): The path to the directory in which to search for files
        include_git (bool, optional): Include the .git folder if found. Defaults to False.

    Raises:
        NotADirectoryError: If the given path is not a directory

    Returns:
        list: The directories in the directory. 
    """
    entries = Path(os.path.expanduser(dir_path))
    files = [str(entry.absolute()) for entry in entries.iterdir() if not entry.is_file()]
    if not include_git:
        files = [f for f in files if not str(f).endswith('.git')]
    return files