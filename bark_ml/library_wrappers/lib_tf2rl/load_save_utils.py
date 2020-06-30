import os

def list_files_in_dir(dir_path: str, file_ending: str = '') -> list:
    """Lists all files in the given directory ending with the given ending.

    Args:
        dir_path (str): The path to the directory in which to search for files
        file_ending (str, optional): The file ending to filter the found files. Defaults to ''.

    Raises:
        ValueError: If the given path is not a directory

    Returns:
        list: The files in the directory, filtered by the ending. 
    """
    if not os.path.isdir(dir_path):
        raise ValueError(f'Cannot list files in {dir_path}. Not a directory.')
    files = [f for f in os.listdir(
        dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    files = [f for f in files if str(f).endswith(file_ending)]
    files = [os.path.join(dir_path, f) for f in files]
    return files


def list_dirs_in_dir(dir_path: str, include_git: bool = False) -> list:
    """Lists all directories in the given directory.

    Args:
        dir_path (str): The path to the directory in which to search for files
        include_git (bool, optional): Include the .git folder if found. Defaults to False.

    Raises:
        ValueError: If the given path is not a directory

    Returns:
        list: The directories in the directory. 
    """
    if not os.path.isdir(dir_path):
        raise ValueError(f'Cannot list directories in {dir_path}. Not a directory.')
    dirs = [f for f in os.listdir(
        dir_path) if not os.path.isfile(os.path.join(dir_path, f))]
    if not include_git:
        dirs = [d for d in dirs if not d.endswith('.git')]
    dirs = [os.path.join(dir_path, f) for f in dirs]
    return dirs