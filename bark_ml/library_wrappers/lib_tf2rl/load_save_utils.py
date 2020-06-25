import os

def list_files_in_dir(dir_path: str, file_ending: str = '') -> list:
    """
    Lists all files in the given dir ending with the given ending.
    """
    if not os.path.isdir(dir_path):
        raise ValueError(f'Cannot list files in {dir_path}. Not a directory.')
    files = [f for f in os.listdir(
        dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    files = [f for f in files if str(f).endswith(file_ending)]
    files = [os.path.join(dir_path, f) for f in files]
    return files


def list_dirs_in_dir(dir_path: str, include_git: bool = False) -> list:
    """
    Lists all dirs in the given dir.
    """
    if not os.path.isdir(dir_path):
        raise ValueError(f'Cannot list directories in {dir_path}. Not a directory.')
    dirs = [f for f in os.listdir(
        dir_path) if not os.path.isfile(os.path.join(dir_path, f))]
    if not include_git:
        dirs = [d for d in dirs if not d.endswith('.git')]
    dirs = [os.path.join(dir_path, f) for f in dirs]
    return dirs