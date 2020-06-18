import os

def list_files_in_dir(dir_path: str, file_ending: str):
    """
    Lists all files in the given dir ending with the given ending.
    """
    assert os.path.isdir(dir_path)
    files = [f for f in os.listdir(
        dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    files = [f for f in files if str(f).endswith(file_ending)]
    files = [os.path.join(dir_path, f) for f in files]
    return files


def list_dirs_in_dir(dir_path: str):
    """
    Lists all dirs in the given dir.
    """
    assert os.path.isdir(dir_path)
    dirs = [f for f in os.listdir(
        dir_path) if not os.path.isfile(os.path.join(dir_path, f)) and not f == '.git']
    dirs = [os.path.join(dir_path, f) for f in dirs]
    return dirs