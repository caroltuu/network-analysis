import os


def initialize_folders():
    # Initialize necessary folders if do not exist:
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    if not os.path.exists('./results'):
        os.makedirs('./results')


def clear_folders():
    # remove files from previous runs:
    os.system("del /S .\\results\*")
    os.system("del /S .\plots\*")
