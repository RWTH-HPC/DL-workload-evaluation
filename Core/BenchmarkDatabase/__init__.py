import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')) + '/'
runtime_filepath = os.path.abspath(os.path.join(root_path, 'Runtimefiles')) + '/'
if not os.path.isdir(runtime_filepath):
    os.mkdir(runtime_filepath)