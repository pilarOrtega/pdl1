import pickle
import os

def pickle_save(file, path, name):
    file_path = os.path.join(path, name)
    with open(file_path, "wb") as f:
        pickle.dump(file, f)

def pickle_load(file_name):
    with open(file_name, "rb") as f:
        file = pickle.load(f)
        print('Document ' + file_name + ' correctly loaded')
        print()
    return file
