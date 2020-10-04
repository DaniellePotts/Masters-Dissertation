import pickle

def load_data(path):
  with open(path, 'rb') as handle:
    return pickle.load(handle)

def save_data(file_name, obj):
  pickle.dump(obj, open(file_name, 'wb'))