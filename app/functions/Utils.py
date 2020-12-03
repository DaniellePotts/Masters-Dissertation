import pickle
import  json

def load_data(path):
  with open(path, 'rb') as handle:
    return pickle.load(handle)

def save_data(file_name, obj):
  pickle.dump(obj, open(file_name, 'wb'))

def write_json_file(file, data):
    with open(file, 'w') as f:
        json.dump(data, f)