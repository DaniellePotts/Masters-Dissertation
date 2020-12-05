import pickle
import  json

#Load data file
def load_data(path):
  with open(path, 'rb') as handle:
    return pickle.load(handle)

#Save data file
def save_data(file_name, obj):
  pickle.dump(obj, open(file_name, 'wb'))

#Write to a JSON file
def write_json_file(file, data):
    with open(file, 'w') as f:
        json.dump(data, f)