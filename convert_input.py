import json
import os
import pickle

import numpy as np
import torch

from file_system_storage import FS

files = FS().list_items("data/dov/")[1:]
count = 0
for i in range(0, len(files), 10):
    current_files = files[i:i+10]
    instances = []
    count += 1
    for file in current_files:
        data = FS().get_data(file)
        data = json.loads(data.decode('utf8'))
        cn = data["c_number"]
        edge = data["change_coord"]
        mat = np.array(data["m"]).reshape(data["v"],data["v"])
        instances.append([mat.tolist(), cn, edge])
    data = pickle.dumps(instances)
    FS().upload_data(
        data, os.path.join("dov2", f'batch_{count}.pkl')
    )