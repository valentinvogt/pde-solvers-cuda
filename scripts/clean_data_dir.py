import os
import json

data_dir = "/cluster/scratch/vogtva/data/bruss"
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".json"):
            d = json.load(open(os.path.join(root, file)))
            if "filename" in d:
                # check if file exists
                if not os.path.isfile(d["filename"]):
                    print("File does not exist")
                    