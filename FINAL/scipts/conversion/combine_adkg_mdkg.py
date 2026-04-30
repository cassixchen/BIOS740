import json
import os

# python ./scripts/conversion/combine_adkg_mdkg.py

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# load
adkg_train = load_json("data/datasets/adkg/adkg_train.json")
adkg_dev   = load_json("data/datasets/adkg/adkg_dev.json")
mdkg_train = load_json("data/datasets/mdkg/mdkg_train.json")
mdkg_dev   = load_json("data/datasets/mdkg/mdkg_dev.json")
adkg_types = load_json("data/datasets/adkg/adkg_types.json")
mdkg_types = load_json("data/datasets/mdkg/mdkg_types.json")

# combine
adkg_mdkg_train = adkg_train + mdkg_train
adkg_mdkg_dev   = adkg_dev + mdkg_dev

adkg_mdkg_types = {
    "entities": {},
    "relations": {}
}
adkg_mdkg_types["entities"].update(adkg_types["entities"])
adkg_mdkg_types["entities"].update(mdkg_types["entities"])
adkg_mdkg_types["relations"].update(adkg_types["relations"])
adkg_mdkg_types["relations"].update(mdkg_types["relations"])

# save
save_json(adkg_mdkg_train, os.path.join("data/datasets/adkg_mdkg", "adkg_mdkg_train.json"))
save_json(adkg_mdkg_dev,   os.path.join("data/datasets/adkg_mdkg", "adkg_mdkg_dev.json"))
save_json(adkg_mdkg_types, os.path.join("data/datasets/adkg_mdkg", "adkg_mdkg_types.json"))