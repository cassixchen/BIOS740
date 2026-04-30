import json
import os
import spacy
from tqdm import tqdm

# python ./scripts/conversion/convert_adkg_mdkg.py

nlp = spacy.load("en_core_web_sm")


def char_to_token_span(doc, char_start, char_end):
    start_token = None
    end_token = None

    for token in doc:
        if start_token is None and token.idx <= char_start < token.idx + len(token):
            start_token = token.i

        if token.idx < char_end <= token.idx + len(token):
            end_token = token.i + 1

    return start_token, end_token


def convert_split(data_split, dataset_name, split):
    converted = []

    counter = 1
    for example in tqdm(data_split, desc=f"{dataset_name}-{split}"):
        text = example["text"]
        doc = nlp(text)

        tokens = [token.text for token in doc]

        # entities
        entities = []
        id_map = {}

        for idx, ent in enumerate(example["entities"]):
            start_char = ent["start"]
            end_char = ent["end"]

            start_tok, end_tok = char_to_token_span(doc, start_char, end_char)

            entities.append({
                "start": start_tok,
                "end": end_tok,
                "type": ent["type"]
            })

            id_map[ent["id"]] = idx

        # relations
        relations = []
        for rel in example["relations"]:

            relations.append({
                "type": rel["type"],
                "head": id_map[rel["head"]["id"]],
                "tail": id_map[rel["tail"]["id"]]
            })
            
        converted.append({
            "tokens": tokens,
            "entities": entities,
            "relations": relations,
            "orig_id": example["doc_id"]
        })
        
    return converted


def convert_dataset(dataset_name):
    input_file = f"data/datasets/{dataset_name}/{dataset_name}.json"

    with open(input_file, "r") as f:
        data = json.load(f)

    out_dir = f"data/datasets/{dataset_name}"

    for split in ["train", "dev", "test"]:
        converted = convert_split(data[split], dataset_name, split)

        out_path = os.path.join(out_dir, f"{dataset_name}_{split}.json")

        with open(out_path, "w") as f:
            json.dump(converted, f, indent=2)


def main():
    datasets = ["adkg", "mdkg"]

    for name in datasets:
        convert_dataset(name)


if __name__ == "__main__":
    main()