from transformers import AutoModel

# python scripts/get_bin.py

model_path = "data/save/adkg_mdkg_train/2026-04-26_00:56:50.274165/final_model"

model = AutoModel.from_pretrained(model_path)

model.save_pretrained(model_path, safe_serialization=False)