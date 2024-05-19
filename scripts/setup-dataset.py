import huggingface_hub as hfh
import os, zipfile


data_path = hfh.hf_hub_download("osbm/cats-dogs", "data.zip", repo_type="dataset")

with zipfile.ZipFile(data_path, 'r') as zip_ref:
    zip_ref.extractall("data")


print(data_path)
