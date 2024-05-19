import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

rows = []

print("started loading cat pictures")
for i in tqdm(range(12500)):
    for animal in ["Cat", "Dog"]:
        if animal == "Cat" and i == 666: # missing image
            continue
        if animal == "Dog" and i == 11702: # missing image
            continue
        row = {}
        row["image_path"] = f"data/PetImages/{animal}/{i}.jpg"
        cat_image = Image.open(row["image_path"])
        row["image_size_x"] = cat_image.size[0]
        row["image_size_y"] = cat_image.size[1]
        cat_image.close()
        row["image_class"] = animal

        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("data/data.csv", index=False)