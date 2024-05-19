from PIL import Image
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

print("started checking cat pictures")
for i in tqdm(range(12500)):
    if i == 666: # missing image
        continue
    cat_image = Image.open(f"data/PetImages/Cat/{i}.jpg")
    assert cat_image.size[0] <= 500
    assert cat_image.size[1] <= 500
    cat_array = np.array(cat_image)
    assert cat_array.dtype == np.uint8
    assert cat_array.min() >= 0
    assert cat_array.max() <= 255
    del cat_array
    cat_image.close()
print("everything is well")

print(r'''
   \    /\
    )  ( ')
    (  /  )
     \(__)|''')

print("started checking dog pictures")
for i in tqdm(range(12500)):
    if i == 11702: # missing image
        continue
    print
    dog_image = Image.open(f"data/PetImages/Dog/{i}.jpg")
    assert dog_image.size[0] <= 500
    assert dog_image.size[1] <= 500
    dog_array = np.array(dog_image)
    assert dog_array.dtype == np.uint8
    assert dog_array.min() >= 0
    assert dog_array.max() <= 255
    del dog_array
    dog_image.close()
print("everything is well")

print(r"""
    ^..^      /
    /_/\_____/
        /\   /\
       /  \ /  \ """)
    