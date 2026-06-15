"""This script de-duplicates the data provided by the VQA-RAD authors,
creates an "imagefolder" dataset and pushes it to the Hugging Face Hub.
"""

import re
import os
import shutil
import datasets
import pandas as pd

# load the data
data = pd.read_json("osfstorage-archive/VQA_RAD Dataset Public.json")

# split the data into training and test
train_data = data[data["phrase_type"].isin(["freeform", "para"])]
test_data = data[data["phrase_type"].isin(["test_freeform", "test_para"])]

# keep only the image-question-answer triplets
train_data = train_data[["image_name", "question", "answer"]]
test_data = test_data[["image_name", "question", "answer"]]

# drop the duplicate image-question-answer triplets
train_data = train_data.drop_duplicates(ignore_index=True)
test_data = test_data.drop_duplicates(ignore_index=True)

# drop the common image-question-answer triplets
train_data = train_data[~train_data.apply(tuple, 1).isin(test_data.apply(tuple, 1))]
train_data = train_data.reset_index(drop=True)

# perform some basic data cleaning/normalization
f = lambda x: re.sub(' +', ' ', str(x).lower()).replace(" ?", "?").strip()
train_data["question"] = train_data["question"].apply(f)
test_data["question"] = test_data["question"].apply(f)
train_data["answer"] = train_data["answer"].apply(f)
test_data["answer"] = test_data["answer"].apply(f)

# copy the images using unique file names
os.makedirs(f"data/train/", exist_ok=True)
train_data.insert(0, "file_name", "")
for i, row in train_data.iterrows():
    file_name = f"img_{i}.jpg"
    train_data["file_name"].iloc[i] = file_name
    shutil.copyfile(src=f"osfstorage-archive/VQA_RAD Image Folder/{row['image_name']}", dst=f"data/train/{file_name}")
_ = train_data.pop("image_name")

os.makedirs(f"data/test/", exist_ok=True)
test_data.insert(0, "file_name", "")
for i, row in test_data.iterrows():
    file_name = f"img_{i}.jpg"
    test_data["file_name"].iloc[i] = file_name
    shutil.copyfile(src=f"osfstorage-archive/VQA_RAD Image Folder/{row['image_name']}", dst=f"data/test/{file_name}")
_ = test_data.pop("image_name")

# save the metadata
train_data.to_csv(f"data/train/metadata.csv", index=False)
test_data.to_csv(f"data/test/metadata.csv", index=False)

# push the dataset to the hub
dataset = datasets.load_dataset("imagefolder", data_dir="data/")
dataset.push_to_hub("flaviagiammarino/vqa-rad")
