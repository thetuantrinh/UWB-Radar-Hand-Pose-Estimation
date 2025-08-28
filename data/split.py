import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

root_dir = "./datasets"
csv_label_path = root_dir + "annotations.csv"
annotations = pd.read_csv(csv_label_path).fillna(0)
print(f"lenfth of the dataset is {annotations.shape}")

train_ratio = 0.8
shuffle = True
seed = 42

dataset_size = len(annotations)
indices = np.random.permutation(dataset_size)
# indices = list(range(dataset_size))
split = int(np.floor(train_ratio * dataset_size))
if shuffle:
    np.random.seed(seed=seed)
    np.random.shuffle(indices)

train_indices, val_indices = indices[:split], indices[split:]
print(f"train length {train_indices.shape} val length {val_indices.shape}")

print(annotations.head())
train, val = train_test_split(annotations, test_size=0.2)
print(train.shape)
print(val.shape)

train_filenames = train['filename']
val_filenames = val['filename']
train_no_hand = 0
val_no_hand = 0

print("ahah")
# print(f'no-hand {train_no_hand} val-no-hand {val_no_hand} {val_no_hand / train_no_hand}')

# print(f'no-hand {train_no_hand} val-no-hand {val_no_hand} {val_no_hand / train_no_hand}')

train_csv_file = root_dir + "train.csv"
val_csv_file = root_dir + "val.csv"

train.to_csv(train_csv_file, index=False)
val.to_csv(val_csv_file, index=False)
