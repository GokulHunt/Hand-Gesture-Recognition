from sequenceClassifier_v2 import df2paddedTensor
import pandas as pd
import torch
from torch.utils.data import DataLoader

# =================================================================================
# Code execution
# =================================================================================
torch.manual_seed(42)

# Data conversion to padded tensor
# Output is of size [BATCH_SIZE x SEQUENCE_LENGTH x ARRAY_DIM]
data_path = r"train_feature_maps_full.csv"
train_maps_df = pd.read_csv(data_path)

x, labels_list, attention_mask_list = df2paddedTensor(train_maps_df, n_samples_each_label = 50)

print("Dimension of x: ", x.shape)
print("First element of x: ", x[0,:,0])
print("First element of labels_list: ", labels_list[0:5])
print("First few elements of attention_mask: ", attention_mask_list[0])
print("Num of nonzeros in queried sequence tensor: ", torch.sum(x[0,:,0] != 0) )
print("Num of ones in quereid attention mask: ", torch.sum(attention_mask_list[0]))

# Specify input data to model
model_input = x

model_kwargs = {
    "embed_dim"     : model_input[2]*2,
    "batch_size"    : model_input[0],
    "latent_dim"    :8,
    "attn_mlp_dim"  :16, 
    "trnfr_mlp_dim" :16, 
    "trnfr_heads"   :8, 
    "dropout"       :0.1, 
    "trnfr_layers"  :6, 
    "n_blocks"      :6, 
    "n_classes"     :10,
    "learning_rate" :0.003
}

# Apply batch_loader
from sequenceClassifier_v2 import make_loaderDataset

myLoaderDataset = make_loaderDataset(x, labels_list, attention_mask_list)

# Define batch size and other loader parameters
batch_size = 4
shuffle = True  # Shuffle the data during training if needed

# Create the DataLoader
data_loader = DataLoader(myLoaderDataset, batch_size=batch_size, shuffle=shuffle)

# Get a single batch from the DataLoader
sample_batch = next(iter(data_loader))

# Check the contents and structure of the batch
# Unpack
# Unpack the batch
input_data, labels, attention_masks = sample_batch

# print("Data loader sample batch: ", sample_batch)

print("sample tensor shape: ", input_data.shape)
print("sample labels: ", labels)
print("sample attn_mask: ", attention_masks)

print("Input data type: ", type(input_data))
print("label type: ", type(labels))
print("attention_mask type:", type(attention_masks))