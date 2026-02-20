import torch
import torch.nn as nn

import numpy as np


# NOTE: Function that applies position embedding to a batch of a sequence of 
#       1-D feature arrays.
class PositionalFeatureArrayEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def positional_encoding(self, x):
        # Initial x of shape [BATCH_SIZE x SEQUENCE_LENGTH x ARRAY DIMENSION]
        # Assume each sequence already padded

        pos_encodings = torch.zeros((x.shape[1], x.shape[2]))

        # For each column
        for col in range(x.shape[2]):
            # If col is even valued apply sin
            if col % 2 == 0:
                pos_encodings[:, col] = torch.sin(torch.arange(x.shape[1]) / (10000 ** (2 * col / x.shape[2])))
            else:
                pos_encodings[:, col] = torch.cos(torch.arange(x.shape[1]) / (10000 ** (2 * col / x.shape[2])))

        return pos_encodings # This would only be the positional encoding for one batch

    def forward(self, x):
        # Initial x of shape [BATCH_SIZE x SEQUENCE_LENGTH x ARRAY DIMENSION]
        # Assume each sequence already padded

        # Positional encoding formula:
        # PE(pos, 2i)      = sin(pos/10000**(2*i/d_model))
        # PE(pos, 2_(i+1)) = cos(pos/10000**(2*i/d_model))

        # Generate tensor of positional encodings based on input
        # We only need to create one instance of this and then apply it wholesale
        # to the entire input tensor x along the BATCH_SIZE dimension

        # Generate positional encodings
        pos_encodings = self.positional_encoding(x)

        # Concatenate pos_encodings to every batch of x
        # Create a new tensor to store the values
        new_x = []
        for batch in range(x.shape[0]):
            new_x.append(torch.cat((x[batch], pos_encodings), dim=1))

        # new_x is of dimension BATCH_SIZE x SEQ_LENGTH x EMBED_DIM
        # Need it to be in SEQ_LENGTH x BATCH_SIZE x EMBED_DIM

        new_x = torch.stack(new_x).permute(1, 0, 2)

        return new_x

    

# Christopher Cheong notes: 
# Adapted from: https://medium.com/@curttigges/the-annotated-perceiver-74752113eefb

from transformers import PerceiverModel # CC: Import of PerceiverModel works in Colab.

class LatentTransformer(nn.Module):
    """Latent transformer module with n_layers count of decoders.
    """
    def __init__(self, embed_dim, mlp_dim, n_heads, dropout, n_layers):
        super().__init__()
        self.transformer = nn.ModuleList([
            PerceiverAttention(
                embed_dim=embed_dim, 
                mlp_dim=mlp_dim, 
                n_heads=n_heads, 
                dropout=dropout) 
            for l in range(n_layers)])

    def forward(self, l):
        
        for trnfr in self.transformer: # CC [002]: These are all PerceiverAttention objects
            l = trnfr(l, l)
        
        return l


# ==============================================================================
# CC: Not modified
class PerceiverBlock(nn.Module):
    """Block consisting of one cross-attention layer and one latent transformer"""
    def __init__(self, embed_dim, attn_mlp_dim, trnfr_mlp_dim, trnfr_heads, dropout, trnfr_layers):
        super().__init__()
        
        self.cross_attention = PerceiverAttention(
            embed_dim, attn_mlp_dim, n_heads=1, dropout=dropout)
        self.lnorm1 = nn.LayerNorm(embed_dim)  # Layer norm for the first skip connection

        self.latent_transformer = LatentTransformer(
            embed_dim, trnfr_mlp_dim, trnfr_heads, dropout, trnfr_layers)
        self.lnorm2 = nn.LayerNorm(embed_dim)  # Layer norm for the second skip connection

    def forward(self, x, l, attention_mask=None):
        # Cross-attention with skip connection
        cross_attn_output = self.cross_attention(x, l, attention_mask)
        l = self.lnorm1(l + cross_attn_output)  # Add & normalize

        # Latent transformer with skip connection
        latent_transformer_output = self.latent_transformer(l)
        l = self.lnorm2(l + latent_transformer_output)  # Add & normalize

        return l

# ==============================================================================
# CC: Not modified
class Classifier(nn.Module):
    """Original Perceiver classification calculation
    """
    def __init__(self, embed_dim, latent_dim, batch_size, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        # latent, batch, embed

        x = self.fc1(x)
        x = x.mean(dim=0)
        x = self.fc2(x)

        return x

# ==============================================================================
# CC[001]: Modified to add attention mask
class PerceiverAttention(nn.Module):
    """Basic decoder block used both for cross-attention and the latent transformer
    """
    def __init__(self, embed_dim, mlp_dim, n_heads, dropout=0.0):
        super().__init__()

        self.lnorm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads)

        self.lnorm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, q, attention_mask=None):
        # x will be of shape [PIXELS x BATCH_SIZE x EMBED_DIM]
        # q will be of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM] when this is
        # used for cross-attention; otherwise same as x

        # CC note [001]: Added attention mask layer since padding was used
        # Apply attention_mask if provided
        # if attention_mask is not None:
            # Expand the attention mask if needed
            # attention_mask = attention_mask.unsqueeze(0)  # Adjust the dimension if needed #TODO
        
        # attention block
        # print("In PerceiverAttention, type of x: ", type(x) )
        
        out = self.lnorm1(x)
        out, _ = self.attn(query=q, key=x, value=x, key_padding_mask=attention_mask)
        # out will be of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM] after matmul
        # when used for cross-attention; otherwise same as x
        
        # first residual connection
        resid = out + q

        # dense block
        out = self.lnorm2(resid)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.drop(out)

        # second residual connection
        out = out + resid

        return out

# ==============================================================================
# CC: Modified for input feature array.

class Perceiver(nn.Module):
    """Complete original Perceiver, without weight sharing
    """
    def __init__( 
        self, margs):
        # Unpack parameters
        embed_dim     = margs["embed_dim"]
        batch_size    = margs["batch_size"]
        latent_dim    = margs["latent_dim"]
        attn_mlp_dim  = margs["attn_mlp_dim"]
        trnfr_mlp_dim = margs["trnfr_mlp_dim"]
        trnfr_heads   = margs["trnfr_heads"]
        dropout       = margs["dropout"]
        trnfr_layers  = margs["trnfr_layers"]
        n_blocks      = margs["n_blocks"]
        n_classes     = margs["n_classes"]

        # Get initializations from the superclass
        super().__init__()
        
        # ----------------------------------------------------------------------

        # Initialize latent array
        # NOTE: Array of learnable parameters
        self.latent = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.zeros((latent_dim, 1, embed_dim)), 
                mean=0, 
                std=0.02, 
                a=-2, 
                b=2))
        # In the paper, a truncated normal distribution was used for initialization, 
        # so this hidden torch function was used to create it.


        # Initialize embedding with position encoding
        self.embed = PositionalFeatureArrayEmbedding() # CC: changed to own positional embedding function.

        # Initialize arbitrary number of Perceiver blocks
        self.perceiver_blocks = nn.ModuleList([
            PerceiverBlock(
                embed_dim=embed_dim, 
                attn_mlp_dim=attn_mlp_dim, 
                trnfr_mlp_dim=trnfr_mlp_dim, 
                trnfr_heads=trnfr_heads, 
                dropout = dropout, 
                trnfr_layers = trnfr_layers)
            for b in range(n_blocks)])

        # Initialize classification layer
        self.classifier = Classifier(embed_dim=embed_dim, latent_dim=latent_dim, batch_size=batch_size, n_classes=n_classes)

    def forward(self, x, attention_mask = None):
        # First we expand our latent query matrix to size of batch
        batch_size = x.shape[0]
        latent = self.latent.expand(-1, batch_size, -1)

        # Next, we pass the sequence through the embedding module to get flattened input
        x = self.embed(x) 

        # Next, we iteratively pass the latent matrix and image embedding through
        # perceiver blocks
        for pb in self.perceiver_blocks:
            latent = pb(x, latent, attention_mask)
        #print(latent.shape)
    
        # Finally, we project the output to the number of target classes
        latent = self.classifier(latent)

        return latent
      
      
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


# CC Note [001]: Probably need to include code that creates an attention mask
#                when creating the padded sequence.
def df2paddedTensor(df,n_samples_each_label, seed_value = 42):
    train_maps_df = df

    # Function that samples n number of groups from the dataset
    def sample_n_rows(group, n, seed_value):
        import numpy as np

        # Set the seed for NumPy (used by Pandas for random number generation)
        np.random.seed(seed_value)
      
        # Sample n unique 'video_name'
        sampled_video_names = group['video_name'].drop_duplicates().sample(n)
        
        # Return all rows that share those 'video_name'
        return group[group['video_name'].isin(sampled_video_names)]

    # Group the DataFrame by 'gesture_label' and apply the function
    # df = train_maps_df.groupby('gesture_label').apply(sample_n_rows).reset_index(drop=True)

    # Group the DataFrame by 'gesture_label' and apply the function
    df = train_maps_df.groupby('gesture_label').apply(lambda group: \
      sample_n_rows(group, n=n_samples_each_label, seed_value = seed_value)).reset_index(drop=True)

    # Remove column 0 and column 2
    print("Size of df: ", df.shape)

    # Assume df is your DataFrame and 'label' is your column of labels
    labels = df['gesture_label']

    # Initialize a LabelEncoder
    le = LabelEncoder()

    # Fit the encoder to the labels and transform the labels to numerical
    labels = le.fit_transform(labels)

    df = df.drop(df.columns[[1,2,3,4]], axis=1)

    # append labels to end of df
    df['gesture_labels'] = labels

    #-------------------------------------------
    # Now slice the df into a list containing elements of rows with unique video_names
    # Do not include 'video_name' columns in the rows.
    # Group the DataFrame by 'video_name'
    grouped = df.groupby('video_name')

    # Initialize an empty list to store the DataFrames
    dataframes = []

    # For each group in grouped
    for name, group in grouped:
        # Append the group to dataframes
        dataframes.append(group)

    # Now dataframes is a list containing the DataFrames for each unique 'video_name'
    #-------------------------------------------

    # NOTE: Keep "video_name" in the df for verification purposes.

    # print(dataframes[0].head(5))

    # Determine the maximum sequence length
    max_row_length = df['video_name'].value_counts().max()

    #----------------------------------------------------------------------------------------- Padded sequence creation [START]
    # Initialize an empty list to store the padded sequences
    padded_sequences = []
    attention_mask_list  = [] # List of 

    # Each element in the list "sequences" is a dataframe, pad out the dataframes
    # with zeros, for the "video_name" and "gesture_labels" just replicate the first row entry.

    # CC note [001]: Adding the attention mask creation here as well
    for ii in range(len(dataframes)):
        # Check maximum dimension < max_row_length
        if dataframes[ii].shape[0] < max_row_length:
            # Create attention mask tensor:
            mask_tensor1 = torch.ones(len(dataframes[ii]))
            mask_tensor0 = torch.zeros(max_row_length - len(dataframes[ii]))
            attention_mask_list.append(torch.cat((mask_tensor1,mask_tensor0), dim = 0))

            # Pad out the dataframe with zeros until the rows reach max_row_length
            padding_df = pd.DataFrame(0, index=range(max_row_length - len(dataframes[ii])), \
              columns=dataframes[ii].columns)
            
            # For each padded row of "video_name" and "gesture_labels" just replicate
            # the first row entry of "video_name" and "gesture_labels", respectively.
            padding_df['video_name'] = dataframes[ii]['video_name'].iloc[0]
            padding_df['gesture_labels'] = dataframes[ii]['gesture_labels'].iloc[0]
            
            # Concatenate the dataframe and the padding_df
            padded_sequences.append(pd.concat([dataframes[ii], padding_df]))
        else:
            # Create and append attention mask
            mask_tensor = torch.ones(max_row_length)

            attention_mask_list.append(mask_tensor)

            # Add sequence of sufficient length directly to padded_sequences list.
            padded_sequences.append(dataframes[ii])


    # Now padded_sequences is a list containing the padded DataFrames 
    # for each unique 'video_name'.
    # print(padded_sequences[0].head(5))

    # For each DataFrame in padded_sequences
    for df in padded_sequences:
        # Check if the number of rows is equal to max_seq_length
        if df.shape[0] != max_row_length:
            print("Padding was not done correctly for a DataFrame.")
            break
    else:
        print("Padding was done correctly for all DataFrames.")

    # Remove the "video_name" columns for all dataframe in dataframes AND split the
    # "gesture_labels" columns into their own list corresponding to the dataframes list.
    #----------------------------------------------------------------------------------------- Padded sequence creation [END]
    # Initialize an empty list to store the labels
    labels_list = []

    # For each DataFrame in padded_sequences
    for i in range(len(padded_sequences)):
        # Split the 'gesture_labels' column into its own list
        labels_list.append(padded_sequences[i]['gesture_labels'].iloc[0]) #----------------------- CC updated [001]: [0] because we only need 1 label to represent each batch.
        
        # Remove the 'video_name' column
        padded_sequences[i] = padded_sequences[i].drop(columns=['video_name'])

    # Now padded_sequences is a list containing the padded DataFrames without the 'video_name' column
    # and labels_list is a list containing the 'gesture_labels' for each DataFrame

    # Finally, we convert the padded_sequences list into a tensor x of dimension:
    #                        [len(padded_sequences) x df_row x df_col]
    # Convert each DataFrame in padded_sequences to a tensor and stack them along a new dimension
    x = torch.stack([torch.tensor(df.values) for df in padded_sequences])

    # Now x is a tensor of shape [len(padded_sequences) x df_row x df_col]

    print("Dimension of x: ", x.shape)
    print("Max row length (Max sequence length for padding):", max_row_length)
    print("Expected dimension format: BATCH x SEQ_LEN x ARRAY_DIM")

    return x, labels_list, attention_mask_list # CC updated [001]: Now the labels_list should only contain one integer per batch entry


# Format the dataset which is a [BATCH_SIZE x SEQUENCE_LENGTH x ARRAY_DIMENSION]
# tensor into a format suitable for dataloaders
from torch.utils.data import Dataset

class make_loaderDataset(Dataset):
    def __init__(self, data, labels, attn_mask):
        self.data = data
        self.labels = labels
        self.attn_mask = attn_mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.attn_mask[idx]






