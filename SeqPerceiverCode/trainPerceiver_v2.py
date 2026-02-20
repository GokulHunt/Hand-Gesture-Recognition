import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sequenceClassifier_v2 import Perceiver
from sequenceClassifier_v2 import make_loaderDataset
from sequenceClassifier_v2 import df2paddedTensor

import random

# Function that finds the lowest common factor
def is_prime(num):
    if num <= 1:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

def nth_smallest(numbers, n):
    sorted_list = sorted(numbers)
    if 1 <= n <= len(sorted_list):
        return sorted_list[n - 1]
    else:
        return None  # Return None for out-of-bounds n

def recursive_denom(input, n_smallest):
    if is_prime(input):
        return int(input)
    else:
        remainder = input/2
        temp = []
        temp.append(remainder)
        while remainder%2 == 0 and remainder > 0:
            remainder = remainder/2
            temp.append(remainder)
            
            #print(remainder)
        
        print(temp)
        return int(nth_smallest(temp, n_smallest))

# Function to calculate accuracy
def calculate_accuracy(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            sequences, labels, attn_mask = batch
            sequences = sequences.to(torch.float32).to(device)
            labels = labels.to(device)
            attn_mask = attn_mask.to(torch.float32).to(device)
            outputs = model(sequences, attn_mask)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_and_evaluate(seed_value):
    print(f"Seed Value: {seed_value}")
     # Set your desired seed value
    random.seed(seed_value)

    # Data conversion to padded tensor
    # Output is of size [BATCH_SIZE x SEQUENCE_LENGTH x ARRAY_DIM]
    # Load the dataset
    data_path = r"train_feature_maps_full.csv"
    full_maps_df = pd.read_csv(data_path)

    # Split the dataset into train and test sets (80% train, 20% test)
    #train_maps_df, test_maps_df = train_test_split(full_maps_df, test_size=0.2, random_state=seed_value)
    no_gesture_count = full_maps_df[full_maps_df['gesture_label'] == 'No gesture'].shape[0]
    filtered_df = full_maps_df[full_maps_df['gesture_label'] != 'No gesture']
    # Group by video identifier
    grouped = filtered_df.groupby('video_name')

    # Get unique video IDs
    video_ids = filtered_df['video_name'].unique()

    # Split video IDs into train and test sets
    train_ids, test_ids = train_test_split(video_ids, test_size=0.25, random_state=42)

    # Create train and test DataFrames
    train_maps_df = pd.concat([grouped.get_group(video_id) for video_id in train_ids])
    test_maps_df = pd.concat([grouped.get_group(video_id) for video_id in test_ids])

    batch_size = 32
    num_epochs = 5

    x, labels_list, attention_mask_list = df2paddedTensor(train_maps_df, n_samples_each_label = 1000, seed_value = seed_value)
    x2, labels_list2, attention_mask_list2 = df2paddedTensor(test_maps_df, n_samples_each_label = 200, seed_value = seed_value)

    myLoaderDataset = make_loaderDataset(x, labels_list, attention_mask_list)
    myLoaderDataset2 = make_loaderDataset(x2, labels_list2, attention_mask_list2)

    dataloader      = DataLoader(myLoaderDataset, batch_size=batch_size, shuffle=True)
    dataloader_test      = DataLoader(myLoaderDataset2, batch_size=batch_size, shuffle=True)

    print("Embed dim: ", x.shape[2]*2)
    print("Labels list type: ", type(labels_list))




    embed_dim   = x.shape[2]*2
    trnfr_heads = recursive_denom(embed_dim, 2)
    n_classes   = max(labels_list) + 1


    print("Number of heads, num_heads: ", trnfr_heads)

    model_kwargs = {
        "embed_dim"     : embed_dim,
        "batch_size"    : batch_size,
        "latent_dim"    : 8,
        "attn_mlp_dim"  : 16, 
        "trnfr_mlp_dim" : 16, 
        "trnfr_heads"   : trnfr_heads, # embed_dim must be divisible by num_heads
        "dropout"       : 0.1, 
        "trnfr_layers"  : 6, 
        "n_blocks"      : 6, 
        "n_classes"     : n_classes,
        "learning_rate" : 0.003
    }

    # Create model, loss function and optimizer
    model = Perceiver(model_kwargs)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    best_loss = 100000
    best_epoch = 0
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            # Unpack batch
            sequences, labels, attn_mask = batch
            
            # Convert tensors to the same data type
            sequences = sequences.to(torch.float32)
            attn_mask = attn_mask.to(torch.float32)
            
            sequences = sequences.to(device)
            labels = labels.to(device).long()
            attn_mask = attn_mask.to(device)

            # Forward pass
            outputs = model(sequences, attn_mask)

            # Compute loss
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            #total_grad_norm = 0.0
           # param_count = 0
            #for param in model.parameters():
            #    if param.grad is not None:
            #        total_grad_norm += param.grad.data.norm(2).item()
            #        param_count += 1

            #average_grad_norm = total_grad_norm / param_count if param_count > 0 else 0
            #print(f'Epoch {epoch+1}, Average Gradient Norm: {average_grad_norm}')

            #gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            optimizer.step()
            total_loss += loss.item()
        #train_accuracy = calculate_accuracy(model, dataloader, device)
        #print(f'Epoch {epoch+1}, Loss: {loss.item()}, Train Accuracy: {train_accuracy:.2f}%')
        avg_loss = total_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch+1
            torch.save(model.state_dict(), f'model-{seed_value}-best.pth')
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')


    #model_filename = f"modelv2-{seed_value}.pth"
    #torch.save(model.state_dict(), model_filename)
    print(f"Model saved for Epoch {best_epoch}")



    # Load your test data similar to train data
    # Assume test_data_loader is created similar to dataloader
    model.load_state_dict(torch.load(f'model-{seed_value}-best.pth'))

    # Calculate train accuracy
    train_accuracy = calculate_accuracy(model, dataloader, device)
    test_accuracy = calculate_accuracy(model, dataloader_test, device)
    print(f"Train Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")
    return train_accuracy, test_accuracy