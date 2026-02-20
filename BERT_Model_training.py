#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install seaborn


# In[30]:


import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import seaborn as sns
import time
import matplotlib.pyplot as plt


# In[31]:


# Load the dataset
input_path = "val_feature_maps_1.csv"
data = pd.read_csv(input_path)

# Isolate hand landmarks and group by video ID and frame number
# Assuming columns after 'frame_number' are hand landmark features

# Preprocess the data: Group by video and frame, and reshape landmarks into a single feature vector per frame
grouped_data = data.groupby(['video_name', 'frame_number']).apply(lambda x: x.iloc[:, 2:].values.flatten())

# Split the data into train test sets
train_ids, test_ids = train_test_split(grouped_data.reset_index()['video_name'].unique(), test_size=0.2, random_state=42)

train_grouped = pd.DataFrame()
#count = 0
for train_id in train_ids:
    rows = grouped_data.reset_index()[grouped_data.reset_index()['video_name']==train_id]
    train_grouped = pd.concat([train_grouped,rows])
    #print(len(train_grouped),len(rows),train_id)
    
test_grouped = pd.DataFrame()
for test_id in test_ids:
    rows = grouped_data.reset_index()[grouped_data.reset_index()['video_name']==test_id]
    test_grouped = pd.concat([test_grouped,rows])


# In[32]:


sns.heatmap(data.drop(['video_name','gesture_label','frame_number'],axis=1).corr())
plt.title("Correlation of hand landmark features")
plt.show()


# In[10]:


data.drop(['video_name','gesture_label','frame_number'],axis=1).corr()


# In[13]:


train_grouped_data = list(train_grouped[0])
test_grouped_data = list(test_grouped[0])

# Dimensionality reduction (training)
pca = PCA(n_components = 30)
train_reduced = pca.fit_transform(train_grouped_data)
test_reduced = pca.transform(test_grouped_data)

# Standardize features (training)
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(list(train_grouped[0]))

# Standardize features (testing)
test_features_scaled = scaler.transform(list(test_grouped[0]))

# # Clustering
# n_clusters = 1000  # Adjust based on your dataset
# kmeans = KMeans(n_clusters=n_clusters)
# kmeans.fit(train_grouped_data)
# tokenized_frames = kmeans.predict(train_grouped_data)

# DB-Scan clustering
dbscan = DBSCAN(eps=1e-2, min_samples=1)
dbscan.fit(train_features_scaled)
tokenized_frames = dbscan.labels_

# K Nearest neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # Adjust the number of neighbors
knn_classifier.fit(train_grouped_data, tokenized_frames)
test_cluster_ids = knn_classifier.predict(test_grouped_data)

# Convert tokenized_frames into sequences per video
train_sequences_wp = train_grouped.groupby('video_name')['frame_number'].apply(list).to_dict()
test_sequences_wp = test_grouped.groupby('video_name')['frame_number'].apply(list).to_dict()
for video_id in train_sequences_wp.keys():
    train_sequences_wp[video_id] = [tokenized_frames[i] for i in train_sequences_wp[video_id]]
for video_id in test_sequences_wp.keys():
    test_sequences_wp[video_id] = [test_cluster_ids[i] for i in test_sequences_wp[video_id]]


# In[14]:


import numpy as np
count = np.zeros(50)
for item in train_sequences_wp:
    for ind in train_sequences_wp[item]:
        count[ind] += 1
count


# In[16]:


# Pad sequences for BERT input
max_seq_length = 512  # Adjust as needed
train_sequences_d = {k: v + [0]*(max_seq_length - len(v)) if len(v) < max_seq_length else v[:max_seq_length] for k, v in train_sequences_wp.items()}
test_sequences_d = {k: v + [0]*(max_seq_length - len(v)) if len(v) < max_seq_length else v[:max_seq_length] for k, v in test_sequences_wp.items()}

# Prepare labels (assuming they are in the original data)
labels = data.groupby('video_name')['gesture_label'].first().astype('category').cat.codes


# In[17]:


# Split data

train_sequences = [train_sequences_d[id] for id in train_ids]
test_sequences = [test_sequences_d[id] for id in test_ids]
train_labels = [labels[id] for id in train_ids]
test_labels = [labels[id] for id in test_ids]

# Function to create attention masks
def create_attention_mask(sequences):
    attention_masks = []
    for sequence in sequences:
        # Create a mask where non-padding tokens are set to 1 and padding tokens to 0
        attention_mask = [1 if token != 0 else 0 for token in sequence]
        attention_masks.append(attention_mask)
    return torch.tensor(attention_masks)

# Create attention masks for training and testing sequences
train_attention_masks = create_attention_mask(train_sequences)
test_attention_masks = create_attention_mask(test_sequences)


# Create DataLoaders
batch_size = 64
train_dataset = TensorDataset(torch.tensor(train_sequences), train_attention_masks, torch.tensor(train_labels))
test_dataset = TensorDataset(torch.tensor(test_sequences), test_attention_masks, torch.tensor(test_labels))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model, optimizer, and loss function
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(labels.unique()))
optimizer = torch.optim.Adam(model.parameters(),lr=0.00018)
criterion = nn.CrossEntropyLoss()


# In[18]:


# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    gpu_count = torch.cuda.device_count()

    print(f"CUDA is available with {gpu_count} GPU(s).")
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Training will be done on CPU.")


# In[19]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# In[20]:


# Evaluate model
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for features, attention_mask, labels in data_loader:
            # Move data to GPU
            features, attention_mask, labels = features.to(device), attention_mask.to(device), labels.to(device)
            output = model(features, attention_mask=attention_mask)
            predictions = output.__dict__['logits']
            loss = criterion(predictions, labels.long())
            total_loss += loss.item()
            total_correct += (predictions.argmax(dim=1) == labels).sum().item()
            all_labels.extend(labels.tolist())
            all_predictions.extend(torch.argmax(predictions, dim=1).tolist())

    # Calculate accuracy
    accuracy = total_correct / len(data_loader.dataset)

    # Calculate F1-score
    f1 = f1_score(all_labels, all_predictions, average="weighted")

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    return total_loss / len(data_loader), accuracy, f1, cm


# In[24]:


# Enable mixed precision training
start_time = time.time()
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Training loop
epochs = 100
total_train_loss = 0
total_correct = 0
all_labels = []
all_predictions = []

for epoch in range(epochs):
    model.train()
    for features, attention_mask, labels in train_loader: 
        
        
        # Move data to GPU
        features, attention_mask, labels = features.to(device), attention_mask.to(device), labels.to(device)
        #print(inputs)
        
        optimizer.zero_grad()

        output = model(features,attention_mask=attention_mask)
        predictions = output.__dict__['logits']

        loss = criterion(predictions, labels.long())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_train_loss += loss.item()
        total_correct += (predictions.argmax(dim=1) == labels).sum().item()
        train_accuracy = total_correct / len(train_loader.dataset)
        all_labels.extend(labels.tolist())
        all_predictions.extend(torch.argmax(predictions, dim=1).tolist())
        
        # Calculate confusion matrix
        train_cm = confusion_matrix(all_labels, all_predictions)

    with torch.no_grad():
        # Evaluate model
        test_loss, test_acc, test_f1, test_cm = evaluate(model, test_loader, criterion)
        print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {total_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Train Confusion Matrix:\n{train_cm}")
        print(f"Epoch: {epoch+1}/{epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        print(f"Confusion Matrix:\n{test_cm}")
end_time = time.time()


# In[25]:


# Calculate and print the time taken for the conditional gradient
time_taken = end_time - start_time
print(f"Time taken for Block 1: {time_taken} seconds")


# In[34]:


plt.figure(figsize=(8, 6))
sns.heatmap(test_cm, annot=True, fmt="d", cmap="Blues", cbar=False)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[ ]:




