import numpy as np
from trainPerceiver_v2 import train_and_evaluate

#Code implementation of the SeqPerciever architecture was made by Devansh Mishra and Christopher Cheong
#mishr174.umn.edu, cheon028.umn.edu


#Uncomment below and insert seeds to test training at
#Make sure to download the data at https://drive.google.com/drive/folders/1HAfYEukwieGzYOhS9kH9mxEEwkOi-in5?usp=drive_link

#seed_values = [] insert seed values here

results = [train_and_evaluate(seed) for seed in seed_values]

train_accuracies = [result[0] for result in results]
test_accuracies = [result[1] for result in results]

average_accuracy = np.mean(test_accuracies)
std_dev_accuracy = np.std(test_accuracies)
conf_interval_95 = 1.96 * std_dev_accuracy / np.sqrt(len(test_accuracies))
max_accuracy = np.max(test_accuracies)
min_accuracy = np.min(test_accuracies)

print(f"Average Test Accuracy: {average_accuracy:.2f}%")
print(f"Standard Deviation of Accuracy: {std_dev_accuracy:.2f}")
print(f"95% Confidence Interval: ±{conf_interval_95:.2f}%")
print(f"Maximum Accuracy: {max_accuracy:.2f}%")
print(f"Minimum Accuracy: {min_accuracy:.2f}%")