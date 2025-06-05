import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to pad sequences manually (using numpy)
def pad_sequences_custom(data, maxlen):
    padded_data = []
    for seq in data:
        # Pad with zeros if the sequence is shorter than maxlen
        if len(seq) < maxlen:
            padded_seq = seq + [0] * (maxlen - len(seq))
        else:
            padded_seq = seq[:maxlen]  # Truncate if the sequence is longer than maxlen
        padded_data.append(padded_seq)
    return np.array(padded_data)

# Load processed data
try:
    data_dict = pickle.load(open('data.pickle', 'rb'))
    data = data_dict['data']
    labels = data_dict['labels']
    
    # Inspect the first few data points to see the structure
    print(f"First few data points:")
    for i in range(min(5, len(data))):  # Inspecting up to 5 data points
        print(f"Data point {i}: {len(data[i])} values")
    
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Check if all sequences in data have the same length
data_lengths = [len(d) for d in data]
print(f"Lengths of data points: {data_lengths[:10]}")  # Display the first 10 lengths

# Find the length of the longest sequence
max_len = max(data_lengths)
print(f"Maximum sequence length: {max_len}")

# Pad the sequences to ensure all sequences have the same length
padded_data = pad_sequences_custom(data, max_len)

# Convert data and labels to numpy arrays
data = np.asarray(padded_data)
labels = np.asarray(labels)

# Check the shapes of data and labels
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# Ensure labels are correctly formatted (one label per sample)
if len(data) != len(labels):
    print("Mismatch between number of samples and number of labels!")
    raise ValueError("Data and labels do not align!")

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, shuffle=True)

print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")

# Train Random Forest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate model accuracy
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)

print(f'{accuracy * 100:.2f}% of samples were classified correctly!')

# Save trained model
try:
    with open('model.p', 'wb') as f:
        pickle.dump({'model': model}, f)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")
    raise
