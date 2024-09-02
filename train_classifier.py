import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data dictionary
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Get the data and labels
data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Find the maximum length of sequences in the data
max_length = max(len(seq) for seq in data)

# Function to pad sequences to the maximum length
def pad_sequence(seq, target_length):
    return seq + [0] * (target_length - len(seq))

# Pad all sequences to the max length
padded_data = [pad_sequence(seq, max_length) for seq in data]

# Convert the padded data to a NumPy array
data = np.asarray(padded_data)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create and train the Random Forest Classifier model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict the labels for the test set
y_predict = model.predict(x_test)

# Calculate the accuracy score
score = accuracy_score(y_predict, y_test)

# Print the classification accuracy
print(f'{score * 100}% of samples were classified correctly!')

# Save the trained model to a file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
