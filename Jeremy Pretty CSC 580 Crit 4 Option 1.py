import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepchem as dc
from sklearn.metrics import accuracy_score

np.random.seed(456)
tf.random.set_seed(456)

# Step 1: Load the Tox21 Dataset
_, (train, valid, test), _ = dc.molnet.load_tox21()
train_X, train_y, train_w = train.X, train.y, train.w
valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
test_X, test_y, test_w = test.X, test.y, test.w

# Step 2: Remove extra datasets
train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]
train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]

# Step 3: Define the model
d = train_X.shape[1]
n_hidden = 50

model = tf.keras.Sequential([
    tf.keras.layers.Dense(n_hidden, activation='relu', input_shape=(d,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Step 4: Compile the model
learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model
batch_size = 100
n_epochs = 10
history = model.fit(train_X, train_y,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    validation_data=(valid_X, valid_y),
                    verbose=1)

# Step 6: Make predictions
valid_y_pred = model.predict(valid_X)
valid_y_pred = np.round(valid_y_pred)

# Step 7: Calculate accuracy
accuracy = accuracy_score(valid_y, valid_y_pred)
print("Validation Accuracy: {:.2f}%".format(accuracy * 100))

# Step 8: Plot the loss curve
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
