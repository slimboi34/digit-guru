import tensorflow as tf
from tensorflow import keras
from keras import layers, models, Input
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape data to include the channel dimension
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Build the model
model = models.Sequential([
    Input(shape=(28, 28, 1)),  # Explicit Input layer
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # Output layer for 10 classes (0-9 digits)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# Save the model in HDF5 format (with full architecture and weights)
model.save('my_model_complete.h5')
print("Model saved successfully as 'my_model_complete.h5'.")

# Verify the saved model by loading it and checking the summary
try:
    loaded_model = tf.keras.models.load_model('my_model_complete.h5')
    print("Model reloaded successfully.")
    loaded_model.summary()

    # Test prediction with dummy data
    test_input = np.random.rand(1, 28, 28, 1)  # Replace with real preprocessed user input if needed
    prediction = loaded_model.predict(test_input)
    print("Test prediction:", prediction)

except OSError as e:
    print(f"Error loading model: {e}")

# Save in TensorFlow SavedModel format for further compatibility
tf.saved_model.save(model, 'saved_model_complete')
print("Model saved successfully as 'saved_model_complete' (TensorFlow SavedModel format).")