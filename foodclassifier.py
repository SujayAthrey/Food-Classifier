import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

#load the 'food101' dataset from the tensor
dataset_name = "food101"
(ds_train, ds_test), ds_info = tfds.load(
    dataset_name, split=["train", "validation"], with_info=True, as_supervised=True
)

# Preprocess and prepare the data
def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))  # Resize to a common size
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize pixel values
    return image, label

ds_train = ds_train.map(preprocess_image).shuffle(1000).batch(32)
ds_test = ds_test.map(preprocess_image).batch(32)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(101, activation='softmax')  # 101 classes in Food101 dataset
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(ds_train, epochs=5)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(ds_test)
print("Test accuracy:", test_accuracy)

# Using numpy for additional operations
# For example, if you want to create a numpy array from the model's predictions
sample_images, sample_labels = next(iter(ds_test))
predictions = model.predict(sample_images)

# Convert numerical predictions to category names
category_names = [ds_info.features['label'].int2str(label) for label in np.argmax(predictions, axis=1)]

# Display images with predictions
for i in range(len(sample_images)):
    plt.figure()
    plt.imshow(sample_images[i])
    plt.title(f"Prediction: {category_names[i]}")
    plt.axis('off')

plt.show()