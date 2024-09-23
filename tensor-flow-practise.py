# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist

# For visualization
import matplotlib.pyplot as plt
import numpy as np

# 1. Load and preprocess the dataset
def load_data():
    # Load Fashion MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    return (train_images, train_labels), (test_images, test_labels)

# 2. Build the Neural Network Model
def build_model():
    model = models.Sequential()
    
    # Input layer: Flatten 28x28 images into a 1D vector
    model.add(layers.Flatten(input_shape=(28, 28)))
    
    # Hidden layer: Dense (fully connected) with 128 neurons and ReLU activation
    model.add(layers.Dense(128, activation='relu'))
    
    # Output layer: Dense with 10 neurons (one for each class) and softmax activation
    model.add(layers.Dense(10, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# 3. Train the model
def train_model(model, train_images, train_labels):
    # Fit the model to the training data
    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_split=0.2, batch_size=64)
    
    return history

# 4. Evaluate the model
def evaluate_model(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'Test accuracy: {test_acc}')
    return test_loss, test_acc

# 5. Visualize the training results
def plot_accuracy_loss(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)

    # Plot training and validation accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# 6. Make predictions
def make_predictions(model, test_images, class_names):
    predictions = model.predict(test_images)

    # Plot the first 5 test images with their predicted and true labels
    plt.figure(figsize=(10, 10))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        predicted_label = np.argmax(predictions[i])
        true_label = test_labels[i]
        color = 'green' if predicted_label == true_label else 'red'
        plt.xlabel(f'{class_names[predicted_label]} ({class_names[true_label]})', color=color)
    plt.show()

# Main function to run the steps
def main():
    # Class names corresponding to Fashion MNIST labels
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Load the dataset
    (train_images, train_labels), (test_images, test_labels) = load_data()

    # Build the model
    model = build_model()

    # Train the model
    history = train_model(model, train_images, train_labels)

    # Evaluate the model
    evaluate_model(model, test_images, test_labels)

    # Plot accuracy and loss during training
    plot_accuracy_loss(history)

    # Make predictions on the test data
    make_predictions(model, test_images, class_names)

if __name__ == "__main__":
    main()
