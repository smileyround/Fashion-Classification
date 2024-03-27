# Fashion-MNIST Image Classification Report

**Author:** MengTian

## Methods

In this lab, I explored three different algorithms to classify images from the Fashion-MNIST dataset: Logistic Regression, Neural Network (NN), and Convolutional Neural Network (CNN). Here are the details of my approach for each method:

### Logistic Regression:
I used scikit-learn's `LogisticRegression` class with its default parameters for this simple model. The images were preprocessed by flattening them and normalizing their pixel values to fall within the range [0, 1].

### Neural Network (NN):
For the NN model, I designed a structure using TensorFlow and Keras that started with a `Flatten` layer to convert the 2D images into 1D vectors. This was followed by a `Dense` layer with 128 neurons and ReLU activation, a `Dropout` layer with a 0.2 rate to help prevent overfitting, and a final `Dense` layer with 10 neurons (one for each class) and softmax activation. I compiled the model using the Adam optimizer and `SparseCategoricalCrossentropy` as the loss function, and trained it for 15 epochs.

### Convolutional Neural Network (CNN):
My CNN model, also built with TensorFlow and Keras, comprised two `Conv2D` layers with ReLU activation and `MaxPooling2D` layers for feature extraction and downsampling, respectively. These were interspersed with `Dropout` layers to mitigate overfitting. The model finished with a `Flatten` layer, a `Dense` layer with 128 neurons and ReLU activation, another `Dropout` layer, and a softmax `Dense` layer for classification. The compilation and training settings were identical to those used for the NN model.

For both the NN and CNN models, I normalized the Fashion-MNIST dataset so that pixel values were in the range [0, 1] before proceeding with training.

## Results and Discussion

My findings from applying each model on the Fashion-MNIST dataset are summarized below:

- **Logistic Regression:** Marked a significant improvement, achieving a training accuracy of 86.62% and a test accuracy of 84.39%. This model served as a solid baseline and demonstrated the effectiveness of logistic regression for image classification tasks, despite its simplicity. The results are not very bad, which is beyond my expectation.

- **Neural Network (NN):** This model achieved a training accuracy of 90.37% and a test accuracy of 88.66%. The observed overfitting (higher accuracy on the training set than on the test set) suggested that while it could capture more complex patterns than logistic regression, it might still benefit from further adjustments such as increased regularization or data augmentation to better generalize.

- **Convolutional Neural Network (CNN):** Marked a significant improvement, achieving a training accuracy of 90.48% and a test accuracy of 90.77%. The CNN model outperformed both previous models with minimal overfitting, showcasing the power of convolutional networks in extracting and learning from spatial hierarchies in image data.

The Fashion-MNIST dataset represents a more challenging problem than traditional MNIST due to the more complex nature of the images (various clothing items) and the higher intra-class variation (e.g., different styles of shirts). One main challenge I encountered was finding the right balance between model complexity and the ability to generalize to unseen data. Overfitting is a significant concern, but through careful model design and training approach, improvements can be made.

## Conclusion

My exploration into classifying images from the Fashion-MNIST dataset with three different models revealed the capabilities and limitations of each method. Logistic regression provided a strong foundation, but it was the neural networks, especially CNNs, that showed superior performance due to their ability to capture complex data patterns more effectively. However, tackling overfitting remains a key challenge, emphasizing the importance of careful model design and training strategies. The success of CNNs for image-related tasks was evident, yet achieving the best results requires appropriate tuning and regularization to ensure models generalize well to new data.
