# Yelp Photo Classifier

This project focuses on developing a sophisticated computer vision system to classify business photos from Yelp into four distinct categories: Food, Drink, Interior, and Exterior. Leveraging deep learning techniques, the project explores various approaches to build and optimize a high-performance image classification model.

## Key Features and Techniques

- **Data Exploration & Preprocessing**: Comprehensive exploratory data analysis (EDA) and preprocessing to understand the dataset, including visualizing images and cleaning data.
- **Deep Neural Networks (DNN)**: Implementation of a deep neural network with 20 hidden layers, using the ReLU activation function and Adam optimization, trained over 100 epochs to establish a baseline performance.
- **Overfitting Reduction**: Application of techniques such as weight regularization, batch normalization, dropout, and early stopping to mitigate overfitting and enhance model generalization.
- **Convolutional Neural Networks (CNN)**: Construction of a CNN architecture with multiple Conv2D layers and MaxPooling layers, optimized for image classification tasks.
- **Transfer Learning**: Utilization of pre-trained models like VGG16, ResNet, Inception, MobileNet, and EfficientNet from the ImageNet dataset. The transfer learning approach involves fine-tuning these models to adapt them to the specific task of classifying Yelp business photos.

## Large Files
The dataset files used in this project are available in the [latest release](https://github.com/yourusername/your-repo/releases/tag/v1.0).

## Objectives

- Build and optimize a computer vision model capable of accurately classifying business photos.
- Compare the performance of different model architectures, including DNN, CNN, and transfer learning techniques.
- Demonstrate the application of advanced deep learning methods to improve model accuracy and robustness.

## Results

The project showcases the effectiveness of deep learning techniques in computer vision tasks, achieving significant improvements in classification accuracy through model tuning and the use of transfer learning.

### Observations from Model Training:

1. **Deep Neural Network (DNN)**:
   - The DNN model, with its excessively deep architecture of 20 hidden layers, demonstrated clear signs of overfitting. While training accuracy improved steadily, validation accuracy plateaued, indicating the model's difficulty in generalizing to unseen data. The erratic behavior of the validation loss, with marked spikes, further underscored this issue. This suggests that while the model was effective at memorizing training data, it struggled to maintain performance on the validation set.

2. **Overfitting Reduction Techniques**:
   - **Data Augmentation**: Significantly improved test accuracy by over 10%, proving effective in enhancing model generalization by increasing the variance in the training data.
   - **Early Stopping**: Implemented across all models, this technique effectively curtailed training before overfitting became pronounced, preventing excessive training on noise within the dataset.
   - **Dropout**: Combined with data augmentation, dropout slightly reduced accuracy compared to augmentation alone but added robustness to noise within the model.
   - **Weight Regularization**: Less effective compared to dropout or data augmentation, suggesting it was not the dominant factor for this particular model or dataset.
   - **Batch Normalization**: Proved to be the least effective in this context, showing minimal impact on reducing overfitting.

   In conclusion, data augmentation emerged as the most effective technique for improving model generalization.

3. **Convolutional Neural Networks (CNN)**:
   - The CNN model, tailored for image classification, significantly outperformed the previous multi-layer perceptron (MLP) model, achieving a test accuracy of 61.95%. The CNN's ability to recognize and utilize spatial hierarchies and features in image data was a key factor in its superior performance, demonstrating the effectiveness of convolutional layers specifically designed for image processing tasks.

4. **Transfer Learning**:
   - **EfficientNetB0** outperformed all other models, achieving the highest accuracy of 89.76% with moderate training time, highlighting the power of transfer learning for this classification task.
   - **VGG16** also performed well with 89.27% accuracy, though it was slightly slower to train.
   - **ResNet50** showed good accuracy (85.85%) but had a higher loss, suggesting some overfitting.
   - **MobileNet** was the quickest to train, achieving 77.56% accuracy, making it a favorable option when efficiency is a priority.
   - **InceptionV3** lagged behind with only 50.73% accuracy, indicating potential issues with model fit or data processing.

   These results demonstrate the varied capabilities of transfer learning models in handling image classification tasks, with EfficientNetB0 leading in accuracy and MobileNet in training efficiency.
