This project demonstrates image classification using Convolutional Neural Networks (CNN) to classify images of celebrities including Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli.
## Dataset
The dataset consists of cropped images of the mentioned celebrities. The dataset is organized into subdirectories for each celebrity, and the images are preprocessed and resized for model training.

## Requirements
- Python 3
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- OpenCV
- PIL (Pillow)
- Scikit-learn

## The CNN model architecture is as follows:
Input Layer
Convolutional Layer 1 with ReLU activation and MaxPooling
Convolutional Layer 2 with ReLU activation and MaxPooling
Convolutional Layer 3 with ReLU activation and MaxPooling
Dropout Layer
Flatten Layer
Dense Layer 1 with ReLU activation
Dense Layer 2 with ReLU activation
Output Layer with Softmax activation


## Data Loading and Preprocessing:
Celebrity images of Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli are loaded and resized to (128, 128) pixels. The dataset is split into training and testing sets.

## Model Architecture:
The model architecture comprises three convolutional layers with max-pooling, complemented by a dropout layer and a flattening layer.
The dropout layer incorporates a dropout rate of 0.2, meaning 20% of the input units are randomly omitted during training. Subsequently, dense layers are added to the model—two with ReLU activation function and another with softmax activation for multiclass classification.
The evaluation metrics include sparse categorical crossentropy loss, accuracy, and the Adam optimizer.

## Training:
The model is trained for 20 epochs on the training set using a batch size of 32. Training progress is monitored, and accuracy and loss metrics are recorded.

## Evaluation:
Graphs are generated to visualize accuracy and loss during the evaluation process.

## Model Prediction:
The trained model predicts labels for the test set. Predictions and actual labels are stored in a CSV file.

## Critical Findings
Accuracy is a key metric indicating the model's ability to generalize to new data.
Experimenting with learning rates, dropout rates, or adjusting the model architecture may yield improvements in accuracy.
