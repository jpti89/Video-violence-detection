# Video-violence-detection

## Training Violence detection model using Tensorflow Keras + MobileNetV2 model

Using the dataset from Kaggle https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset we managed to fine-tune and train a CNN model.

The architecture of the Full project is:
- Download the Dataset from Google Drive
- Extract image frames of every video standarizing every image at 128x128 px
- Separating the images in train and test sets
- Load the MobileNetV2
- Finetune the learning rates, epochs and batch size to get the best results
- Train the model
- Save weights and model
- Evaluate
