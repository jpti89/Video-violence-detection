# Video violence detection

## Training Violence detection model using Tensorflow Keras + MobileNetV2 model

Using the dataset from Kaggle https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset we managed to fine-tune and train a CNN model.

The architecture of the Train_violence_model.ipynb project is:
- Download the Dataset from Google Drive
- Extract image frames of every video standarizing every image at 128x128 px
- Separating the images in train and test sets
- Load the MobileNetV2
- Finetune the learning rates, epochs and batch size to get the best results
- Train the model
- Save weights and model
- Evaluate

The labels of each frame are mantained for the whole video. That means, if there's violence only in some frames of the video, the network learns from the whole video anyway. This could generate a problem with some incorrect predictions using videos from outside the traning dataset.


Then the architecture of the Flask Violence detection.ipynb project is:
- (One time) Generate thumbnails of the videos
- Load the already trained model
- Generate a Web APP that shows a thumbnail of the videos, and responds with the inference of the model of how likely is that the image has violence (Value between 0 and 1, being 1 more likely of violence)
