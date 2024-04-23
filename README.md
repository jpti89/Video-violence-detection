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

### Validation results:
![Reults1](https://github.com/jpti89/Video-violence-detection/blob/main/Train_results/results.png)
![Reults2](https://github.com/jpti89/Video-violence-detection/blob/main/Train_results/results2.png)

### Limitations
The labels of each frame are mantained for the whole video. That means, if there's violence only in some frames of the video, the network learns from the whole video anyway. This could generate a problem with some incorrect predictions using videos from outside the traning dataset.

### Use of MobileNetV2
MobileNetV2 was choosen because it's lightweight, fast and easy to train model, accurate for our CNN clasification use case. It's so light that it can be implemented on edge computing devices for continuing training and live inferencing.

### Architectural Features of MobileNetV2
1. Inverted Residuals:
In traditional residual networks (ResNets), the main idea is to learn residual functions with reference to the layer inputs, which are then added to the layer outputs. This helps in easier training of very deep networks.
In MobileNetV2, the concept of residuals is inverted. Instead of the input being added to the output, the output is added to the input. This is particularly useful in mobile architectures where reducing the number of computations is crucial.

2. Linear Bottlenecks:
Traditional bottleneck architectures, as seen in models like ResNet, typically involve reducing the dimensionality of the feature maps (using 1x1 convolutions) before applying more computationally expensive operations (like 3x3 convolutions).
In MobileNetV2, linear bottlenecks are introduced to maintain low-dimensional feature representations throughout the network. This is achieved by avoiding non-linearities (like ReLU activation) after the 1x1 convolutions in the bottleneck layers.

![The-proposed-MobileNetV2-network-architecture](https://github.com/jpti89/Video-violence-detection/assets/18633422/825b9b1f-76e1-494c-8948-d9b254031444)
* Instead of Softmax, we use a Sigmoid function

MobileNetV2 is a convolutional neural network architecture that seeks to perform well on mobile devices. It is based on an inverted residual structure where the residual connections are between the bottleneck layers. The intermediate expansion layer uses lightweight depthwise convolutions to filter features as a source of non-linearity. As a whole, the architecture of MobileNetV2 contains the initial fully convolution layer with 32 filters, followed by 19 residual bottleneck layers.


Then the architecture of the Flask Violence detection.ipynb project is:
- (One time) Generate thumbnails of the videos
- Load the already trained model
- Generate a Web APP that:
1. Shows a thumbnail of a random video from the Train/Val Dataset, and responds with the inference of the model of how likely is that the image has violence, (Value between 0 and 1, being 1 more likely of violence)
2. Evaluate the model on YouTube videos
3. Evaluate the model using the WebCamera of the users PC
4. Evaluate the model using an RTSP feed

