# Food-Classifier

This neural network can accurately classify images of foods based on their category. 

A category includes Pizzas, Burgers, Waffles, Fries, Tofu, etc. Overall, the dataset is trained to be able to distinguish 101 different categories of foods in total. 

The dataset used here is the 'food101' dataset acquired from TensorFlows pre-established datasets for downloading. This dataset contains 101 different food categories (as mentioned above), where there are 101,000 images of food in total. For the purposes of this neural network, 75,750 of those images will be used to train our model and the rest (25,250 images) will be used to then test the model.

The images in the dataset are first pre-processed into subsections for training and testing, then we define, compile, and fit our neural network model to our required layers, sizes, and necessary spatial dimensions. This can be done using Keras, the high-level API that is already integrated into TensorFlow. This process includes many steps, including setting up an activation function, loss function, layer size, flattening the layers, and densening the layers, just to name a few. 

To gain the optimal accuracy, the weights/cost and activation function of each hidden layer in our network is revised through back-propagation once the loss function based on the predicted v. expected labels for each food image was calculated. This is done during the training of our model using the trained dataset containing 75,750 images.

In order to make the output more human-readable and pretty, I imported NumPy and Pyplot python libraries in order to convert numerical predictions into readable categories and to visually display images within the dataset with their associated predictions.



