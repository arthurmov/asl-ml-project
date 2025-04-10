# Arthur Movsesyan
# CISC 3440: Machine Learning
# P3
#
# The project involves two parts:
# 1. The first part implements a Multi-layer perceptron to model the AND gate.
# 2. The second part uses Pytorch Library to work with Auto-Encoders.

# Part I: AND Gate.
# 
# **Training Data** We start by generating some training data. The data corresponds to four possible values for a two input AND gate. Below is the logic table corresponding to the AND gate.
# 
# | Input \(x_1\) | Input \(x_2\) | Output \(y\) |
# |---------------|---------------|--------------|
# | 0             | 0             | 0            |
# | 0             | 1             | 0            |
# | 1             | 0             | 0            |
# | 1             | 1             | 1            |
# 
# We created 1000 samples for each possible combination of input and output values. The code below generates the training dataset and stores it in the variable name *dataset*. Most of the 
# computations in this project will depend on the Numpy library. 

import numpy as np

dataset=np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]]*1000)
print(dataset.shape)


# **Weights and Initializations:**
# 
# The code below initializes the weights and bias values for the hidden and output layers. We initialize the hidden values to small random values and set the bias to zero.

weights_hidden =np.random.normal(0,0.1,2)
weights_out=np.random.normal(0,0.1,1)[0]
bias_hidden=0.0
bias_out=0.0
print("Weights hidden:{},weights_out:{},bias_hidden:{},bias out:{}".format(weights_hidden,weights_out,bias_hidden,bias_out))


# This forward function takes the following inputs:
# 
#    sample - a single sample from the input dataset
#    weights_hidden - weight values corresponding to the hidden units
#    weights_out - weight values corresponding to the output unit
#    bias_hidden - bias for hidden units
#    bias_out - bias for the output unit
# 
# The function returns the output value.

def forward(sample, weights_hidden, weights_out, bias_hidden, bias_out):
    """
    Computes the forward pass for a simple neural network with one hidden layer and one output layer.

    Parameters:
    ----------
    sample : numpy.ndarray
        The input vector for the forward pass. It should have a size matching the `weights_hidden` dimensions.
    weights_hidden : numpy.ndarray
        The weights of the hidden layer. This should match the dimensions of the input `sample`.
    weights_out : float
        The weight of the output layer connecting the hidden layer to the output neuron.
    bias_hidden : float
        The bias term added to the hidden layer computation.
    bias_out : float
        The bias term added to the output layer computation.

    Returns:
    -------
    output_value : float
        The final output value computed from the network after the forward pass.
    hidden_value : float
        The intermediate value computed at the hidden layer before applying the output computation.
    """
    # Compute the hidden layer value (linear combination of inputs and weights + bias)
    hidden_value = sum(sample * weights_hidden) + bias_hidden

    # Compute the output value (hidden value * output weight + bias)
    output_value = hidden_value * weights_out + bias_out

    return output_value, hidden_value


# Next, we implemented the backward pass. In the backward pass, you receive a sample, make a call to the forward pass and obtain the prediction. With the obtained prediction you estimate the error and backpropagate it. Before we begin the implementation, we define the different elements of the parameters and establish the equations for their gradients.
# 
# We define the error as the square difference between the label and the output (prediction):
# $$E(error) = \frac{1}{2} (\text{label} - \text{output})^2$$
# 
# Below are the notations denoting different parameters.
# $$\text{sample} = [x_{1}, x_{2}]$$
# $$\text{weights_hidden} = [w_{h1}, w_{h2}]$$
# $$\text{weights_out} = w_{o}$$
# $$\text{bias_hidden} = b_{h}$$
# $$\text{bias_out} = b_{o}$$
# 

def backward(sample, label, weights_hidden, weights_out, bias_hidden, bias_out, lr):
    """
    Performs the backward pass of a simple neural network to compute gradients
    and update weights and biases based on the error between predicted and true values.

    Parameters:
    ----------
    sample : numpy.ndarray
        The input vector for the backward pass. It should have the same size as `weights_hidden`.
    label : float
        The true target value corresponding to the input sample.
    weights_hidden : numpy.ndarray
        The current weights of the hidden layer.
    weights_out : float
        The current weight of the output layer connecting the hidden layer to the output neuron.
    bias_hidden : float
        The current bias term for the hidden layer.
    bias_out : float
        The current bias term for the output layer.
    lr : float
        The learning rate to scale the updates to weights and biases.

    Returns:
    -------
    weights_hidden : numpy.ndarray
        The updated weights for the hidden layer after applying gradient descent.
    weights_out : float
        The updated weight for the output layer after applying gradient descent.
    bias_hidden : float
        The updated bias for the hidden layer after applying gradient descent.
    bias_out : float
        The updated bias for the output layer after applying gradient descent.
    error : float
        The squared error between the predicted output and the true label.
        Example:
    -------
    >>> import numpy as np
    >>> sample = np.array([1.0, 2.0])
    >>> label = 1.0
    >>> weights_hidden = np.array([0.5, -0.3])
    >>> weights_out = 1.2
    >>> bias_hidden = 0.1
    >>> bias_out = -0.2
    >>> lr = 0.01
    >>> updated_weights_hidden, updated_weights_out, updated_bias_hidden, updated_bias_out, error = backward(
    ...     sample, label, weights_hidden, weights_out, bias_hidden, bias_out, lr)
    """

    # First make a forward pass on the sample to compute outputs for the hidden and the output layers
    output, hidden = forward(sample, weights_hidden, weights_out, bias_hidden, bias_out)
    
    # Compute the error
    error = 0.5 * (label - output) ** 2
    
    # Gradients for output layer
    gradient_wo = -(label - output) * hidden
    gradient_bo = -(label - output)
    
    # Gradients for hidden layer
    gradient_wh1 = -(label - output) * weights_out * sample[0]
    gradient_wh2 = -(label - output) * weights_out * sample[1]
    gradient_bh = -(label - output) * weights_out
    
    # Update weights and biases using gradient descent
    weights_hidden[0] = weights_hidden[0] - lr * gradient_wh1
    weights_hidden[1] = weights_hidden[1] - lr * gradient_wh2
    weights_out = weights_out - lr * gradient_wo
    bias_hidden = bias_hidden - lr * gradient_bh
    bias_out = bias_out - lr * gradient_bo
    
    return weights_hidden, weights_out, bias_hidden, bias_out, error


# **Training:**
# 
# Here, we train the parameters using SGD (Stochastic Gradient Descent).

def train(train_data,labels,epochs):
  """
    Trains a simple neural network using stochastic gradient descent (SGD)
    over a specified number of epochs.

    Parameters:
    ----------
    train_data : numpy.ndarray
        The training dataset, where each row is a sample and each column represents a feature.
    labels : numpy.ndarray
        The true labels corresponding to the training samples. Must have the same number of rows as `train_data`.
    epochs : int
        The number of epochs (iterations over the entire dataset) to train the model.

    Returns:
    -------
    weights : list
        A list of tuples, where each tuple contains the weights of the hidden layer
        and the weight of the output layer at each epoch.
    biases : list
        A list of tuples, where each tuple contains the bias of the hidden layer
        and the bias of the output layer at each epoch.
    epoch_errors : list
        A list of average errors (mean squared error) per epoch.

    Notes:
    -----
    - This function updates the global variables `weights_hidden`, `weights_out`,
      `bias_hidden`, and `bias_out` during training.
    - The dataset is shuffled at the start of each epoch to ensure randomness
      in training and to prevent overfitting to a specific order of data.
  """

  global weights_hidden
  global weights_out
  global bias_hidden
  global bias_out
  weights=[]
  biases=[]
  epoch_errors=[]
  for epoch in range(epochs):
    epoch_error=0
    weights.append([weights_hidden,weights_out])
    biases.append([bias_hidden,bias_out])
    shuffle=np.random.permutation(train_data.shape[0])
    train_data=train_data[shuffle]
    labels=labels[shuffle].reshape(-1)
    for x,y in zip(train_data.tolist(),labels.tolist()):
      weights_hidden,weights_out,bias_hidden,bias_out,error=backward(x,y,weights_hidden,weights_out,bias_hidden,bias_out,lr=0.0001)
      epoch_error+=error
    epoch_errors.append(epoch_error/train_data.shape[0])
  return weights, biases, epoch_errors


train_data=dataset[:,:2]
labels=dataset[:,2:]
weights, biases, epoch_errors=train(train_data,labels,100)


# **Evaluation:**
# 
# After training for the desired number of epochs, we plot the training error over the epochs.

import matplotlib.pyplot as plt
plt.plot(epoch_errors)

weights_=weights[-1]
bias_=biases[-1]
weights_hidden=weights_[0]
weights_out=weights_[1]
bias_hidden=bias_[0]
bias_out=bias_[1]
o,h=forward(np.array([0,0]),weights_hidden,weights_out,bias_hidden,bias_out)

print("Prediction for {}, is {}".format([0,0],o))
o,h=forward(np.array([0,1]),weights_hidden,weights_out,bias_hidden,bias_out)
print("Prediction for {}, is {}".format([0,1],o))
o,h=forward(np.array([1,0]),weights_hidden,weights_out,bias_hidden,bias_out)
print("Prediction for {}, is {}".format([1,0],o))
o,h=forward(np.array([1,1]),weights_hidden,weights_out,bias_hidden,bias_out)
print("Prediction for {}, is {}".format([1,1],o))


# ## Part II Auto-encoders with Pytorch.
# 
# An autoencoder is a type of neural network used to learn efficient representations of data, typically for dimensionality reduction or feature learning. It consists of two main parts:
# 
# - **Encoder:** Maps the input data to a smaller, compressed representation called the "latent space" or "bottleneck".
# - **Decoder:** Reconstructs the original input data from the compressed representation.
# 
# The network is trained to minimize the difference between the input and the reconstructed output, ensuring that the encoder captures the most relevant features of the data. To implement, autoencoders we will use [Pytorch](https://pytorch.org/), which is popular deep learning framework.

# **Network Structure**
# 
# The network structure for the autoencoder is shown below, in general we care about compressing or reducing the dimensionality of the input which can be used to reconstruct the output. Both the encode and decode layers are modelled as separate multilayer perceptrons optimized with a single error function, the reconstruction loss.
# 
#        Input (x)
#           |
#     [Encoder Layers]
#           |
#     Compressed Representation (latent)
#           |
#     [Decoder Layers]
#           |
#     Reconstructed Input (x_)
# 
# 
# The encoder takes the input and outputs the compressed latent vector, the decoder takes as input compressed latent vector and outputs the reconstructed  input from the latent vector. Below you will find the code to implement the autoencoder network in Pytorch. Similar to NumPy arrays, Pytorch works with objects of type *Tensor*. All the parameters, and inputs should be provided as tensors.
# 
# **Note:**
# - [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) creates a hidden layer with the number of neurons specified in the hidden dimension.
# -  [nn.ReLU()](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) is an activation function similar to sigmoid, etc.
# - The model will be trained on a GPU. To use GPUs in Google Colab, go to "Runtime" > "Change runtime type" and select a GPU like T4, A100, etc.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    """
    A simple autoencoder model implemented in PyTorch.
    This model compresses input data into a smaller representation (latent space) and
    then reconstructs the input data from the compressed representation.

    Parameters:
    ----------
    input_dim : int
        The size (number of features) of the input data.
    hidden_dim : int
        The size of the hidden layers used in both encoder and decoder.
    binary_dim : int
        The size of the compressed representation (latent space).

    Layers:
    -------
    - Encoder:
        * A series of fully connected layers that reduce the input dimension to the binary_dim.
        * Includes ReLU activation functions for non-linear transformations.
    - Decoder:
        * A series of fully connected layers that reconstruct the input data from the binary_dim.
        * Includes ReLU activation functions for non-linear transformations.

    Methods:
    -------
    forward(x):
        Performs a forward pass through the network. Encodes the input data to a latent
        representation and then decodes it back to the reconstructed input.
        Returns both the reconstructed input and the latent representation.
    """
    def __init__(self, input_dim, hidden_dim, binary_dim):
        super(Autoencoder, self).__init__()

        # Define the encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # First hidden layer
            nn.ReLU(),                         # Non-linearity
            nn.Linear(hidden_dim, hidden_dim), # Second hidden layer
            nn.ReLU(),                         # Non-linearity
            nn.Linear(hidden_dim, binary_dim), # Output layer of the encoder (compressed representation)
        )

        # Define the decoder network
        self.decoder = nn.Sequential(
            nn.Linear(binary_dim, hidden_dim), # First hidden layer
            nn.ReLU(),                         # Non-linearity
            nn.Linear(hidden_dim, hidden_dim), # Second hidden layer
            nn.ReLU(),                         # Non-linearity
            nn.Linear(hidden_dim, input_dim),  # Output layer of the decoder (reconstructed input)
        )

    def forward(self, x):
        """
        Performs a forward pass through the autoencoder.
        """
        latent = self.encoder(x)  # Compressed representation
        x_ = self.decoder(latent)  # Reconstructed input
        return x_, latent


# **Dataset**
# 
# 
# We used a hand pose dataset for training. The goal was to reduce the dimensions of the data. The hand pose is comprised of 21 joints in an Image. The images where obtained using [Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/guide). The figure below shows the 21 hand pose points obtained from Mediapipe.
# 
# ![link text](https://ai.google.dev/static/mediapipe/images/solutions/hand-landmarks.png)

# Let us first load our dataset and visualize some poses. The dataset we will use for this is [How2Sign Dataset](https://how2sign.github.io/), an ASL dataset for Automatic ASL recognition. To train our autoencoder I have extracted 296743 right hand poses which you will find in the **right_hand.npy** Numpy file provided along with your project.
# 
# We start by doing some data transformations and visualize the data.

import numpy as np

right_hand_data=np.load("right_hand.npy",allow_pickle=True)
print(right_hand_data.shape) # There are 21 joints each with 4 dimensions X,Y,Z and handedness (right or left)
#We will ignore the last two dimensions
right_hand_data=right_hand_data[:,:,:2]
print("Min:{} and Max:{} values before transformation".format(np.min(right_hand_data),np.max(right_hand_data)))
#We scale the data between 0 and 1
right_hand_data_transformed=(right_hand_data-np.min(right_hand_data))/(np.max(right_hand_data)-np.min(right_hand_data))
print("Min:{} and Max:{} values after transformation".format(np.min(right_hand_data_transformed),np.max(right_hand_data_transformed)))


# We used the function below to draw the hand_pose on an image and visualize it. The function takes a matplotlib axis and a hand pose to plot the hand pose on the given axis.

def draw_hand_from_numpy(ax, hand_pose, color='black', radius=1, thickness=2):
    """
    Draw the hand landmarks on a matplotlib axis.

    Args:
    - ax: The matplotlib axis to draw on.
    - hand_pose: A numpy array of shape (21, 3) containing the x, y, z coordinates of hand landmarks.
    - color: The color for the landmarks and connections (default is white).
    - radius: Radius of circles representing the landmarks.
    - thickness: Thickness of the lines connecting the landmarks.
    """
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),   # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),   # Index finger
        (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (13, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]

    # Plot connections
    for start_idx, end_idx in connections:
        start_point = hand_pose[start_idx, :2]  # (x, y)
        end_point = hand_pose[end_idx, :2]  # (x, y)
        ax.plot([start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                color=color, linewidth=thickness)

    # Plot landmarks
    for i in range(hand_pose.shape[0]):
        x, y = hand_pose[i, 0], hand_pose[i, 1]
        ax.scatter(x, y, color=color, s=radius * 10)


import matplotlib.pyplot as plt

rows, cols = 3, 3
fig, axes = plt.subplots(rows, cols, figsize=(5, 5))
axes = axes.flatten()
for i in range(9):
  draw_hand_from_numpy(axes[i],right_hand_data_transformed[i])
plt.tight_layout()
plt.show()


# **Training:**
# 
# We are now ready to Train. **Note the training assumes that you are operating on a GPU node. If not please change the runtime before you begin.** Not doing so might lead to error.
# 
# The code below creates train/test split and also initializes the hyperparameters we will use to define the network and train.

from sklearn.model_selection import train_test_split
right_hand_data_transformed=right_hand_data_transformed.reshape(-1,42).astype(dtype=np.float32) # Reshape each sample into a 1D vector
# # Hyperparameters
input_dim = 42      # Input dimension (number of features)
hidden_dim = 32      # Hidden layer size
latent_dim = 10     # Bottleneck latent representation size
learning_rate = 0.001
num_epochs = 50
train,test=train_test_split(right_hand_data_transformed,test_size=0.2,random_state=42)
print(train.shape,test.shape)


# The next two cells of code, initialize the model and use methods to load the data. We use MSE (Means Square Error) as the loss function and the Adam optimizer to optimize our network parameters, that wills take care of the backward propagation.
# 
# If you looked at the Autoencoder model definition you will only find the forward definition. Given that Pytorch figures out the required backpropagation computations. Which simplifies designining models and reduces probable errors in our code.

train=TensorDataset(torch.tensor(train,dtype=torch.float))
dataloader = DataLoader(train, batch_size=100, shuffle=True)
model = Autoencoder(input_dim, hidden_dim, latent_dim)
model=model.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
      epoch_loss=0
      batch_count=0
      for batch_train in dataloader:
        model.train()
        optimizer.zero_grad()

        # Forward pass
        reconstructed, latent = model(batch_train[0].cuda())
        loss = criterion(reconstructed.cuda(), batch_train[0].cuda())

        # Backward pass
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
        batch_count+=1
      if (epoch + 1) % 5 == 0:
          print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss/batch_count:.9f}")


# **Evaluation**
# 
# We evaluate the trained model on the test dataset. We compare the trained models reconstruction loss and visualize the original and reconstructed poses.

latent_values=[]
input_values=[]
reconstructed_values=[]
model.eval()
with torch.no_grad():
    for i in range(test.shape[0]):
      test_sample=torch.tensor(test[i],dtype=torch.float32)
      reconstructed, binary_latent= model(test_sample.cuda())
      latent_values.append(binary_latent.cpu().tolist())
      input_values.append(test_sample.cpu().tolist())
      reconstructed_values.append(reconstructed.cpu().tolist())


print("The error for the test data is {}".format(criterion(torch.tensor(input_values),torch.tensor(reconstructed_values))))


# Below we visually examine the original and reconstructed poses. The plots in green are the original ones and the ones in Red are reconstructed. As you can see the reconstructions are accurate for the examined samples.


rows,cols=4,2
fig, axes = plt.subplots(rows, cols, figsize=(5, 5))
axes = axes.flatten()
for i in range(0,8,2):
    ground_truth=np.array(input_values[i+10]).reshape(21,-1)
    predicted=np.array(reconstructed_values[i+10]).reshape(21,-1)
    draw_hand_from_numpy(axes[i],ground_truth,color='green')
    draw_hand_from_numpy(axes[i+1],ground_truth,color='red')
plt.tight_layout()
plt.show()


#  Below, the trained version of the model gets saved.
import pickle

# Save the trained model to a .pkl file
model_file = "autoencoder_model.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {model_file}")

