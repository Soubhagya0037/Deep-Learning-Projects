# Neural Style Transfer (NST) Project

This project demonstrates the implementation of Neural Style Transfer (NST) using two different optimization techniques: Adam optimizer and L-BFGS optimizer. NST is a technique of blending the content of one image with the style of another image using deep learning models.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
5. [Results](#results)
6. [Code Explanation](#code-explanation)
7. [References](#references)

## Introduction

Neural Style Transfer uses a pre-trained VGG19 model to extract features from the content and style images. The goal is to generate a new image that maintains the content of the content image while adopting the style of the style image.

### Techniques Used

1. **Adam Optimizer**: A popular optimization algorithm in deep learning.
2. **L-BFGS Optimizer**: A quasi-Newton method that is efficient for problems with large-scale data.

## Requirements

Ensure you have the following packages installed:

- numpy
- matplotlib
- torch
- torchvision
- tqdm
- torch_snippets
- PIL (Python Imaging Library)
- tensorflow (for the Adam optimizer implementation)

You can install the required packages using the following commands:

```sh
pip install numpy matplotlib torch torchvision tqdm torch_snippets pillow tensorflow
```

## Project Structure

```
.
├── adam_optimizer_nst.py        # NST implementation using Adam optimizer
├── lbfgs_optimizer_nst.py       # NST implementation using L-BFGS optimizer
├── Content_image.jpg            # Example content image
├── Painting_Style.jpg           # Example style image
└── README.md                    # Project documentation
```

## Usage

### Adam Optimizer Implementation

The following code demonstrates NST using the Adam optimizer:

```python
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
from IPython.display import clear_output

# Set up the device
device = "/device:GPU:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"
print(f"Using device: {device}")

# Define preprocess and postprocess functions
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)

# Load the images
content_path = "Content_image.jpg"
style_path = "Painting_Style.jpg"

content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')

# Define the VGG19 model
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# Define layers and loss functions
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def vgg_layers(layer_names):
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

# Define the style-content model
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [self.gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}

    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations

# Instantiate the model
extractor = StyleContentModel(style_layers, content_layers)

# Extract style and content targets
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

image = tf.Variable(content_image)

# Optimizer
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# Loss functions
style_weight = 1e-2
content_weight = 1e4

def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2) for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
    return x_var, y_var

def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)

# Training step
@tf.function
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_loss(image)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    return loss

# Training loop with tqdm
epochs = 10
steps_per_epoch = 5
step = 0
progress_bar = tqdm(total=epochs * steps_per_epoch, desc="Training")

for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        progress_bar.update(1)
    clear_output(wait=True)
    plt.imshow(tf.squeeze(image.read_value(), axis=0) / 255.0)
    plt.title("Train step: {}".format(step))
    plt.show()

progress_bar.close()
```

### L-BFGS Optimizer Implementation

The following code demonstrates NST using the L-BFGS optimizer:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch_snippets import Report, show
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # For progress tracking
from torchvision.models import vgg19, VGG19_Weights

# Set the device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

# Define preprocess and postprocess functions
preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    T.Lambda(lambda x: x.mul_(255))
])

postprocess = T.Compose([
    T.Lambda(lambda x: x.mul(1./255)),
    T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
    T.Lambda(lambda x: x.clamp(0, 1))  # Ensure values are clamped between 0 and 1
])

# Load images
imgs = [Image.open(path).resize((512, 512)).convert('RGB') for path in ['/content/Content_image.jpg', '/content/Painting_Style.jpg']]
style_image, content_image = [preprocess(img).to(device)[None] for img in imgs]

# Modify content image with `require_grad=True`
opt_img

 = content_image.data.clone()
opt_img.requires_grad = True

# Display the images
fig, axes = plt.subplots(nrows=1, ncols=len(imgs), figsize=(len(imgs) * 5, 5))
for idx in range(len(imgs)):
    axes[idx].imshow(imgs[idx])
    axes[idx].axis('off')  # Hide the axis
plt.show()

# Define the VGG19 model
class VGG19Modified(nn.Module):  # Changed class name to VGG19Modified for clarity
    def __init__(self):
        super().__init__()
        features = list(vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features)
        self.features = nn.ModuleList(features).eval()
        
    def forward(self, x, layers=[]):
        _results, results = [], []  # Changed to store outputs of requested layers
        for ix, model in enumerate(self.features):
            x = model(x)
            if ix in layers: _results.append(x)
        order = np.argsort(layers)  # Ensure the layers are sorted
        for o in order: results.append(_results[o])
        return results if layers else x  # Ensure a list is returned if layers are specified
    
# Define the Gram Matrix
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        feat = input.view(b, c, h * w)
        G = torch.bmm(feat, feat.transpose(1, 2))
        return G.div_(h * w)
    
# Define the Gram MSE Loss
class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = F.mse_loss(GramMatrix()(input), target)
        return out

# Instantiate the modified VGG19 model
vgg = VGG19Modified().to(device)

# Define layers and corresponding names for style and content
style_layers_indices = [0, 5, 10, 19, 28]
content_layers_indices = [21]

# Define the loss functions
loss_fns = [GramMSELoss()] * len(style_layers_indices) + [nn.MSELoss()] * len(content_layers_indices)
loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]

# Weights for the losses
style_weights = [1e3 / n**2 for n in [64, 128, 256, 512, 512]]
content_weights = [1e0]
weights = style_weights + content_weights

# Extract style and content targets
style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers_indices)]
content_targets = [A.detach() for A in vgg(content_image, content_layers_indices)]
targets = style_targets + content_targets

# Define the optimizer
max_iters = 500
optimizer = optim.LBFGS([opt_img])

# Training loop with tqdm
n_iter = [0]
progress_bar = tqdm(total=max_iters, desc="Optimizing")

while n_iter[0] <= max_iters:
    def closure():
        optimizer.zero_grad()
        out = vgg(opt_img, style_layers_indices + content_layers_indices)
        layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
        loss = sum(layer_losses)
        loss.backward()
        n_iter[0] += 1
        if n_iter[0] % 50 == 0:
            progress_bar.set_postfix({"Loss": loss.item()})
        progress_bar.update(1)
        return loss
    
    optimizer.step(closure)

progress_bar.close()

# Display the final image
out_img = postprocess(opt_img[0].cpu()).permute(1, 2, 0)
plt.imshow(out_img)
plt.title("Optimized Image")
plt.axis('off')
plt.show()
```

## Results

The generated images will be displayed after the completion of the training loops for both optimizers. You should compare the quality of the images produced by the Adam optimizer and the L-BFGS optimizer.

## Code Explanation

- **Adam Optimizer Code**: This code utilizes TensorFlow and the Adam optimizer for NST. It defines the VGG19 model, extracts content and style features, and optimizes the image iteratively while displaying the progress using `tqdm`.

- **L-BFGS Optimizer Code**: This code uses PyTorch and the L-BFGS optimizer for NST. It sets up a similar structure as the Adam optimizer code but employs the L-BFGS optimizer, which can often lead to faster convergence for NST problems.

## References

- [Neural Style Transfer Using TensorFlow](https://www.tensorflow.org/tutorials/generative/style_transfer)
- [Neural Style Transfer Using PyTorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
- [Torch Snippets](https://github.com/harvitronix/torch-snippets)


