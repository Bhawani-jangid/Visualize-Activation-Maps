import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json, Model

# Load the model structure
with open('model_a1.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)

# Load the model weights
model.load_weights('model_weights.weights.h5')

# Load an image for visualization
image_path = r"D:\project\Data Analysis\Visualize Activation Map\download Ing\ospan-ali-i-y1Q7i3faI-unsplash.jpg"  # Replace with your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (48, 48))
image = image / 255.0
image = np.expand_dims(image, axis=0)
image = np.expand_dims(image, axis=-1)

# Choose a layer to visualize
layer_name = 'max_pooling2d_28'  # Replace with the name of the layer you want to visualize

# ValueError: No such layer: conv2d_10. Existing layers are: ['input_layer_7', 'conv2d_28', 'batch_normalization_42', 'activation_42', 'max_pooling2d_28', 'dropout_42', 'conv2d_29', 'batch_normalization_43', 'activation_43', 'max_pooling2d_29', 'dropout_43', 'conv2d_30', 'batch_normalization_44', 'activation_44', 'max_pooling2d_30', 'dropout_44', 'conv2d_31', 'batch_normalization_45', 'activation_45', 'max_pooling2d_31', 'dropout_45', 'flatten_7', 'dense_21', 'batch_normalization_46', 'activation_46', 'dropout_46', 'dense_22', 'batch_normalization_47', 'activation_47', 'dropout_47', 'dense_23'].

# Create a new model that outputs the activations of the chosen layer
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(image)

# Number of feature maps in the chosen layer
num_feature_maps = intermediate_output.shape[-1]

plt.figure(figsize=(15, 15))
for i in range(num_feature_maps):
    plt.subplot(8, 8, i+1)  # Adjust the subplot grid size according to the number of feature maps
    plt.imshow(intermediate_output[0, :, :, i], cmap='viridis')
    plt.axis('off')
plt.show()
