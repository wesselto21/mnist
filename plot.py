import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dnn import DNN

train = pd.read_csv("/Users/wesselto/Desktop/GitHub/MNIST/train.csv")

# Variable to select which image to display (0 to 59999 for training set)
image_index = 1  # Change this number to display different images

image = train.iloc[image_index, 1:].values.reshape(28, 28)
input_image = train.iloc[image_index, 1:].values / 255.0 * 0.99 + 0.01
actual_label = train.iloc[image_index, 0]

# Load the trained model
dnn = DNN(sizes=[784, 128, 64, 10], epochs=10, lr=0.001)
dnn.load_model("/Users/wesselto/Desktop/GitHub/MNIST/trained_dnn.pkl")  # Make sure this file exists from training

# Make prediction
output = dnn.forward_pass(input_image)
predicted_digit = np.argmax(output)
confidence = np.max(output)

# Display results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Show the image
ax1.imshow(image, cmap="gray")
ax1.set_title(f"Image {image_index}\nActual Label: {actual_label}")
ax1.axis('off')

# Show prediction results
ax2.text(0.1, 0.8, f"Predicted Digit: {predicted_digit}", fontsize=16, fontweight='bold')
ax2.text(0.1, 0.6, f"Actual Digit: {actual_label}", fontsize=16)
ax2.text(0.1, 0.4, f"Confidence: {confidence:.2%}", fontsize=16)
ax2.text(0.1, 0.2, f"Correct: {'✓' if predicted_digit == actual_label else '✗'}", 
         fontsize=16, color='green' if predicted_digit == actual_label else 'red')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

plt.tight_layout()
plt.show()

print(f"\nResults:")
print(f"Predicted digit: {predicted_digit}")
print(f"Actual digit: {actual_label}")
print(f"Confidence: {confidence:.2%}")
print(f"Prediction correct: {'Yes' if predicted_digit == actual_label else 'No'}")