# File: resnet_huggingface_folder.py

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import time
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# Step 1: Set up the environment
# Ensure you have torch and transformers installed. If not, install them using:
# !pip install torch transformers

# Step 2: Download the ResNet model from Hugging Face
model_name = "microsoft/resnet-50"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode

# Define the image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Specify the folder containing images
image_folder = "/home/orangepi/Pictures/"  # Replace with your folder path

# Output log file
log_file = "resnet_output.txt"

# Labels for ImageNet
labels = model.config.id2label

def process_images_batch(image_paths):
    images = []
    for image_path in image_paths:
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img)
        images.append(img_tensor)
    
    inputs = feature_extractor(images=images, return_tensors="pt")
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_classes = logits.argmax(-1).tolist()
    end_time = time.time()
    
    inference_time = end_time - start_time
    return [(os.path.basename(image_paths[i]), labels[predicted_classes[i]], inference_time) for i in range(len(image_paths))]

def process_folder(folder_path, batch_size=4):
    total_images = 0
    correct_predictions = 0  # Assume we have a way to check correctness (requires true labels)
    total_inference_time = 0.0
    log_lines = []

    image_paths = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file_name))]
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        results = process_images_batch(batch_paths)
        
        for image_name, predicted_class, inference_time in results:
            log_lines.append(f"Image: {image_name} - Predicted class: {predicted_class} - Inference time: {inference_time:.4f} seconds")
            total_images += 1
            total_inference_time += inference_time
            
            # Assuming we have a way to get the true label to check accuracy
            # true_label = get_true_label(image_name)
            # if true_label == predicted_class:
            #     correct_predictions += 1

    average_inference_time = total_inference_time / total_images if total_images else 0
    accuracy = (correct_predictions / total_images) * 100 if total_images else 0

    log_lines.append(f"\nTotal images processed: {total_images}")
    log_lines.append(f"Total inference time: {total_inference_time:.4f} seconds")
    log_lines.append(f"Average inference time per image: {average_inference_time:.4f} seconds")
    log_lines.append(f"Accuracy: {accuracy:.2f}%")

    with open(log_file, "w") as f:
        f.write("\n".join(log_lines))

# Process all images in the folder
process_folder(image_folder)
