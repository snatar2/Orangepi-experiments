import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time
import json
import requests
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Load ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
class_labels = json.loads(requests.get(LABELS_URL).text)

# Define the inference function within the script
def load_model():
    model = models.resnet50(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    if torch.cuda.is_available():
        model = model.to('cuda')
    return model

def inference(model, input_batch):
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')

    with torch.no_grad():
        outputs = model(input_batch)
    
    _, preds = torch.max(outputs, 1)
    return preds

# Define the batch processing function
def batch_process(input_dir, output_file, batch_size):
    # Define image transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the model
    model = load_model()

    # Prepare the results
    results = []
    total_inference_time = 0
    total_images = 0

    # Collect image file paths
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('png', 'jpg', 'jpeg')) and os.path.isfile(os.path.join(input_dir, f))]

    # Process images in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        batch_image_names = []

        for image_path in batch_files:
            image = Image.open(image_path).convert('RGB')
            input_tensor = preprocess(image)
            batch_images.append(input_tensor)
            batch_image_names.append(os.path.basename(image_path))

        input_batch = torch.stack(batch_images)

        # Perform inference
        start_time = time.time()
        preds = inference(model, input_batch)
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        total_images += len(batch_image_names)

        # Collect the results for this batch
        for image_name, pred in zip(batch_image_names, preds):
            predicted_class = class_labels[pred.item()]
            results.append(f"Image: {image_name} - Predicted class: {predicted_class} - Inference time: {inference_time:.4f} seconds")

        print(f"Processed batch {i // batch_size + 1}")

    # Save results to the output file
    with open(output_file, 'w') as f:
        for result in results:
            f.write(result + "\n")

        f.write(f"\nTotal images processed: {total_images}\n")
        f.write(f"Total inference time: {total_inference_time:.4f} seconds\n")
        f.write(f"Average inference time per image: {total_inference_time / total_images:.4f} seconds\n")

    print(f"Results saved to {output_file}")

# Main function to set input/output paths and batch size
def main():
    # Set your input directory and output file here
    input_dir = "/home/orangepi/Pictures/"
    output_file = "/home/orangepi/sahana/resnet/batch64_output.txt"
    batch_size = 4  # Set batch size to 32

    batch_process(input_dir, output_file, batch_size)

if __name__ == "__main__":
    main()
