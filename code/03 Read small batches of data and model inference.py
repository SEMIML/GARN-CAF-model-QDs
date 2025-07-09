#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from PIL import Image
import onnxruntime as ort
from collections import Counter
import subprocess
import sys

def preprocess_and_infer(folder_path, session, input_name, output_name, model_name, window_size=24):
    # Define label correspondences
    if model_name == "Shutter_model.onnx":
        labels = ["No", "Yes"]
    elif model_name == "Temperature_model.onnx":
        labels = ["High", "Low", "Suitable"]
    elif model_name == "Reconstruction_model.onnx":
        labels = ["As cap", "c(4×4)", "(2×4)", "Oxidation", "Deoxidation", "(n×6)"]
    else:
        raise ValueError(f"Unrecognized model name: {model_name}")

    # Get all PNG file paths and sort them
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
    image_paths = sorted(image_paths)  # Ensure that it is loaded sorted by name

    num_images = len(image_paths)
    if num_images < window_size:
        raise ValueError(f"There are not enough pictures in the folder {window_size} 。")

    # Preprocess all images
    processed_images = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            gray_img = img.convert('L')  # Convert to greyscale
            resized_img = gray_img.resize((128, 128))  # Resizing
            img_array = np.array(resized_img, dtype=np.float32) / 255.0  # 归一化至 [0, 1]
            processed_images.append(img_array)  # Save preprocessed data

    # Sliding window to generate samples and perform inference
    results = []
    label_counts = Counter()  # For counting the number of labels
    for i in range(num_images - window_size + 1):
        # Extract images from the window (24, 128, 128)
        sample_data = np.stack(processed_images[i:i + window_size], axis=0)  # (24, 128, 128)

        # Adjust dimensions to fit ONNX inputs (1, 24, 128, 128)
        sample_data = np.expand_dims(sample_data, axis=0)

        # Inference using models
        output = session.run([output_name], {input_name: sample_data})
        output_scores = output[0][0]  # Getting the results of the first batch
        predicted_label = labels[np.argmax(output_scores)]  # Get labels based on maximum value index

        results.append((output_scores, predicted_label))  # Save scores and labels
        label_counts[predicted_label] += 1  # Updated Label Statistics

        print(f"Processed sample {i + 1}/{num_images - window_size + 1}: Predicted label = {predicted_label}")

    return results, label_counts

if __name__ == "__main__":
    # Retrieve the model path and folder path from command-line parameters
    onnx_model_path = sys.argv[1]
    folder_path = sys.argv[2]

    print("Finish loading")

    # Get model file name (without path)
    model_name = os.path.basename(onnx_model_path)

    # Load ONNX model and force CPU environment
    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run preprocessing and inference
    results, label_counts = preprocess_and_infer(folder_path, session, input_name, output_name, model_name)

    # Percentage of statistical labelling
    total_samples = sum(label_counts.values())
    label_ratios = {label: count / total_samples for label, count in label_counts.items()}

    # Output results and statistics
    print("\nAll results processed:")
    for idx, (scores, label) in enumerate(results):
        print(f"Sample {idx + 1}: Scores = {scores}, Predicted Label = {label}")

    print("\nLabel counts and proportions:")
    for label, count in label_counts.items():
        print(f"Label '{label}': Count = {count}, Proportion = {label_ratios[label]:.2%}")








