import numpy as np
import torch
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

import utils

# Global variables
num_test_samples = 64 // 2 # Number of samples to test the detector
num_detector_samples = 64 # Number of samples to construct the detector
batch_size = 64 # Batch size for detector construction
max_length = None # If none, uses maximum length of the dataset

# Load model and dataset
model = utils.HookedModel("openai-community/gpt2")
model.set_layers(model.model.transformer.h)

# Get dataset for constructing direction detector
dataset = utils.Dataset()
dataset.load_data('datasets/ilikecats_20000.csv', 'Animal', 'Non-Animal')

# Get dataset for testing direction detector
test_dataset = dataset = utils.Dataset()
test_dataset.load_data('datasets/(non)animal_2000.csv', 'Animal', 'Non-Animal')
test_data, labels = test_dataset.combine_dataset_get_labels(num_test_samples)

# Construct direction detector vector
animal_detection_vectors = model.calculate_detector_direction(dataset, num_detector_samples, max_length, batch_size)

# Create test activations dataset
test_activations = model.get_last_token_activations(test_data, max_tokens = max_length, batch_size = batch_size)

# Evaluate detector
accuracies = utils.evaluate_detector(test_activations, animal_detection_vectors, labels, model.layer_indices_to_track)
for acuracy in accuracies.values():
    print(acuracy)