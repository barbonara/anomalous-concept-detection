import numpy as np
import random
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

### Classes ###

class Dataset:
    """
    A class for handling datasets containing sentences labeled as positive or negative.

    Attributes:
        pos_dataset (list): List of sentences labeled as 'positive'.
        neg_dataset (list): List of sentences labeled as 'negative'.
    """

    def __init__(self):
        """
        Initializes the Dataset object with empty lists for positive and negative sentences.
        """
        self.pos_dataset = []
        self.neg_dataset = []

    def shuffle_datasets(self):
        """
        Shuffles the sentences in both the positive and negative datasets to ensure randomness.
        """
        random.shuffle(self.pos_dataset)
        random.shuffle(self.neg_dataset)

    def get_subset_sentences(self, subset_size):
        """
        Retrieves a subset of sentences from both the positive and negative datasets.

        Args:
            subset_size (int): The number of sentences to retrieve from each dataset.

        Returns:
            tuple: A tuple containing two lists, one of positive and one of negative sentences.
        """
        # Ensure the sentences are shuffled before getting a subset
        self.shuffle_datasets()
        return (self.pos_dataset[:subset_size], self.neg_dataset[:subset_size])
    
    def load_data(self, path_name, pos_label, neg_label):
        """
        Loads data from a CSV file and segregates sentences into positive and negative datasets based on specified labels.

        Args:
            path_name (str): The path to the CSV file containing the data.
            pos_label (str): The label in the CSV that identifies a sentence as positive.
            neg_label (str): The label in the CSV that identifies a sentence as negative.
        """
        df = pd.read_csv(path_name)
        # Check if the necessary columns exist
        if 'Label' not in df.columns or 'Sentence' not in df.columns:
            raise ValueError("Dataframe must contain 'Label' and 'Sentence' columns.")
        
        # Correctly assign sentences to the appropriate dataset based on the provided labels
        self.pos_dataset = df[df['Label'] == pos_label]['Sentence'].tolist()
        self.neg_dataset = df[df['Label'] == neg_label]['Sentence'].tolist()

    def combine_dataset_get_labels(self, total_num_samples):
        """
        Combine positive and negative datasets to create a combined dataset and corresponding labels.

        Args:
            total_num_samples (int): The total number of samples to return.

        Returns:
            tuple: A tuple containing the combined dataset and corresponding labels.
                - combined_dataset (list): The combined dataset containing positive and negative samples.
                - labels (list): The labels corresponding to each sample in the combined dataset.
                    - 1 for positive samples
                    - 0 for negative samples
        """
        num_samples = total_num_samples // 2
        pos_dataset, neg_dataset = self.get_subset_sentences(num_samples)

        combined_dataset = pos_dataset + neg_dataset
        labels = [1] * len(pos_dataset) + [0] * len(neg_dataset)

        return combined_dataset, labels
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.pos_dataset) + len(self.neg_dataset)

class LinearFeatureWrapper():
    """
    A class for wrapping a transformer model to collect activations and explore linear features
    First call call init with arguments. Then use set_layers and self.model to set layers.
    Then capture activations from last tokens of specified layers with get_pos_neg_activations.
    Uses batch processing to avoid issues with large datasets.
    """

    def __init__(self, model_name, dataset_path, pos_labels, neg_labels, memory_saving=False):
        """
        Initializes the HookedModel with a specified transformer model.
        
        Note that because of GPU memory issues, the model is loaded first and then 
        get_pos_neg_activations should be called to collect activations for training probes, PCA, etc.
        
        Args:
            model_name (str): The name of the model to load, as recognized by Hugging Face transformers.
            dataset_path (str): Path for loading model data.
            pos_labels, neg_labels (list): For loading positive and negative examples.
            memory_saving (bool): Implements GPU RAM memory saving measures if activated.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model.eval()  # Set model to evaluation mode
        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer: {e}")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # if memory_saving:
            # self.model.half()

        self.memory_saving = memory_saving
        self.layers = []
        self.layer_indices_to_track = None
        self.activations = {}
        self.hook_handles = []
        self.dataset = Dataset()
        self.dataset.load_data(dataset_path, pos_labels, neg_labels)
        self.pos_layer_activations = {}
        self.neg_layer_activations = {}
        self.direction_layer_vectors = {}

        
    def set_layers(self, layers):
        """
        Sets the layers to track activations for.

        Args:
            layers (iterable): An iterable of torch.nn.Module layers to attach hooks to.
        """
        self.layers = layers
        if self.layer_indices_to_track is None:
            self.layer_indices_to_track = range(len(layers))
        else:
            self.layer_indices_to_track = self.layer_indices_to_track
            

    def setup_hooks(self):
        """
        Sets up forward hooks on specified layers to capture activations.

        Args:
            layers (iterable): An iterable of torch.nn.Module layers to attach hooks to.
        """
        
        def get_activation(name):
            """Defines a hook function that captures activations."""
            def hook(model, input, output):
                self.activations[name] = output[0].cpu().detach()
            return hook

        for j, layer in enumerate(self.layers):
            if j in self.layer_indices_to_track:
                handle = layer.register_forward_hook(get_activation(f'Layer_{j}'))
                self.hook_handles.append(handle)

    def remove_hooks(self):
        """Removes all hooks from the model."""
        for handle in self.hook_handles:
            handle.remove()
    
    def get_last_token_activations(self, dataset: list, max_tokens=None, batch_size=None):
        """
        Captures and returns activations for the last token of sentences in the dataset for specified layers.
        Takes in an arbitrary dataset as a list of input sentences, not self.dataset

        Args:
            dataset (list of str): List of sentences to process.
            layers (iterable): Layers to hook for activation capture.
            max_tokens (int, optional): Maximum number of tokens to consider in each sentence.
            batch_size (int, optional): Number of sentences to process in each batch.

        Returns:
            dict: A dictionary containing layer-wise activations for the last tokens.
        """
        # Set max_tokens to be the largest number of tokens in dataset if not provided.
        if max_tokens is None:
            max_tokens = self.tokenizer(dataset, return_tensors="pt", padding=True, truncation=False, max_length=None)['input_ids'].shape[-1]

        if batch_size is None:
            batch_size = len(dataset)

        layer_last_token_activations = {f'Layer_{j}': [] for j in self.layer_indices_to_track}  # Initialize storage for each layer

        for i in range(0, len(dataset), batch_size):

            print(f"Processing batch {i // batch_size + 1}/{len(dataset) // batch_size + 1}")


            batch_sentences = dataset[i:i + batch_size]
            if not batch_sentences:
                continue

            inputs = self.tokenizer(batch_sentences, max_length=max_tokens, padding='max_length', truncation=True, return_tensors="pt")
            # if self.memory_saving == True:
            #     inputs = {key: val.to(self.device).half() for key, val in inputs.items()}
            # else:
            #     
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            if self.device == 'cuda':
                torch.cuda.empty_cache()  # Clear unused memory

            self.setup_hooks()
            with torch.no_grad():
                outputs = self.model(**inputs)

            self.remove_hooks()

            last_token_indices = (inputs['attention_mask'].sum(dim=1) - 1).tolist()

            # Store only the activations corresponding to the last token for each sentence in each layer
            for j, idx in enumerate(last_token_indices):
                for layer_key in self.activations.keys():
                    layer_last_token_activations[layer_key].append(self.activations[layer_key][j, idx, :].cpu())

            # Clear activations after processing to save memory
            self.activations.clear()

        # Convert lists to tensors for uniformity and easier handling later
        for layer_key in layer_last_token_activations:
            layer_last_token_activations[layer_key] = torch.stack(layer_last_token_activations[layer_key])

        return layer_last_token_activations
    
    def get_pos_neg_activations(self, num_samples, max_tokens, batch_size):
        """
        Calculate the detector direction vectors for each layer. Stores result in class.
        Uses self.dataset.

        Args:
            num_samples (int): number of samples from dataset.
            max_tokens (int): Maximum number of tokens to process.
            batch_size (int): Batch size for processing the datasets.

        Returns:
            dict: A dictionary containing the detector direction vectors for each layer.
        """
        dataset = self.dataset
        pos_dataset, neg_dataset = dataset.get_subset_sentences(num_samples)

        self.pos_layer_activations = self.get_last_token_activations(pos_dataset, max_tokens, batch_size)
        self.neg_layer_activations = self.get_last_token_activations(neg_dataset, max_tokens, batch_size)

    
    def calculate_detector_direction(self):
        """
        Calculate the detector direction vectors for each layer.

        Args:
            pos_dataset (Dataset): Dataset containing positive examples.
            neg_dataset (Dataset): Dataset containing negative examples.
            max_tokens (int): Maximum number of tokens to process.
            batch_size (int): Batch size for processing the datasets.

        Returns:
            dict: A dictionary containing the detector direction vectors for each layer.
        """
        if len(self.pos_layer_activations) == 0 or len(self.neg_layer_activations) == 0:
            raise Exception("Positive and negative activations haven't been generated. Use self.get_pos_neg_activations.")
        
        pos_activations = self.pos_layer_activations
        neg_activations = self.neg_layer_activations

        direction_layer_vectors = {}
        for layer_name in pos_activations.keys():
            # Extract activations and convert to numpy arrays
            pos_layer_activations = pos_activations[layer_name].cpu().numpy()
            neg_layer_activations = neg_activations[layer_name].cpu().numpy()

            # Compute the difference in means between the two categories for each layer
            direction_layer_vectors[layer_name] = np.mean(pos_layer_activations, axis=0) - np.mean(neg_layer_activations, axis=0)

        self.direction_layer_vectors = direction_layer_vectors
        # return direction_layer_vectors
    
    def plot_pca(self):
        """
        Plots PCA of the last token activations for specified layers in a grid layout.

        Args:
            pos_activations (dict): Dictionary of activations for positive examples, keyed by layer.
            neg_activations (dict): Dictionary of activations for negative examples, keyed by layer.
            layers (list): List of layer names for which PCA plots are generated.
        """
        pos_activations = self.pos_layer_activations
        neg_activations = self.neg_layer_activations

        layer_names = list(pos_activations.keys())

        num_layers = len(layer_names)
        rows = int(np.ceil(np.sqrt(num_layers)))  # Rows in the subplot grid
        cols = int(np.ceil(num_layers / rows))  # Columns in the subplot grid
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # Set up the subplot grid
        axes = axes.flatten() if num_layers > 1 else [axes]  # Flatten if grid is 2D

        for j, ax in enumerate(axes):
            if j >= num_layers:
                fig.delaxes(ax)  # Remove extra axes if any
                continue
            layer_name = layer_names[j]  # Construct the layer name
            try:
                # Combine datasets for current layer
                combined_activations = np.vstack((pos_activations[layer_name], neg_activations[layer_name]))
                labels = np.array([0] * len(pos_activations[layer_name]) + [1] * len(neg_activations[layer_name]))

                # Perform PCA
                pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
                reduced_activations = pca.fit_transform(combined_activations)

                # Plotting on the specified axis
                ax.scatter(reduced_activations[labels == 0, 0], reduced_activations[labels == 0, 1], c='red', label='Positive', alpha=0.5)
                ax.scatter(reduced_activations[labels == 1, 0], reduced_activations[labels == 1, 1], c='blue', label='Negative', alpha=0.5)
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                ax.set_title(layer_name)
                ax.legend()
            except Exception as e:
                print(f"Error processing {layer_name}: {e}")

        plt.tight_layout()
        plt.show()

    def evaluate_MD_detector(self, dataset: list, labels, max_tokens=None, batch_size=None):
        """

        Evaluate the detector's accuracy for each layer. And plot results.

        Args:
            activations (dict): A dictionary containing activations for each layer.
            direction_vectors (dict): A dictionary containing direction vectors for each layer.
            labels (list): A list of labels corresponding to the activations and direction vectors.
            layer_indices (list): A list of layer indices to evaluate.

        Returns:
            Accuracies
        """
        def calculate_accuracy(activations, detection_vectors, labels):
            """
            Calculate the accuracy of classification where the sign of the dot product of activations and detection vectors
            should correspond to binary labels. Positive values predict label '1' and negative values predict label '0'.

            Parameters:
            - activations (np.array): Array of activations from which predictions are derived.
            - labels (list of int): Corresponding list of binary labels (1s and 0s) to compare against predictions.
            - detection_vectors (np.array): Vector used to transform activations into a scalar prediction value.

            Returns:
            - float: The accuracy of the predictions, represented as a fraction between 0 and 1.
            """
            # Calculate values by projecting activations onto the detection vectors
            values = np.dot(activations, detection_vectors)
            correct_count = 0
            total_count = len(values)

            for value, label in zip(values, labels):
                # Predict 1 if value is positive, 0 if negative
                predicted_label = 1 if value >= 0 else 0
                # Increment correct count if prediction matches the label
                if predicted_label == label:
                    correct_count += 1

            # Calculate accuracy as the proportion of correct predictions
            accuracy = correct_count / total_count
            return accuracy
        
        activations = self.get_last_token_activations(dataset, max_tokens, batch_size)
        accuracies = {}
        layer_names = list(self.pos_layer_activations.keys())

        for layer_name in layer_names:
            # Calculate accuracy for the current layer
            accuracy = calculate_accuracy(activations[layer_name], self.direction_layer_vectors[layer_name], labels)
            accuracies[layer_name] = accuracy

        plt.figure(figsize=(10, 5))
        # layer_names_sorted = sorted(accuracies.keys()) # Optional: Sort layer names if needed
        accuracy_values = [accuracies[name] for name in layer_names]

        # Create a bar plot of accuracies
        plt.bar(layer_names, accuracy_values, color='blue')
        plt.xlabel('Layer Names')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Layer')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)  # Set the y-axis limits from 0 to 1
        plt.tight_layout() # Adjust layout to make room for rotated x-axis labels
        plt.show()

        return accuracies
    def evaluate_MD_detector(self, dataset: list, labels, max_tokens=None, batch_size=None):
        """
        Evaluate the detector's accuracy for each layer based on the mean direction vector. Plots the results.

        Args:
            dataset (list): A list of sentences to process.
            labels (list): A list of labels corresponding to the dataset.
            max_tokens (int, optional): Maximum number of tokens to consider in each sentence.
            batch_size (int, optional): Number of sentences to process in each batch.

        Returns:
            dict: Accuracies per layer
        """
        def calculate_accuracy(activations, detection_vectors, labels):
            """
            Calculate the accuracy of classification where the sign of the dot product of activations and detection vectors
            should correspond to binary labels. Positive values predict label '1' and negative values predict label '0'.

            Parameters:
            - activations (np.array): Array of activations from which predictions are derived.
            - labels (list of int): Corresponding list of binary labels (1s and 0s) to compare against predictions.
            - detection_vectors (np.array): Vector used to transform activations into a scalar prediction value.

            Returns:
            - float: The accuracy of the predictions, represented as a fraction between 0 and 1.
            """
            values = np.dot(activations, detection_vectors)
            predictions = np.where(values >= 0, 1, 0)
            accuracy = np.mean(predictions == labels)
            return accuracy

        # Get activations for the dataset
        activations = self.get_last_token_activations(dataset, max_tokens, batch_size)
        accuracies = {}
        layer_names = list(self.pos_layer_activations.keys())

        for layer_name in layer_names:
            # Calculate accuracy for the current layer using the mean direction vector
            layer_activations = activations[layer_name].cpu().numpy() if hasattr(activations[layer_name], 'cpu') else activations[layer_name]
            detection_vector = self.direction_layer_vectors[layer_name]
            accuracy = calculate_accuracy(layer_activations, detection_vector, labels)
            accuracies[layer_name] = accuracy

        # Plotting the accuracies
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(accuracies)), accuracies.values(), color='blue')
        plt.xticks(range(len(accuracies)), layer_names, rotation=45)
        plt.xlabel('Layer Names')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Layer for MD Detector')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

        return accuracies

    def train_and_evaluate_probes(self):
        """
        Trains linear classifiers (probes) on the activations and evaluates their accuracy.
        The function assumes that activations for positive and negative classes have been captured
        and stored in self.pos_layer_activations and self.neg_layer_activations respectively.
        
        Returns:
            dict: A dictionary containing accuracy scores for probes on each layer.
        """
        accuracies = {}
        layer_names = []  # To store layer names for plotting

        # Loop over each layer for which activations have been captured
        for layer in self.pos_layer_activations:
            # Get activations from the layer for both classes
            pos_activations = self.pos_layer_activations[layer]
            neg_activations = self.neg_layer_activations[layer]
            
            # Combine the activations and create labels
            X = np.vstack((pos_activations, neg_activations))
            y = np.array([1] * len(pos_activations) + [0] * len(neg_activations))
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train a logistic regression model
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(X_train, y_train)
            
            # Predict on the test set and calculate accuracy
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies[layer] = accuracy
            layer_names.append(layer)  # Append layer name for plotting

        # Plot the accuracies
        plt.figure(figsize=(10, 5))
        plt.bar(layer_names, accuracies.values(), color='skyblue')
        plt.xlabel('Layer Names')
        plt.ylabel('Accuracy')
        plt.title('Probing Accuracy Across Layers')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)  # Optionally set the y-axis limits to 0-1 for better comparison
        plt.tight_layout()  # Adjust layout to handle longer layer names
        plt.show()

        return accuracies
    

    



### Functions ###

def plot_pca(pos_activations, neg_activations, layers):
    """
    Plots PCA of the last token activations for specified layers in a grid layout.

    Args:
        pos_activations (dict): Dictionary of activations for positive examples, keyed by layer.
        neg_activations (dict): Dictionary of activations for negative examples, keyed by layer.
        layers (list): List of layer names for which PCA plots are generated.
    """
    num_layers = len(layers)
    rows = int(np.ceil(np.sqrt(num_layers)))  # Rows in the subplot grid
    cols = int(np.ceil(num_layers / rows))  # Columns in the subplot grid
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # Set up the subplot grid
    axes = axes.flatten() if num_layers > 1 else [axes]  # Flatten if grid is 2D

    for j, ax in enumerate(axes):
        if j >= num_layers:
            fig.delaxes(ax)  # Remove extra axes if any
            continue
        layer_name = f'Layer_{j}'  # Construct the layer name
        try:
            # Combine datasets for current layer
            combined_activations = np.vstack((pos_activations[layer_name], neg_activations[layer_name]))
            labels = np.array([0] * len(pos_activations[layer_name]) + [1] * len(neg_activations[layer_name]))

            # Perform PCA
            pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
            reduced_activations = pca.fit_transform(combined_activations)

            # Plotting on the specified axis
            ax.scatter(reduced_activations[labels == 0, 0], reduced_activations[labels == 0, 1], c='red', label='Positive', alpha=0.5)
            ax.scatter(reduced_activations[labels == 1, 0], reduced_activations[labels == 1, 1], c='blue', label='Negative', alpha=0.5)
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_title(layer_name)
            ax.legend()
        except Exception as e:
            print(f"Error processing {layer_name}: {e}")

    plt.tight_layout()
    plt.show()

def compute_direction_vectors(pos_activations, neg_activations):
    """
    Computes the direction vector (difference in means of activations) for positive vs. negative
    sentences across specified layers in a neural network model.

    Parameters:
    - pos_activations (dict): Dictionary of activations for sentences classified as positive,
                              where keys are layer names and values are activation tensors.
    - neg_activations (dict): Dictionary of activations for sentences classified as negative,
                              where keys are layer names and values are activation tensors.

    Returns:
    - dict: A dictionary with layer names as keys and direction vectors as values.
    """
    direction_vectors = {}

    for layer_name in pos_activations.keys():
        # Extract activations and convert to numpy arrays
        pos_layer_activations = pos_activations[layer_name].cpu().numpy()
        neg_layer_activations = neg_activations[layer_name].cpu().numpy()

        # Compute the difference in means between the two categories for each layer
        direction_vectors[layer_name] = np.mean(pos_layer_activations, axis=0) - np.mean(neg_layer_activations, axis=0)

    return direction_vectors




def evaluate_detector(activations, direction_vectors, labels, layer_indices):
    """
    Evaluate the detector's accuracy for each layer. And plot results.

    Args:
        activations (dict): A dictionary containing activations for each layer.
        direction_vectors (dict): A dictionary containing direction vectors for each layer.
        labels (list): A list of labels corresponding to the activations and direction vectors.
        layer_indices (list): A list of layer indices to evaluate.

    Returns:
        Accuracies
    """
    accuracies = {}
    layer_names = [f'Layer_{i}' for i in layer_indices]

    for layer_name in layer_names:
        # Calculate accuracy for the current layer
        accuracy = calculate_accuracy(activations[layer_name], direction_vectors[layer_name], labels)
        accuracies[layer_name] = accuracy

    plt.figure(figsize=(10, 5))
    # layer_names_sorted = sorted(accuracies.keys()) # Optional: Sort layer names if needed
    accuracy_values = [accuracies[name] for name in layer_names]

    # Create a bar plot of accuracies
    plt.bar(layer_names, accuracy_values, color='blue')
    plt.xlabel('Layer Names')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Layer')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)  # Set the y-axis limits from 0 to 1
    plt.tight_layout() # Adjust layout to make room for rotated x-axis labels
    plt.show()

    return accuracies