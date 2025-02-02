import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
import torch.nn as nn
from datasets import DatasetDict, load_from_disk
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from typing import Dict, Union


class TBSentimentClassifier:
    """
    A class to encapsulate the end-to-end sentiment analysis workflow for TinyBERT.
    """

    def __init__(self, model_name: str, max_length: int = 128) -> None:
        """
        Initializes the classifier with a given model and tokenizer.

        Args:
            model_name (str): HuggingFace model name.
            max_length (int): Maximum token length for padding/truncation.
        """
        # Temporarily suppress all non-error transformers logging
        transformers_logger = logging.getLogger("transformers")
        current_level = transformers_logger.getEffectiveLevel()
        transformers_logger.setLevel(logging.ERROR)

        try:
            self.model_name: str = model_name
            self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)
            self.model: BertForSequenceClassification = BertForSequenceClassification.from_pretrained(
                model_name, num_labels=3
            )
        finally:
            # Restore the original logging level
            transformers_logger.setLevel(current_level)

        self.max_length: int = max_length
        self.dataset: Union[DatasetDict, None] = None
        self.class_weights: Union[torch.Tensor, None] = None

    def load_and_preprocess_data(self, data_path: str) -> DatasetDict:
        """
        Loads and preprocesses the SST-5 dataset.

        Returns:
            DatasetDict: Preprocessed dataset ready for training.
        """
        self.data_path = data_path
        print("=" * 80)
        print(f"Loading and preprocessing dataset from '{self.data_path}'...")
        print("Data source: Stanford Sentiment Treebank (SST-5) via HuggingFace (https://huggingface.co/datasets/SetFit/sst5/tree/main)\n")
        self.dataset = load_from_disk("./data/sst5_dataset")

        def preprocess_labels(example: Dict[str, str]) -> Dict[str, int]:
            label = example["label_text"]
            if label in ["very negative", "negative"]:
                return {"labels": 0}
            elif label == "neutral":
                return {"labels": 1}
            else:
                return {"labels": 2}
        self.dataset = self.dataset.map(preprocess_labels)
        self.dataset = (
            self.dataset.rename_column("text", "sentence")
            .remove_columns(["label_text"])
        )
        print("Tokenizing dataset:\n")
        self.dataset = self.dataset.map(                                # TODO: Very slow, need to optimize
            lambda x: self.tokenizer(
                x["sentence"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            ),
            batched=True,
        )
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return self.dataset

    def compute_class_weights(self, dataset_name: str = "train") -> None:
        """
        Computes class weights based on the training dataset.

        Args:
            dataset_name (str): The subset of the dataset to compute weights from.
        """
        print("\nComputing class weights...\n")
        labels = self.dataset[dataset_name]["labels"].numpy()
        classes = np.unique(labels)
        self.class_weights = torch.tensor(
            compute_class_weight("balanced", classes=classes, y=labels),
            dtype=torch.float,
        )

    def train(self, training_args: TrainingArguments) -> None:
        """
        Trains the model using the Trainer API.

        Args:
            training_args (TrainingArguments): Training arguments for the model.
        """

        class WeightedLossTrainer(Trainer):
            def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
                super().__init__(*args, **kwargs)
                self.class_weights = class_weights

            def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
                loss = loss_fct(logits, labels)
                return (loss, outputs) if return_outputs else loss

        self.trainer = WeightedLossTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            class_weights=self.class_weights,
        )

        start_time = time.time()    # Start time
        print("Model training in progress...\n")
        self.trainer.train()        # Train the model
        end_time = time.time()      # End time
        training_time = end_time - start_time
        print(f"\nTraining completed in {training_time // 60:.0f} minutes and {training_time % 60:.0f} seconds.\n")

    def save_model_and_tokenizer(self, model_path: str, tokenizer_path: str) -> str:
        """
        Saves the trained model and tokenizer.

        Args:
            model_path (str): Path to save the model.
            tokenizer_path (str): Path to save the tokenizer.
        
        Returns:
            str: Confirmation message.
        """
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(tokenizer_path)
        return f"Model and tokenizer saved under {model_path} and {tokenizer_path}.\n"

    def predict_and_report(self, dataset_name: str = "test") -> None:
        """
        Predicts labels on a dataset and generates evaluation reports.

        Args:
            dataset_name (str): The subset of the dataset to make predictions on.
        """
        # Get predictions
        predictions, labels, _ = self.trainer.predict(self.dataset[dataset_name])
        predicted_labels = np.argmax(predictions, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predicted_labels, average="weighted"
        )

        # Store evaluation results in a dictionary
        eval_results = {
            "Accuracy": accuracy,
            "Precisions": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Classification Report": classification_report(
                labels, predicted_labels, target_names=["negative", "neutral", "positive"]
            )
        }

        # Print evaluation metrics
        print("\n" + "=" * 80)
        print("Evaluation Results:\n")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print("=" * 80 + "\n")
        print("Classification Report:\n\n")
        print(eval_results["Classification Report"])
        print("=" * 80 + "\n")

        # Save evaluation results to a text file
        with open('./output/evaluation_results.txt', 'w') as f:
            for key, value in eval_results.items():
                f.write(f"{key}:\n{value}\n\n")
        print("Evaluation results saved to './output/evaluation_results.txt'.\n")

        # Compute confusion matrix
        print("Generating confusion matrix...\n")
        predictions, labels, _ = self.trainer.predict(self.dataset[dataset_name])
        preds = np.argmax(predictions, axis=1)
        conf_matrix = confusion_matrix(labels, preds)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
        plt.xlabel('Predicted Labels', labelpad=10)
        plt.ylabel('Actual Labels', labelpad=10)
        plt.title('Confusion Matrix', pad=20)

        # Save the plot
        plt.savefig('./output/confusion_matrix.png')
        print("\nConfusion matrix saved to './output/confusion_matrix.png'.\n")

if __name__ == "__main__":
    # Initialize and run the pipeline
    model_name = "huawei-noah/TinyBERT_General_4L_312D"
    max_length = 128
    learning_rate = 2e-5
    batch_size = 8
    num_epochs = 3
    weight_decay = 0.01

    classifier = TBSentimentClassifier(model_name, max_length=max_length)
    classifier.load_and_preprocess_data(data_path="./data/sst5_dataset")
    classifier.compute_class_weights()

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        logging_dir="./logs",
    )

    classifier.train(training_args=training_args)
    classifier.save_model_and_tokenizer(model_path="./model", tokenizer_path="./tokenizer")
    classifier.predict_and_report()