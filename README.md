# BERT Based Sentiment Analysis 

## Project Overview
This project implements a sentiment analysis model using a pre-trained BERT model, enhanced with a sarcasm detection component. The goal is to accurately classify text sentiment (positive, neutral, negative) while accounting for sarcasm, which can invert the true sentiment of a statement.

## Features
-   **BERT-based Sentiment Analysis**: Utilizes `bert-base-uncased` for robust text classification.
-   **Custom Preprocessing**: Includes functions for cleaning text by removing URLs, HTML tags, mentions, and special characters.
-   **Data Loading & Preparation**: Automatically detects text and label columns from a CSV, maps labels to numerical IDs, and splits data into training and validation sets.
-   **Sarcasm Detection Integration**: Incorporates a separate model (`_sar_mod`) to detect sarcasm, which then adjusts the predicted sentiment.
-   **Model Training**: Trains the sentiment classification model using the Hugging Face `Trainer` API.
-   **Evaluation Metrics**: Computes accuracy, precision, recall, and F1-score for model performance assessment.
-   **Model Saving & Loading**: Saves the trained model and tokenizer, and can load a 'stronger' pre-trained model for fine-tuning or inference.
-   **Interactive Inference**: Provides an interactive command-line interface for real-time sentiment prediction with sarcasm awareness.

## Setup

### Dependencies
Ensure you have the following Python libraries installed. The first cell handles `transformers` installation. Other essential libraries are imported at the beginning of the notebook.

```bash
pip install --upgrade transformers pandas numpy torch scikit-learn datasets evaluate
```

### Data
The project expects a CSV file containing text data and corresponding labels. By default, it looks for `IMDB Dataset.csv` at the path specified in `config.DATA_FILE` (`/content/drive/MyDrive/Colab Notebooks/IMDB Dataset.csv`). Please ensure this file is accessible or update `Config.DATA_FILE`.

### Model Paths
-   `Config.OUTPUT_DIR`: Specifies where the trained model and tokenizer will be saved (e.g., `/content/drive/MyDrive/Colab Notebooks/BestModel`).
-   `BERT_PATH`: (Optional) Path to a stronger pre-trained BERT model that can be loaded to overwrite the fine-tuned model (e.g., `/content/drive/MyDrive/Colab Notebooks/savedmodel/`).

## Usage

### 1. Configuration
Modify the `Config` class in **Cell 2**  to adjust parameters such as:
-   `DATA_FILE`: Path to your dataset.
-   `MODEL_NAME`: Base model for `AutoTokenizer` and `AutoModelForSequenceClassification` (default: `bert-base-uncased`).
-   `OUTPUT_DIR`: Directory for saving the trained model.
-   `TRAIN_BATCH_SIZE`, `NUM_EPOCHS`.

### 2. Data Preprocessing and Loading
-   Run **Cell 3**  and **Cell 4**  to load and preprocess your data. The `load_data` function automatically detects common text and label column names. Ensure your dataset has columns like 'tweet', 'content', 'text', 'sentence', or 'review' for text, and 'target', 'sentiment', 'label', 'class', or 'airline_sentiment' for labels.

### 3. Model Training
-   **Cell 5**  initializes the sentiment classification model, defines the `compute_metrics` function, and sets up the `TrainingArguments` and `Trainer`.
-   **Cell 6**  executes the training process. After training, it saves the best model to `Config.OUTPUT_DIR`.
    -   *Note*: This cell also includes functionality to overwrite the newly trained model with a model from `BERT_PATH` if you have a specific, stronger pre-trained model you wish to use.

### 4. Interactive Inference
-   **Cell 7**  provides an interactive loop for real-time sentiment analysis. It takes user input, preprocesses it, detects sarcasm, and then predicts the sentiment, adjusting for sarcasm if detected.

    ```python
    # Example interaction:
    Enter sentence: This is just great, another Monday morning.
    Sentiment: negative (conf=0.987)

    Enter sentence: I absolutely love getting stuck in traffic every day. So much fun!
    Sentiment: negative (conf=0.952) # Sentiment flipped due to sarcasm
    ```

## Code Structure

-   **Cell 1**: Imports necessary libraries and sets up environment variables (e.g., `WANDB_DISABLED`, random seeds, device detection).
-   **Cell 2 (Config)**: Defines project-wide configuration parameters.
-   **Cell 3 (Preprocessing & Data Loader)**: Contains functions for cleaning text and loading the dataset.
-   **Cell 4 (Dataset Preparation)**: Loads data, maps labels, splits data, tokenizes text, and creates `Dataset` objects for training.
-   **Cell 5 (Model + Trainer)**: Initializes the `AutoModelForSequenceClassification`, sets up evaluation metrics, and configures the `Trainer`.
-   **Cell 6 (Train & Save)**: Executes model training and saves the model artifacts.
-   **Cell 7 (BERT Inference)**: Provides an interactive command-line interface for sentiment prediction.
