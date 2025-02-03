import json
import html
import matplotlib.pyplot as plt
import random
import re
import torch
import os
import pandas as pd
import sys
from textblob import TextBlob
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS


class SentimentAnalyzer:
    """Handles sentiment analysis using a pre-trained TinyBERT model."""

    def __init__(self, model_path: str, tokenizer_path: str):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}

    def predict(self, text: str) -> str:
        """Predicts sentiment for a given text."""
        inputs = self.tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )  
        with torch.no_grad():
            output = self.model(**inputs)
        predicted_class = torch.argmax(output.logits, dim=1).item()
        return self.label_map[predicted_class]

    def predict_batch(self, texts: list[str]) -> list[str]:
        """Predicts sentiments for a batch of texts."""
        dataset = TextDataset(texts)
        batch_size = 32
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_predictions = []
        total_batches = len(dataloader)

        with torch.no_grad():
            for counter, batch in enumerate(dataloader, start=1):
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                outputs = self.model(**inputs)
                all_predictions.extend(torch.argmax(outputs.logits, dim=1).tolist())
                
                # Print progress
                sys.stdout.write(f"\rProcessing batch {counter}/{total_batches}...")   # \r moves cursor back to the start of the line
                sys.stdout.flush()     # Ensures immediate output to the console

        print("\n") # Move to a new line after batch processing is complete
        return [self.label_map[pred] for pred in all_predictions]


class DataProcessor:
    """Handles data loading, cleaning, and preprocessing."""

    URL_REGEX = re.compile(r'http\S+|www\S+|https\S+')
    MENTION_HASHTAG_REGEX = re.compile(r'@\w+|#\w+')
    SPECIAL_CHAR_REGEX = re.compile(r'[^\w\s,\'\.?!]')
    EMOJI_REGEX = re.compile(r'[\U00010000-\U0010FFFF]')
    SPACE_REGEX = re.compile(r'\s+')

    def __init__(self, file_path: str):
        self.data = self.load_json(file_path)
        self.df = self.normalize_data()

    def load_json(self, file_path: str) -> list:
        """Loads JSON data from the given file path."""
        with open(file_path, "r") as file:
            return json.load(file)

    def normalize_data(self) -> pd.DataFrame:
        """Normalizes and processes the JSON data into a DataFrame."""
        posts_df = pd.json_normalize(self.data)
        comments_data = [comment for post in self.data for comment in post["comments"]]
        comments_df = pd.DataFrame(comments_data)
        posts_df.drop("comments", axis=1, inplace=True)
        combined_df = pd.concat([posts_df, comments_df], ignore_index=True)
        combined_df = combined_df[
            [
                "post_id",
                "comment_id",
                "title",
                "text",
                "created_time",
                "subreddit",
                "url",
            ]
        ]
        combined_df = combined_df[
            combined_df["text"].str.strip().notna()
            & (combined_df["text"].str.strip() != "")
        ]
        combined_df = combined_df[combined_df["text"].str.split().str.len() > 12]
        combined_df.drop_duplicates(subset="text", inplace=True)
        combined_df["text"] = combined_df["text"].apply(self.clean_text)
        return combined_df

    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans the input text by removing URLs, mentions, special characters, and extra spaces."""
        text = DataProcessor.URL_REGEX.sub('', text)
        text = DataProcessor.MENTION_HASHTAG_REGEX.sub('', text)
        text = DataProcessor.SPECIAL_CHAR_REGEX.sub('', text)
        text = DataProcessor.EMOJI_REGEX.sub('', text)
        text = DataProcessor.SPACE_REGEX.sub(' ', text).strip()
        text = html.unescape(text)
        text = text.replace('\u200B', '')
        return text


class TextDataset(Dataset):
    """Dataset class for handling text data."""

    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


class SentimentAnalysisPipeline:
    """Executes the full sentiment analysis pipeline."""

    def __init__(self, data_path: str, model_path: str, tokenizer_path: str, output_path: str):
        self.processor = DataProcessor(data_path)
        self.analyzer = SentimentAnalyzer(model_path, tokenizer_path)
        self.vader = SentimentIntensityAnalyzer()
        self.emoji_map = {"positive": "🌟", "neutral": "⚖️", "negative": "🤔"}
        self.output_path = output_path

    def ensure_output_directory(self):
        """Ensures that the output directory exists."""
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def predict_random_sample(self):
        """Predicts sentiment for a randomly selected sample and prints it."""
        cleaned_df = self.processor.df
        random_index = random.randint(0, len(cleaned_df) - 1)
        random_text = cleaned_df.iloc[random_index]["text"]
        predicted_label = self.analyzer.predict(random_text)

        print()
        print("=" * 50)
        print(f"📝 Sample Post:\n\n{random_text}\n")
        print("=" * 50)
        print(f"🔍 Predicted Sentiment: {predicted_label}")
        print("=" * 50 + "\n")

    def sentiment_distribution(self):
        """Runs the sentiment analysis pipeline and prints results."""
        print("Calculating sentiment distribution...\n")
        cleaned_df = self.processor.df
        cleaned_df["predicted_sentiment"] = self.analyzer.predict_batch(
            cleaned_df["text"].tolist()
        )

        # Print sentiment distribution
        sentiment_counts = cleaned_df["predicted_sentiment"].value_counts()
        sentiment_percentages = (sentiment_counts / sentiment_counts.sum()) * 100
        sentiment_counts_formatted = sentiment_counts.reset_index()
        sentiment_percentages_formatted = sentiment_percentages.reset_index(drop=True)
        sentiment_counts_formatted.columns = ["Sentiment", "Count"]
        sentiment_counts_formatted["Percentage"] = sentiment_percentages_formatted.map(
            "{:.2f}%".format
        )

        print("=" * 50)
        print("Sentiment Distribution:\n")
        print(sentiment_counts_formatted.to_string(index=False))
        print("=" * 50)

        # Print overall sentiment
        overall_sentiment = sentiment_counts.idxmax()

        print(
            f"Overall Sentiment: {overall_sentiment} {self.emoji_map[overall_sentiment]}"
        )
        print("=" * 50 + "\n")

        # Save sentiment distribution to a txt file
        with open("./output/sentiment_distribution.txt", "w") as f:
            f.write("Sentiment Distribution:\n\n")
            f.write(sentiment_counts_formatted.to_string(index=False))
            f.write("\n\nOverall Sentiment: ")
            f.write(f"{overall_sentiment}")

        print(
            "Sentiment distribution saved to './output/sentiment_distribution.txt'.\n"
        )
    
    def classify_vader_score(compound: float) -> str:
        """Classify the sentiment of the compound score."""
        if compound >= 0.05:
            return "positive"
        elif compound <= -0.05:
            return "negative"
        else:
            return "neutral"

    def sentiment_intensity(self):
        """Calculates sentiment intensity using VADER and classifies the results."""
        print("Calculating sentiment intensity...\n")
        cleaned_df = self.processor.df
        vader_analyzer = SentimentIntensityAnalyzer()

        # Calculate VADER compound scores
        cleaned_df['vader_compound'] = cleaned_df['text'].apply(lambda x: vader_analyzer.polarity_scores(x)['compound'])
        # Classify the compound scores
        cleaned_df['vader_sentiment'] = cleaned_df['vader_compound'].apply(SentimentAnalysisPipeline.classify_vader_score)
        
        intensity_counts = cleaned_df['vader_sentiment'].value_counts()
        intensity_percentages = (intensity_counts / intensity_counts.sum()) * 100
        
        intensity_counts_formatted = intensity_counts.reset_index()
        intensity_percentages_formatted = intensity_percentages.reset_index(drop=True)
        intensity_counts_formatted.columns = ['Sentiment', 'Count']
        intensity_counts_formatted['Percentage'] = intensity_percentages_formatted.map("{:.2f}%".format)

        # Display sentiment intensity distribution
        print("Sentiment Intensity Distribution:\n")
        print(intensity_counts_formatted.to_string(index=False))
        print("=" * 50)

        # Get overall sentiment intensity
        overall_intensity = intensity_counts.idxmax()

        # Print overall sentiment intensity
        print(
            f"Overall Sentiment Intensity: {overall_intensity} {self.emoji_map[overall_intensity]}"
        )
        print("=" * 50 + "\n")

        # Save sentiment intensity distribution to a txt file
        with open("./output/sentiment_intensity.txt", "w") as f:
            f.write("Sentiment Intensity Distribution:\n\n")
            f.write(intensity_counts_formatted.to_string(index=False))
            f.write("\n\nOverall Sentiment Intensity: ")
            f.write(f"{overall_intensity}")

        print("Sentiment intensity saved to './output/sentiment_intensity.txt'.\n")

    def generate_wordclouds(self):
        """Generates a word cloud per sentiment for nuons and adjectives based on the text data."""
        print("Generating word clouds...\n")
        cleaned_df = self.processor.df

        # Extract nouns and adjectives
        def extract_nouns_adjectives(text: str) -> str:
            blob = TextBlob(text)
            nouns_adjectives = [
                word
                for word, tag in blob.tags
                if tag in ("NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS")
            ]
            return " ".join(nouns_adjectives)

        cleaned_df["nouns_adjectives"] = cleaned_df["text"].apply(
            extract_nouns_adjectives
        )

        # Generate word cloud for nouns and adjectives
        def show_wordcloud(sentiment: str):
            # Filter DataFrame based on sentiment
            sentiment_df = cleaned_df[cleaned_df["predicted_sentiment"] == sentiment]

            # Combine all the nouns and adjectives
            combined_text = " ".join(sentiment_df["nouns_adjectives"].tolist())

            # Generate word cloud
            wordcloud = WordCloud(
                width=800, height=400, background_color="white", stopwords=STOPWORDS
            ).generate(combined_text)

            # Display word cloud
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.title(
                f"{sentiment.capitalize()} Sentiment Word Cloud", pad=20, fontsize=16
            )
            plt.axis("off")

            # Save the word cloud
            plt.savefig(f"./output/{sentiment}_wordcloud.png")
            print(
                f"Word cloud for {sentiment} sentiment saved to './output/{sentiment}_wordcloud.png'.\n"
            )

        # Generate word cloud for each sentiment
        for sentiment in ["positive", "neutral", "negative"]:
            show_wordcloud(sentiment)

    # Generate sentiment trend analysis over time
    def sentiment_trend_analysis(self):
        """Generates a polarity trend analysis over time."""
        print("Generating sentiment trend analysis over time...\n")
        cleaned_df = self.processor.df

        # Convert the created_time column to datetime
        cleaned_df["created_time"] = pd.to_datetime(cleaned_df["created_time"])

        # Extract the date from the created_time column
        cleaned_df["date"] = cleaned_df["created_time"].dt.date

        # Group by date and sentiment and calculate the average polarity score
        polarity_trend = (
            cleaned_df.groupby(["date", "predicted_sentiment"])
            .size()
            .unstack()
            .fillna(0)
        )

        # Plot the polarity trend over time
        plt.figure(figsize=(12, 6))
        polarity_trend.plot(ax=plt.gca())
        plt.title("Sentiment Trend Over Time", pad=20, fontsize=16)
        plt.xlabel("Date", labelpad=10)
        plt.ylabel("Number of Comments", labelpad=10)
        plt.legend(title="Sentiment")
        plt.grid(True)

        # Save the plot
        plt.savefig("./output/sentiment_trend.png")
        print(
            "Sentiment trend analysis over time saved to './output/sentiment_trend.png'.\n"
        )


if __name__ == "__main__":
    pipeline = SentimentAnalysisPipeline(
        data_path="./data/reddit_data.json",
        model_path="./model",
        tokenizer_path="./tokenizer",
        output_path="./output",
    )
    pipeline.ensure_output_directory()
    pipeline.predict_random_sample()
    pipeline.sentiment_distribution()
    pipeline.sentiment_intensity()
    pipeline.generate_wordclouds()
    pipeline.sentiment_trend_analysis()
    print("Sentiment analysis pipeline completed successfully.\n")
