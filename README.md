# Sentiment Analysis Project

## Project Overview

This sentiment analysis project was conducted as a solo project within the BeCode Data Science and AI Bootcamp over the course of 5 days. The objective of this project was to analyze public reactions to Netflix's series "The Queen's Gambit" through sentiment analysis using data scraped from Reddit. The project combines web scraping, natural language processing (NLP), and data analysis to provide insights into audience sentiments around the Netflix series 'The Queen's Gambit' using data scraped from Reddit.

## Project Components

* **Web Scraping:** Utilized the Reddit API to scrape 10,000 posts containing the hashtag #queengambit.

* **Data Preprocessing:** Cleaned and prepared the data for sentiment analysis, including tokenization, stop-word removal, and normalization.

* **Model Selection:** Chose TinyBERT for its efficiency in fine-tuning on CPU resources, due to constraints.

* **Model Fine-Tuning:** Fine-tuned TinyBERT on a simplified version of the SST5 dataset, reducing the labels from 5 to 3 (positive, neutral, negative).

* **Prediction Generation:** Generated sentiment predictions for the scraped Reddit data.

* **Further Analysis:** Conducted in-depth analysis including:

  * Sentiment Distribution Analysis

  * Sentiment Intensity Analysis

  * Sentiment Trend Analysis

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd sentiment-analysis-project
```

2. Set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate # On Windows use: venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Download SST5 Dataset and save under "./data/"

```bash
from datasets import load_dataset
dataset = load_dataset("SetFit/sst5")
```

## Usage

1. Web Scraping:

    * Use the Reddit API to scrape data related to "The Queen's Gambit".

2. Fine-tuning TinyBERT:

    * Fine-tune TinyBERT on the SST5 dataset, reduced to three sentiment classes (positive, neutral, negative).

3. Sentiment Analysis:

    * Generate predictions on the scraped Reddit data.

4. Data Analysis:

    * Conduct various analyses, including sentiment distribution, intensity, and trends over time.

## Methodology

### Web Scraping

Data was scraped using the Reddit API, focusing on posts with the hashtag #queengambit. A total of 10,000 posts were collected.

### Sentiment Analysis

* Model: TinyBERT

* Dataset: SST5 (reduced to 3 classes)

* Techniques: Preprocessing, Fine-tuning, and Prediction

### Data Analysis

* Sentiment Distribution: Frequency of positive, neutral, and negative sentiments.

* Sentiment Intensity: Strength of the sentiment expressed.

* Sentiment Trends: Changes in sentiment over time.

## Results

* Positive, neutral, and negative sentiment distribution across Reddit posts.

* Insights into how sentiment evolved over the series' airing and its aftermath.

* Identification of significant spikes or shifts in public opinion.

## Conclusion

This project demonstrated the efficacy of combining web scraping and NLP for sentiment analysis. Fine-tuning TinyBERT provided accurate sentiment classification, and the subsequent analyses yielded valuable insights into public opinion on "The Queen's Gambit".

## Future Work

Expansion to other platforms like Twitter or Facebook.

Deployment of the model as an API for real-time sentiment analysis.

## Acknowledgements

* BeCode Data Science and AI Bootcamp

* Hugging Face for the TinyBERT model

* Reddit for data availability
