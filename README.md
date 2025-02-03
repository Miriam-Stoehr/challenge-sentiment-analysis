# Sentiment Analysis Project

## Project Overview

This sentiment analysis project was conducted as a solo project within the BeCode Data Science and AI Bootcamp over the course of 5 days. The objective of this project was to analyze public reactions on the Netflix's series "The Queen's Gambit" through sentiment analysis using data scraped from Reddit. The project combines web scraping, natural language processing (NLP), and data analysis to provide insights into audience sentiments around the Netflix series 'The Queen's Gambit'.

## Project Components

### Web Scraping

Initially, platforms like X (formerly Twitter) were considered for data collection. However, due to financial constraints associated with the X API and legal considerations regarding data scraping, the Reddit API emerged as a viable alternative. Data was collected from various subreddits, including:

- r/QueensGambit
- r/Netflix
- r/TVShows

The search query "Queen's Gambit" was used to identify relevant posts. Recognizing that posts alone might not capture the full spectrum of user sentiment, comments were also scraped. A challenge encountered was that some comments deviated from the main topic. To address this, the following preprocessing steps were implemented:

- **Filtering**: Removed comments that were too short, contained only links, emojis, or were empty.
- **Future Consideration**: Developing methods to include only comments directly related to the primary topic.

### Data Preprocessing

The collected data underwent several preprocessing steps to prepare it for sentiment analysis:

- **Tokenization**: Breaking down text into individual tokens.
- **Cleaning**: Removing unnecessary elements such as links, special characters, and emojis.
- **Filtering**: Excluding comments that were too short or irrelevant to ensure data quality.

### Model Selection

Several models were evaluated for sentiment analysis:

- **BERT**: While powerful, BERT was computationally intensive for fine-tuning in this context.
- **DistilBERT**: Offered a more resource-efficient alternative but still posed challenges in terms of computational demands.
- **TinyBERT**: Provided a balance between performance and resource efficiency, delivering slightly better evaluation results and being more suitable for the available computational resources.
- **Hidden Markov Model (HMM)**: Inspired by resources such as [this GitHub repository](https://github.com/zestones/HMM-Sentiment-Analysis/tree/main) and studies (see [IEEE](https://ieeexplore.ieee.org/document/8885272) and [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0950705121005943)), HMM was tested but yielded lower performance (accuracy of 0.15). This indicated a need for more complex data preprocessing or a different approach compared to transformer models.

### Model Fine-Tuning

For fine-tuning TinyBERT, the Stanford Sentiment Treebank with 5 labels [SST-5](https://huggingface.co/datasets/SetFit/sst5) dataset was utilized. This dataset comprises 11,855 single sentences extracted from movie reviews, each labeled as very positive, positive, neutral, negative, or very negative. Due to the project's requirements and to enhance performance, the five sentiment categories were consolidated into three: positive, neutral, and negative. This reduction led to class imbalance issues, which were addressed by:

- **Oversampling**: Attempted for minority classes but did not yield significant improvements.
- **Weighted Approach**: Applied class weights during training, resulting in slight performance enhancements.

Future work may involve exploring alternative datasets or models capable of handling more than three sentiment classes.

### Model Performance

The fine-tuned TinyBERT model achieved the following evaluation metrics:

- **Accuracy**: 0.713
- **Precision**: 0.719
- **Recall**: 0.713
- **F1 Score**: 0.715

**Classification Report**:

| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Negative  | 0.78      | 0.74   | 0.76     | 912     |
| Neutral   | 0.34      | 0.37   | 0.36     | 389     |
| Positive  | 0.82      | 0.83   | 0.83     | 909     |

The model demonstrated high performance in classifying positive and negative sentiments but faced challenges with neutral sentiments, likely due to class imbalance.

### Further Analysis

Post-prediction analyses included:

- **Sentiment Distribution**: Examining the frequency of positive, neutral, and negative sentiments across the dataset.
- **Sentiment Intensity**: Assessing the strength of expressed sentiments.
- **Sentiment Trends**: Observing changes in sentiment over time.

## File Structure

```plaintext

    challenge-sentiment-analysis/
    ├── data/
    │   ├── sst5_dataset/
    │   │   ├── test/
    │   │   ├── train/
    │   │   ├── validation/
    │   │   └── dataset_dict
    │   └── reddit_data.json
    ├── model/
    │   ├── config.json
    │   └── model.safetensors
    ├── output/
    │   ├── confusion_matrix.png
    │   ├── evaluation.txt
    │   ├── sentiment_distribution.txt
    │   ├── sentiment_intensity.txt
    │   ├── negative_wordcloud.png
    │   ├── neutral_wordcloud.png
    │   ├── positive_wordcloud.png
    │   └── sentiment_trend.png
    ├── results/
    ├── tokenizer/
    │   ├── special_tokens_map.json
    │   ├── tokenizer_config.json
    │   └── vocab.txt
    ├── 1-scraping.py
    ├── 2-model-training.py
    ├── 3-prediction.py
    ├── config.json
    └── README.md

```

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

5. Set up configurations
   * Register as developer at reddit.com
   * Register your project and enter your credentials in the `config.json` file.

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

Data was scraped using the Reddit API, focusing on posts with the hashtag #queensgambit. A total of 10,000+ posts were collected.

### Sentiment Analysis

* Model: TinyBERT

* Dataset: [SST5](https://huggingface.co/datasets/SetFit/sst5) (reduced to 3 classes)

* Techniques: Preprocessing, Fine-tuning, and Prediction

### Data Analysis

* Sentiment Distribution: Frequency of positive, neutral, and negative sentiments.

* Sentiment Intensity: Strength of the sentiment expressed.

* Sentiment Trends: Changes in sentiment over time.

## Results

* Positive, neutral, and negative sentiment distribution across Reddit posts.

* Intensity of sentiments and most prominent words for each category.

* Insights into how sentiment evolved over the series' airing and its aftermath.

## Conclusion

This project demonstrated the efficacy of combining web scraping and NLP for sentiment analysis. Fine-tuning TinyBERT provided accurate sentiment classification, and the subsequent analyses yielded valuable insights into public opinion on "The Queen's Gambit".

## Future Work

Exploration of Multinomial Hidden Markov Model for sentiment analysis.

Comparison of performance of 'simpler' models, such as Naive Bayes Classifier, etc.

Deployment of the model as an API for real-time sentiment analysis.

## Acknowledgements

* BeCode Data Science and AI Bootcamp

* Hugging Face for the TinyBERT model

* Reddit for data availability

<p>&nbsp;</p>

# Annex: Model Overview

In this project, various models were explored to perform sentiment analysis, each with distinct architectures and methodologies. Below is an overview of the models considered:

## BERT (Bidirectional Encoder Representations from Transformers)

**Description:** BERT is a transformer-based model developed by Google that has significantly advanced natural language processing tasks. Unlike traditional models that read text sequentially, BERT processes text bidirectionally, capturing context from both preceding and following words simultaneously.

**How It Works:** BERT employs a method called Masked Language Modeling (MLM), where it randomly masks some tokens in a sentence and trains the model to predict the missing words. This approach enables BERT to understand the context of a word based on its surroundings.

**Pros:**

* Captures deep contextual relationships between words.
* Excels in various NLP tasks due to its bidirectional context understanding.

**Cons:**

* Resource-intensive, requiring substantial computational power for training and inference.

## DistilBERT

**Description:** DistilBERT is a compressed version of BERT, designed to be lighter and faster while retaining most of BERT's performance capabilities. It achieves this through a process called knowledge distillation, where a smaller model (the student) learns to replicate the behavior of a larger model (the teacher).

**How It Works:** By training on the outputs of the larger BERT model, DistilBERT reduces the number of parameters and computational requirements, making it more efficient for deployment.

**Pros:**

* Approximately 40% smaller and 60% faster than BERT, with about 97% of BERT's language understanding capabilities.
* More suitable for environments with limited computational resources.

**Cons:**

* Slightly less accurate than the original BERT model.
  
## TinyBERT

**Description:** TinyBERT is an even more compact version of BERT, further reducing the model size and computational demands. It is particularly useful for applications where resources are highly constrained.

**How It Works:** Similar to DistilBERT, TinyBERT utilizes knowledge distillation but focuses on both the embedding layer and the attention layers, ensuring that the smaller model closely mimics the original BERT's internal workings.

**Pros:**

* Significantly smaller model size, leading to faster inference times.
* Maintains a high level of accuracy despite the reduced size.

**Cons:**

* May not capture as much contextual nuance as larger models.

## Comparison with Traditional Models

In the realm of sentiment analysis, models can be broadly categorized into traditional machine learning models and advanced transformer-based models.

### Traditional Machine Learning Models:

Traditional models, such as logistic regression, support vector machines (SVM), and decision trees, have been widely used for sentiment analysis tasks. These models often rely on manually crafted features and bag-of-words representations, which can capture basic patterns in text data. While they are computationally efficient and perform well on simpler tasks, they may struggle with capturing complex linguistic nuances and long-range dependencies in text.

### Transformer-Based Models:

Transformer-based models, like BERT, DistilBERT, and TinyBERT, have revolutionized natural language processing by introducing mechanisms to capture context and meaning more effectively. They utilize self-attention mechanisms to weigh the importance of different words in a sentence, allowing them to understand context in a bidirectional manner.

### Key Differences:

* **Contextual Understanding:** Transformers capture the context of words in a bidirectional manner, leading to a deeper understanding of language nuances. Traditional models often rely on word frequency and may not effectively capture context.

* **Performance:** Transformer models generally achieve higher accuracy in NLP tasks due to their sophisticated architectures. Traditional models, while faster and less resource-intensive, may not reach the same level of accuracy, especially in tasks involving complex language understanding.

* **Resource Efficiency:** Models like DistilBERT and TinyBERT are optimized for efficiency, providing a balance between performance and resource consumption. They are suitable alternatives when computational resources are limited, offering substantial speed improvements with minimal loss in accuracy compared to the original BERT model.

In summary, while traditional models offer simplicity and efficiency, transformer-based models provide a more nuanced understanding of language, leading to superior performance in sentiment analysis tasks.


