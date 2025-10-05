# Sentiment Analysis Project

A sentiment analysis system comparing traditional machine learning and deep learning approaches on Twitter data. The project classifies tweets into four sentiment categories: Positive, Negative, Neutral, and Irrelevant, using Logistic Regression, RNN, LSTM, and GRU models.

## Dataset

- **Training Set**: 74,682 tweets
- **Validation Set**: 1,000 tweets
- **Classes**: 4 sentiment categories (Negative, Positive, Neutral, Irrelevant)

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Project

1. **Clone the repository**
   ```bash
   git clone <https://github.com/Deolinda1506/Sentiment-Analysis.git>
   cd Sentiment-Analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks**
   - `Notebooks/Sentiment_Classification_with_Logistic_Regression.ipynb` - Traditional ML
   - `Notebooks/deep_learning_sentiment_final.ipynb` - Deep learning models

## Project Structure

```
Sentiment-Analysis/
├── Datasets/
│   ├── twitter_training.csv
│   └── twitter_validation.csv
├── Notebooks/
│   ├── Sentiment_Classification_with_Logistic_Regression.ipynb
│   ├── deep_learning_sentiment_final.ipynb
│   └── plots-final/
│       ├── results_table.csv
│       └── [visualization files]
├── requirements.txt
├── README.md

```

## Models Implemented

### Traditional Machine Learning
- **Logistic Regression**: TF-IDF and Count Vectorization

### Deep Learning Models
- **RNN**: SimpleRNN with embeddings
- **LSTM**: Bidirectional LSTM
- **GRU**: Bidirectional GRU
- **TF-IDF Dense**: Dense neural network with TF-IDF features

## Results Summary

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| **TF-IDF Dense** | **93.4%** | **93.4%** |
| **LSTM** | **90.2%** | **90.1%** |
| **RNN** | **87.4%** | **87.3%** |
| **GRU** | **84.4%** | **84.0%** |
| **Logistic Regression** | **76.8%** | **76.7%** |

## Key Findings

1. **Deep Learning Superiority**: All deep learning models significantly outperformed traditional ML
2. **TF-IDF Dense Winner**: Achieved highest performance (93.4% accuracy)
3. **LSTM Effectiveness**: Strong performance (90.2%) validating recurrent architectures

## Experiment Tables

### Logistic Regression Hyperparameter Experiments

| Feature Engineering | Max Features | C Value | Solver | Accuracy | F1-Score |
|-------------------|--------------|---------|--------|----------|----------|
| TF-IDF | 5000 | 0.1 | liblinear | 59.0% | 56.6% |
| TF-IDF | 5000 | 0.1 | lbfgs | 60.5% | 58.6% |
| TF-IDF | 5000 | 1.0 | liblinear | 66.6% | 65.9% |
| TF-IDF | 5000 | 1.0 | lbfgs | 67.3% | 67.0% |
| TF-IDF | 5000 | 10.0 | liblinear | 69.4% | 69.1% |
| TF-IDF | 5000 | 10.0 | lbfgs | 70.2% | 70.0% |
| Count | 10000 | 0.1 | liblinear | 67.1% | 66.3% |
| Count | 10000 | 0.1 | lbfgs | 68.7% | 68.1% |
| Count | 10000 | 1.0 | liblinear | 74.1% | 73.8% |
| Count | 10000 | 1.0 | lbfgs | 74.5% | 74.3% |
| **Count** | **10000** | **10.0** | **liblinear** | **76.8%** | **76.7%** |
| Count | 10000 | 10.0 | lbfgs | 76.5% | 76.5% |

### Deep Learning Model Experiments

| Model Type | Learning Rate | Batch Size | Optimizer | Epochs | Accuracy | F1-Score |
|------------|---------------|------------|-----------|-------|----------|----------|
| RNN | 0.001 | 64 | Adam | 5 | 87.4% | 87.3% |
| LSTM | 0.001 | 64 | Adam | 5 | 90.2% | 90.1% |
| GRU | 0.001 | 64 | RMSprop | 5 | 84.4% | 84.0% |
| **TF-IDF Dense** | **0.001** | **64** | **Adam** | **5** | **93.4%** | **93.4%** |
