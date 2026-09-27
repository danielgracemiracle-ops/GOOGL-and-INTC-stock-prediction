# 📈 GOOGL & INTC Stock Price Prediction

A **Deep Learning time-series forecasting project** that uses **Long Short-Term Memory (LSTM)** networks to predict stock closing prices for **Alphabet Inc. (GOOGL)** and **Intel Corporation (INTC)**.

The project compares a **baseline LSTM model** with an **enhanced LSTM model** to investigate how architectural and training improvements affect stock price forecasting performance.

---

## 📌 Project Overview

Stock prices are sequential time-series data where previous observations can provide information about future observations.

This project explores the use of **LSTM neural networks**, a recurrent neural network architecture designed to learn patterns and dependencies in sequential data.

Two stocks were selected for the experiment:

* **GOOGL** — Alphabet Inc.
* **INTC** — Intel Corporation

Each dataset is trained using two different LSTM configurations:

1. **Baseline LSTM**
2. **Enhanced LSTM**

The models are then evaluated using multiple regression metrics.

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze historical stock price time-series data.
2. Prepare sequential data for deep learning.
3. Build an LSTM baseline model.
4. Develop an enhanced LSTM model.
5. Compare model performance on GOOGL and INTC.
6. Evaluate predictions using RMSE, MAE, and MAPE.
7. Analyze the effect of model improvements on forecasting performance.

---

# 📊 Datasets

## GOOGL

The GOOGL dataset contains historical daily stock market data.

**Dataset characteristics:**

* Rows: **3,932**
* Features: **7**
* Period: **August 19, 2004 – April 1, 2020**
* Growth over the dataset period: **2,094.53%**

The dataset contains historical market information with the **Close** price used as the primary prediction target.

### Train-Test Split

The testing period begins from:

```text
April 1, 2019
```

---

## INTC

The INTC dataset contains a longer historical period of Intel stock data.

**Dataset characteristics:**

* Rows: **10,098**
* Features: **7**
* Period: **March 17, 1980 – April 1, 2020**
* Growth over the dataset period: **15,837.54%**

The **Close** price is used as the target variable for forecasting.

### Train-Test Split

The testing period begins from:

```text
April 1, 2019
```

---

# ⚙️ Time-Series Preparation

Before training the LSTM models, the stock data is transformed into sequential samples.

### Window Size

```text
5 previous trading days
```

### Forecast Horizon

```text
1 day ahead
```

Conceptually:

```text
Day 1 ─┐
Day 2  │
Day 3  │ → LSTM → Day 6 Closing Price
Day 4  │
Day 5 ─┘
```

The sliding-window approach converts the historical time-series data into sequences that can be processed by the LSTM network.

---

# 🧠 LSTM Models

Two models were developed for each stock.

## 1. Baseline LSTM

The baseline model provides a reference point for evaluating the effect of subsequent improvements.

The model learns temporal relationships from historical closing-price sequences and predicts the closing price for the following trading day.

---

## 2. Enhanced LSTM

The enhanced model introduces improvements to the baseline architecture/training configuration to improve the model's ability to learn temporal patterns and generalize to unseen data.

The enhanced model is evaluated using exactly the same testing methodology as the baseline model.

This allows the performance difference between the two approaches to be measured objectively.

---

# 📈 Model Performance

## GOOGL Results

| Model         |        RMSE |         MAE |        MAPE |
| ------------- | ----------: | ----------: | ----------: |
| Baseline LSTM |     28.8649 |     19.4574 |     1.5653% |
| Enhanced LSTM | **25.9005** | **16.9470** | **1.3781%** |

The enhanced GOOGL model achieved lower RMSE, MAE, and MAPE compared with the baseline model.

---

## INTC Results

| Model         |       RMSE |        MAE |        MAPE |
| ------------- | ---------: | ---------: | ----------: |
| Baseline LSTM |     1.5540 |     0.9430 |     1.7912% |
| Enhanced LSTM | **1.4360** | **0.9068** | **1.7222%** |

The enhanced INTC model also produced lower error values across the reported metrics.

---

# 📊 Evaluation Metrics

The models were evaluated using three regression metrics.

### RMSE — Root Mean Squared Error

Measures the square root of the average squared prediction error.

Lower RMSE indicates smaller prediction errors.

### MAE — Mean Absolute Error

Measures the average absolute difference between predicted and actual prices.

Lower MAE indicates predictions that are closer to the actual values.

### MAPE — Mean Absolute Percentage Error

Measures prediction error as a percentage of the actual value.

Lower MAPE indicates lower relative prediction error.

---

# 🔬 Model Comparison

The experiment compares the models based on:

* RMSE
* MAE
* MAPE
* Validation loss
* Prediction behavior
* Generalization on unseen test data

The results show that the enhanced configuration reduced forecasting errors for both datasets.

---

# 🏗️ Machine Learning Pipeline

```text
┌─────────────────────────────┐
│ Historical Stock Data       │
│                             │
│ GOOGL / INTC                │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Data Preprocessing          │
│                             │
│ • Cleaning                  │
│ • Scaling                   │
│ • Feature Preparation       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Sliding Window              │
│                             │
│ Window = 5 days             │
│ Horizon = 1 day             │
└──────────────┬──────────────┘
               │
               ▼
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌─────────────┐  ┌─────────────┐
│ Baseline    │  │ Enhanced    │
│ LSTM        │  │ LSTM        │
└──────┬──────┘  └──────┬──────┘
       │                │
       └───────┬────────┘
               ▼
┌─────────────────────────────┐
│ Model Evaluation            │
│                             │
│ RMSE / MAE / MAPE           │
└─────────────────────────────┘
```

---

# 🧠 Why LSTM?

LSTM is a recurrent neural network architecture designed to process sequential data and retain information across time steps.

This makes it suitable for time-series problems where the ordering of observations matters.

For stock-price forecasting, the model can learn temporal patterns from previous observations rather than treating each data point as an independent sample. Similar LSTM-based stock forecasting projects commonly use sliding windows to convert historical prices into sequences for next-step prediction.

However, stock prices are affected by many factors outside historical price patterns, so forecasting performance should be interpreted within the limitations of the available data.

---

# 🛠️ Tech Stack

### Programming

* Python

### Deep Learning

* TensorFlow
* Keras
* LSTM

### Data Processing

* Pandas
* NumPy
* Scikit-learn

### Visualization

* Matplotlib
* Seaborn

### Development Environment

* Jupyter Notebook
* Google Colab

---

# 📂 Project Structure

```text
GOOGL-INTC-Stock-Prediction/
│
├── dataset/
│   ├── GOOGL.csv
│   └── INTC.csv
│
├── notebooks/
│   └── stock_prediction.ipynb
│
├── models/
│   ├── googl_baseline/
│   ├── googl_enhanced/
│   ├── intc_baseline/
│   └── intc_enhanced/
│
├── results/
│   ├── googl/
│   └── intc/
│
├── requirements.txt
└── README.md
```

---

# 📊 Visualizations

The project includes visualizations to analyze model behavior, including:

### Actual vs Predicted Price

Comparison between the actual closing price and the price predicted by the LSTM model.

### Training & Validation Loss

Visualization of model loss during training to analyze convergence and potential overfitting.

### Model Comparison

Comparison of baseline and enhanced models based on RMSE, MAE, and MAPE.

---

# 🔄 Baseline vs Enhanced Model

The project follows an experimental approach:

```text
Baseline Model
      ↓
Evaluate Performance
      ↓
Identify Improvement Opportunities
      ↓
Enhanced Model
      ↓
Evaluate Again
      ↓
Compare Results
```

This approach makes it possible to determine whether changes to the model configuration actually improve forecasting performance.

---

# 📚 Learning Outcomes

Through this project, I gained practical experience in:

* Deep Learning
* Time-Series Forecasting
* LSTM
* Sequential Data Preparation
* Sliding Window Technique
* Data Normalization
* Regression Evaluation
* RMSE
* MAE
* MAPE
* Model Comparison
* Training & Validation Analysis
* Financial Time-Series Analysis

The project also strengthened my understanding of how deep learning models can be applied to sequential financial data.

---

# 🚀 Future Improvements

Potential improvements include:

* Increasing the historical window size
* Adding Open, High, Low, and Volume features
* Incorporating technical indicators such as RSI and MACD
* Experimenting with GRU and Bidirectional LSTM
* Hyperparameter tuning
* Walk-forward validation
* Attention mechanisms
* Transformer-based time-series models
* Combining technical indicators with market sentiment

Using additional financial variables is a common direction for stock forecasting projects because relying only on historical closing prices limits the information available to the model.

---

# ⚠️ Disclaimer

This project is intended for **educational and portfolio purposes only**.

The predictions generated by the models should not be interpreted as financial advice or a guarantee of future stock prices.

Financial markets are influenced by many unpredictable factors, including economic conditions, company events, investor behavior, and market sentiment.

---

# 👨‍💻 Project Role

**Deep Learning / Data Science Developer**

Responsibilities included:

* Preparing historical stock datasets
* Performing time-series preprocessing
* Creating sliding-window sequences
* Developing LSTM models
* Training baseline and enhanced models
* Evaluating model performance
* Comparing model configurations
* Visualizing actual vs predicted prices
* Analyzing model results

---

## 📌 Key Results

### GOOGL

**Enhanced LSTM**

* RMSE: **25.9005**
* MAE: **16.9470**
* MAPE: **1.3781%**

### INTC

**Enhanced LSTM**

* RMSE: **1.4360**
* MAE: **0.9068**
* MAPE: **1.7222%**

Overall, the project demonstrates an end-to-end **Deep Learning time-series forecasting workflow**, from historical stock data preparation and sequence generation to LSTM training, evaluation, and comparison.
