# Stock Price Prediction Model
Developed a machine learning model that analyzes historical market data and trends to predict future stock prices and support data-driven investment decisions.

## Business Problem
Investors and firms need reliable tools to forecast stock price movements from historical data to reduce risk and make informed trading decisions.

## Data Preprocessing, Handling and Analysis
#### 1. Collect Data – Gather historical stock prices using yfinance.

#### 2. Preprocess Data – Clean, normalize, and handle missing values.

#### 3. Feature Engineering – Create features like moving averages, volume, etc.

#### 4. Split Data – Divide into training and testing sets.

#### 5. Build Model – Choose an ML model (e.g., LSTM).

#### 6. Train Model – Fit the model on training data.

#### 7. Evaluate Model – Check performance on test data using metrics.

#### 8. Make Predictions – Forecast future stock prices.

#### 9. Visualize Results – Plot actual vs predicted prices.

## Data Visualization
#### Stock Data - GOOG
![image alt](https://github.com/SundeepChaluvadi/Stock-Price-Prediction-Model/blob/db0abd0b4a44b2bc1446836c469a2eac05831ba8/Images/Stock%20Data.png)

#### 100-day moving average of the closing price of the stock
![image alt](https://github.com/SundeepChaluvadi/Stock-Price-Prediction-Model/blob/db0abd0b4a44b2bc1446836c469a2eac05831ba8/Images/100_Days.png)

#### 200-day moving average of the closing price of the stock
![image alt](https://github.com/SundeepChaluvadi/Stock-Price-Prediction-Model/blob/417be2c79a4e28abcce49291bdbcb5bd1e54474f/Images/200_Days.png)


## Dependencies
```bash
  pip install -r requirements.txt
```

## Demo
https://stock-price-prediction-model-123.streamlit.app/

## Installation
Clone the repository:

```bash
  git clone https://github.com/SundeepChaluvadi/Stock-Price-Prediction-Model.git
  cd Stock-Price-Prediction-Model
```

## Model Evaluation
#### Model Summary
1. The model uses LSTM neural networks to predict stock prices by learning patterns from past data. <br>
2. It has four LSTM layers to capture trends over time, with small dropout layers to avoid overfitting. <br>
3. The final layer predicts the next stock price based on learned patterns. <br>
4. Overall, it’s designed to forecast future stock prices by understanding past price movements. <br>
5. The model has 536,285 parameters in total, of which 178,761 are trainable. <br>
6. There are no non-trainable parameters, and the optimizer uses 357,524 parameters. <br>
7. The total model size is approximately 2.05 MB, making it compact yet effective for stock price prediction. <br>

![image alt](https://github.com/SundeepChaluvadi/Stock-Price-Prediction-Model/blob/db0abd0b4a44b2bc1446836c469a2eac05831ba8/Images/ModelSummary.png)


#### Original Price V/S Predicted Price
![image alt](https://github.com/SundeepChaluvadi/Stock-Price-Prediction-Model/blob/db0abd0b4a44b2bc1446836c469a2eac05831ba8/Images/Prediction_VS_Actual.png)


## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)]([https://www.linkedin.com/](https://www.linkedin.com/in/sundeep-chaluvadi))

## Sources
Data for this project is sourced using the yfinance library, which retrieves historical stock price data directly from Yahoo Finance.

```python
import yfinance as yf
