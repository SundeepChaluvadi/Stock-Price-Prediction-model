# Library Imports and Setup

# Use non-GUI backend for Matplotlib to avoid warnings in Flask
import matplotlib
matplotlib.use('Agg')

# Numerical and data handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

# Machine Learning / Deep Learning
from keras.models import load_model

# Web app framework
from flask import Flask, render_template, request, send_file

# Date and time
import datetime as dt

# Finance data
import yfinance as yf

# Data preprocessing
from sklearn.preprocessing import MinMaxScaler

# System operations
import os

app = Flask(__name__)

# Load the model
model = load_model('Stock_Price_Model.keras')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock') or 'GOOG'
        
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime.now()
        
        df = yf.download(stock, start=start, end=end, auto_adjust=True)
        
        data_desc = df.describe()
        
        # Simple Moving Averages
        df['MA_20'] = df['Close'].rolling(20).mean()
        df['MA_50'] = df['Close'].rolling(50).mean()
        df['MA_100'] = df['Close'].rolling(100).mean()
        df['MA_200'] = df['Close'].rolling(200).mean()
        df['MA_250'] = df['Close'].rolling(250).mean()
        
        # Data splitting
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)
        
        past_250_days = data_training.tail(250)
        final_df = pd.concat([past_250_days, data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)
        
        x_test, y_test = [], []
        for i in range(250, input_data.shape[0]):
            x_test.append(input_data[i-250:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        y_predicted = model.predict(x_test)
        y_predicted = scaler.inverse_transform(y_predicted.reshape(-1,1))
        y_test = scaler.inverse_transform(y_test.reshape(-1,1))
        
        # Plotting
        plots = {
            "sma_250": ("Closing Price vs 250-Day SMA", "MA_250", "b"),
            "sma_200": ("Closing Price vs 200-Day SMA", "MA_200", "g"),
            "sma_100": ("Closing Price vs 100-Day SMA", "MA_100", "r"),
            "sma_250_100": ("Closing Price vs 250-Day & 100-Day SMA", ["MA_250","MA_100"], ["b","r"])
        }

        for filename, (title, cols, colors) in plots.items():
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['Close'], 'y', label='Closing Price')
            if isinstance(cols, list):
                for c, color in zip(cols, colors):
                    ax.plot(df[c], color, label=c)
            else:
                ax.plot(df[cols], colors, label=cols)
            ax.set_title(title)
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            ax.legend()
            fig.savefig(f"static/{filename}.png")
            plt.close(fig)
        
        # Prediction plot
        fig_pred, ax_pred = plt.subplots(figsize=(12,6))
        ax_pred.plot(y_test, 'g', label="Original Price", linewidth=1)
        ax_pred.plot(y_predicted, 'r', label="Predicted Price", linewidth=1)
        ax_pred.set_title("Prediction vs Original Trend")
        ax_pred.set_xlabel("Time")
        ax_pred.set_ylabel("Price")
        ax_pred.legend()
        prediction_chart_path = "static/stock_prediction.png"
        fig_pred.savefig(prediction_chart_path)
        plt.close(fig_pred)

        csv_file_path = f"static/{stock}_dataset.csv"
        df.to_csv(csv_file_path)

        return render_template('index.html',
                                plot_path_sma_250="static/sma_250.png",
                                plot_path_sma_200="static/sma_200.png",
                                plot_path_sma_100="static/sma_100.png",
                                plot_path_sma_250_100="static/sma_250_100.png",
                                plot_path_prediction=prediction_chart_path,
                                data_desc=data_desc.to_html(classes='table table-bordered'),
                                dataset_link=csv_file_path)
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
