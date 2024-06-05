# Stock Market Regression - Starbucks Corp <img src="https://cdn.worldvectorlogo.com/logos/starbucks-coffee.svg" alt="starbucks" width="50" style="fmargin:0 10px 10px 0"/> </a>



The goal of this notebook is to implement classic regression models and implement metrics from scratch to calculate the quality of the regressors, aslo show some interesting plots that tells the steps involved.
Join me on this funny journey :coffee:

## Tools
<p align="left"> <a href="https://www.python.org/" target="_blank"> <img src="https://cdn.worldvectorlogo.com/logos/python-5.svg" alt="Python" width="40" height="40"/> </a>
<a href="https://matplotlib.org/stable/" target="_blank"> <img src="https://cdn.worldvectorlogo.com/logos/matplotlib-1.svg" alt="Matplotlib" width="40" height="40"/> </a>
<a href="https://scikit-learn.org/stable/" target="_blank"> <img src="https://icon.icepanel.io/Technology/svg/scikit-learn.svg" alt="scikit-learn" width="40" height="40"/> </a>
<a href="https://numpy.org/" target="_blank"> <img src="https://cdn.worldvectorlogo.com/logos/numpy-1.svg" alt="Numpy" width="40" height="40"/> </a>
<a href="https://pandas.pydata.org/" target="_blank"> <img src="https://cdn.worldvectorlogo.com/logos/pandas.svg" alt="Pandas" width="40" height="40"/> </a>
<a href="https://www.jetbrains.com/pycharm/?var=1" target="_blank"> <img src="https://upload.wikimedia.org/wikipedia/commons/1/1d/PyCharm_Icon.svg" alt="PyCharm" width="40" height="40"/> </a>

## Dataset
In this project we use data from the SBUX stock market. The data goes from 2019-06-05 to 2024-06-05.
The data was extracted directly from Yahoo Finance using the API and the library [yfinance](https://pypi.org/project/yfinance/).

## Metrics
| **Metric** | **Formula** | **Interpretation** |
|------------|-------------|--------------------|
| MSE | $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$ | Lower values indicate a better fit |
| RMSE | $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$ | Lower values indicate a better fit, same units as \( y \) |
| MAE | $$MAE = \frac{1}{n} \sum_{i=1}^{n} \|y_i - \hat{y}_i\|$$ | Lower values indicate a better fit |
| RSE | $$RSE = \frac{\sum_{i=1}^{n} (y_i - \hat{y}\_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$ | Values closer to 0 indicate a better fit |
| RAE | $$RAE = \frac{\sum_{i=1}^{n} \|y_i - \hat{{y}\_i} \| }{\sum_{i=1}^{n} \|y_i - \bar{y}\|}$$ | Values closer to 0 indicate a better fit |
| R | $$R = \frac{\sum_{i=1}^{n} (y_i - \bar{y})(\hat{y}\_i - \bar{\hat{y}})}{\sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2 \sum_{i=1}^{n} (\hat{y}_i - \bar{\hat{y}})^2}}$$ | Values closer to 1 or -1 indicate a strong linear relationship |
| $$R^2$$ | $$R^2 = 1 - \frac{ \sum_{i=1}^{n} (y_{i} - \hat{y}\_{i})^2}{ \sum_{i=1}^{n} (y_i - \bar{y})^2}$$ | Values closer to 1 indicate a better fit |

## Dataset attributes
Using `import yfinance as yf` and `sbux_data = yf.download('SBUX', period='5y')` we have the following dataset.
Notice that the attributes are _Date, Open, High, Low, Close_ and _Volume_.
The dataset header is the following:

| __Date__       | __Open__      | __High__      | __Low__       | __Close__     | __Volume__  |
|------------|-----------|-----------|-----------|-----------|---------|
| 2019-06-05 | 78.790001 | 79.970001 | 78.660004 | 79.959999 | 7437100 |
| 2019-06-06 | 80.029999 | 81.629997 | 79.900002 | 81.400002 | 10457200 |
| 2019-06-07 | 81.599998 | 83.330002 | 81.510002 | 82.480003 | 11278800 |
| 2019-06-10 | 82.849998 | 82.860001 | 81.379997 | 81.930000 | 8102800 |
| 2019-06-11 | 82.300003 | 82.860001 | 81.849998 | 82.370003 | 6226400 |


## Plotting the historic attribute _Close_
Here, we take the _Close_ attribute from the entire historic and plot it to see the evoluting of the stock over time.

<p align="center">
  <img src="https://github.com/Edgar-La/Regression_models_Stock_Prices/blob/main/images/historical_data.png" alt="historical" width="1000"/> </a>
 </p>



## Make train and test splits
Becasuse we seek to implement regression over time-series data, we need to take sequences, therefore, to make train and test splits, we take a sequence for train split and another sequence for test split.
* The train split goes from Jan 01, 2022 to Mar 01, 2024.
* The test split goes from Mar 01, 2023 to Jun 03, 2024.
We can see this in the following image:

<p align="center">
  <img src="https://github.com/Edgar-La/Regression_models_Stock_Prices/blob/main/images/train_test_data.png" alt="train_test" width="1000"/> </a>
 </p>
 
