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



