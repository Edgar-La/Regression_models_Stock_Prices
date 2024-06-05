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
 
## Process data
### Make chunks of time-data

In order to find paterns in the data, we give the data some special estructure that we call chunks i.e.time windows:
* take the data and makes chunk, the chunk shape is a window of (7,5)
* 7 are the days and 5 are the attributes (open, high, low, close, volume)
* for wach chunk, we add a target that will be the _Close_ value of the 8th day
* the chunks will have an offset of 1 day i.e. the window will slide one day to make the next time-window and the next target value

### Reshape data
In order to employ classic regression models, we need to give the data a valid shape to be compatible with the model.
Thats why we flateen our data. Previously oru data has a shape of (7,5), now will be (35,)

### Normalize data
We implement a normalization using standard-deviation like:
$$z = \frac{X - \mu}{\sigma}$$


## Regressor models
### Random Forest Regressor
It's an ensemble learning method that operates by constructing a multitude of decision trees during training time and outputting the mean prediction of the individual trees for regression problems. It's known for its robustness against overfitting and high performance.
### Gradient Boosting Regressor
It's another ensemble method that builds trees one at a time, where each new tree helps to correct errors made by the previously trained set of trees. Gradient boosting tends to be more sensitive to overfitting than Random Forests but can often yield better performance if tuned correctly.
### Extra Trees Regressor
This is similar to Random Forests, but with one key difference: instead of searching for the best split point in the feature when building trees, it selects a random split point. This randomization can lead to faster training times and sometimes better performance, especially for high-dimensional data.

## Train and predict
The previously regressor models where implemented directly from the scikit-learn modules.
* the trainning was implemented with the `fit()` method on the train split data
* and the prediction with the `predict()` method with the test split data

## Pathway
The following image shows the steps summarized in this project, notice that the last plots shows the results of prediction values for _Close_ for each regressor model.
<p align="center">
  <img src="https://github.com/Edgar-La/Regression_models_Stock_Prices/blob/main/images/regression_outputs_comparison.png" alt="pathway" width="1000"/> </a>
 </p>

## Actual vs Predicted
In this image, we can see a comparison for the actual values and th predicted ones for each regressor.
<p align="center">
  <img src="https://github.com/Edgar-La/Regression_models_Stock_Prices/blob/main/images/actual_vs_predicted.png" alt="actual-predicted" width="1000"/> </a>
 </p>
 
## Residual analysis
In this image, we can see a comparison for the residual analysis for each regressor.
<p align="center">
  <img src="https://github.com/Edgar-La/Regression_models_Stock_Prices/blob/main/images/residual_analysis.png" alt="residual" width="1000"/> </a>
 </p>
 
## Metrics chart
In this plot, we can observe a summarization of the metrics calculated for each regressors model.
<p align="center">
  <img src="https://github.com/Edgar-La/Regression_models_Stock_Prices/blob/main/images/metrics.png" alt="metrics-plot" width="1000"/> </a>
 </p>


 ## Results: Regressor quality
As we can observe, the regressor models have a very similar behaviour in all the seven metrics calculated. Nevertheless, the Random Forest Regressor show an slightly improvement in the metrics.
| Regressors            |   MSE   |  RMSE  |   MAE   |   RSE   |   RAE   |    R    |    R2    |
|-----------------------|---------|--------|---------|---------|---------|---------|---------|
| RandomForestReg       |  9.4459 | 3.0734 |  1.9380 |  3.1298 |  0.3270 |  0.8927 |  0.7737 |
| GradientBoostingReg   |  9.9440 | 3.1534 |  2.0396 |  3.2113 |  0.3441 |  0.8845 |  0.7617 |
| ExtraTreesReg         | 10.3015 | 3.2096 |  1.9351 |  3.2685 |  0.3265 |  0.8804 |  0.7532 |



# Conclussions
Regression analysis is a powerful tool for understanding relationships between variables and making predictions. By carefully considering the assumptions and properly interpreting the results, regression models can provide valuable insights in many fields, from economics and finance to biology and engineering.

In this project, we explored various metrics to evaluate the performance of regression models. The metrics we focused on included Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Relative Squared Error (RSE), Relative Absolute Error (RAE), Mean Absolute Percentage Error (MAPE), the Correlation Coefficient (R), and the Coefficient of Determination (R2).

Each metric provides unique insights into the performance of regression models:
* MSE and RMSE are useful for understanding the average magnitude of errors, with RMSE being particularly interpretable due to its same units as the dependent variable.
* MAE offers a robust measure less sensitive to outliers compared to MSE and RMSE.
* RSE and RAE provide relative measures comparing model performance to a baseline model, with values closer to 0 indicating superior performance.
* R helps assess the strength and direction of the linear relationship between actual and predicted values, with values closer to 1 or -1 indicating stronger linear relationships.
* R2 indicates the proportion of variance in the dependent variable explained by the independent variables, with higher values signifying better model fit.

The comprehensive evaluation using these metrics allows an understanding of model performance. For example, while a model might exhibit low MSE and RMSE, indicating small average errors, it could still have high RSE or RAE values if the baseline model performs similarly well. Similarly, a high R2 value signifies that a significant portion of the variance is explained by the model, but it doesn’t provide information about the actual size of prediction errors, which metrics like MAE and RMSE do.

In conclusion, employing a diverse set of evaluation metrics provides a holistic view of the regression model’s effectiveness, enabling more informed decisions in model selection and refinement. The combination of these metrics ensures that the models not only fit the data well but also generalize effectively to new data, ultimately leading to more accurate and reliable predictions in practical applications.
