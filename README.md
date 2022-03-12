# Time_Series_Analysis-Forecasting

![image](https://user-images.githubusercontent.com/88995459/158030296-b011ed75-7158-4711-9f83-eb20a6abec9a.png)


Time series forecasting is the task of predicting future values based on historical data. Examples across industries include forecasting of weather, sales numbers and stock prices. More recently, it has been applied to predicting price trends for cryptocurrencies such as Bitcoin and Ethereum. Given the prevalence of time series forecasting applications in many different fields, every data scientist should have some knowledge of the available methods for carrying it out. 

A wide array of methods are available for time series forecasting. One of the most commonly used is Autoregressive Moving Average (ARMA), which is a statistical model that predicts future values using past values. This method is flawed, however, because it doesn’t capture seasonal trends. It also assumes that the time series data is stationary, meaning that its statistical properties wouldn’t change over time. This type of behavior is an idealized assumption that doesn’t hold in practice, however, which means ARMA may provide skewed results. 

An extension of ARMA is the Autoregressive Integrated Moving Average (ARIMA) model, which doesn’t assume stationarity but does still assume that the data exhibits little to no seasonality. Fortunately, the seasonal ARIMA (SARIMA) variant is a statistical model that can work with non-stationary data and capture some seasonality. 

Python provides many easy-to-use libraries and tools for performing time series forecasting. Specifically, the stats library in Python has tools for building ARMA, ARIMA and SARIMA models with just a few lines of code. Since all of these models are available in a single library, you can easily run many experiments using different models in the same script or notebook. 

## Basic Stats Model for time series forecasting

#### Exponentially weighted moving average  - it will put more weight on the values that occured recently than the past  - smoothing factor (α)

#### Holt's Method (Double Exponenetial Smoothing) - it has a smoothing factor (β) that addresses the trend along with smoothing factor (α)

#### Holt Winter's Method (Triple Exponential Smoothing) -  it has as parameter (γ) that addresses the seasonality along with  smoothing factor (α) and (β)



## STATIONARITY

This is a very important concept in Time Series Analysis. In order to apply a time series model, it is important for the Time series to be stationary; in other words all its statistical properties (mean,variance) remain constant over time. This is done basically because if you take a certain behavior over time, it is important that this behavior is same in the future in order for us to forecast the series. There are a lot of statistical theories to explore stationary series than non-stationary series. (Thus we can bring the fight to our home ground!)


![image](https://user-images.githubusercontent.com/88995459/158029633-c07347e2-c7ef-4204-96fe-eba185c75734.png)


In practice we can assume the series to be stationary if it has constant statistical properties over time and these properties can be:

• constant mean

• constant variance

• an auto co-variance that does not depend on time.

######  Dickey-fuller Test :This is one of the statistical tests for checking stationarity. First we consider the null hypothesis: the time series is non- stationary. The result from the rest will contain the test statistic and critical value for different confidence levels. The idea is to have Test statistics less than critical value, in this case we can reject the null hypothesis and say that this Time series is indeed stationary

## MAKING THE TIME SERIES STATIONARY

There are two major factors that make a time series non-stationary. They are:

• Trend: non-constant mean

• Seasonality: Variation at specific time-frames

The basic idea is to model the trend and seasonality in this series, so we can remove it and make the series stationary. Then we can go ahead and apply statistical forecasting to the stationary series. And finally we can convert the forecasted values into original by applying the trend and seasonality constrains back to those that we previously separated.
  

### Trend

There are some methods to model these trends and then remove them from the series. Some of the common ones are:

• Smoothing: using rolling/moving average

• Aggression: by taking the mean for a certain time period (year/month)


### Seasonality (along with Trend)

Previously we saw just trend part of the time series, now we will see both trend and seasonality. Most Time series have trends along with seasonality. There are two common methods to remove trend and seasonality, they are:

• Differencing: by taking difference using time lag

• Decomposition: model both trend and seasonality, then remove them




### Auto Regressive Integrated Moving Average(ARIMA)

An ARIMA model is a class of statistical models for analyzing and forecasting time series data.

An ARIMA model comes along with both the flavours of AR and MA model where the time series is differenced at least once to make it stationary.

It explicitly caters to a suite of standard structures in time series data, and as such provides a simple yet powerful method for making skillful time series forecasts.

ARIMA is an acronym that stands for AutoRegressive Integrated Moving Average. It is a generalization of the simpler AutoRegressive Moving Average and adds the notion of integration.

This acronym is descriptive, capturing the key aspects of the model itself. Briefly, they are:

AR: Autoregression. A model that uses the dependent relationship between an observation and some number of lagged observations.

I: Integrated. The use of differencing of raw observations (e.g. subtracting an observation from an observation at the previous time step) in order to make the time series stationary.

MA: Moving Average. A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.

Each of these components are explicitly specified in the model as a parameter. A standard notation is used of ARIMA(p,d,q) where the parameters are substituted with integer values to quickly indicate the specific ARIMA model being used.

The parameters of the ARIMA model are defined as follows:
It is like a liner regression equation where the predictors depend on parameters (p,d,q) of the ARIMA model .

Let me explain these dependent parameters:

• p : This is the number of AR (Auto-Regressive) terms . Example — if p is 3 the predictor for y(t) will be y(t-1),y(t-2),y(t-3).
p: The first time where the PACF crosses the upper confidence interval

• q : This is the number of MA (Moving-Average) terms . Example — if p is 3 the predictor for y(t) will be y(t-1),y(t-2),y(t-3).
q: The first time where the ACF crosses the upper confidence interval

• d :This is the number of differences or the number of non-seasonal differences .

### SARIMAX Model
SARIMAX model is built by extending the ARIMA model, discussed in our previous section. In addition to terms AR, I and MA terms, there are four seasonal elements that are not part of ARIMA that must be configured; they are:

P: Seasonal autoregressive order.

D: Seasonal difference order.

Q: Seasonal moving average order.

m: The number of time steps for a single seasonal period.


## The following diagram illustrates the mechanism of using AR/MA/ARIMA/SARIMA by self choosing the model orders (p, q, d) and self tuning it.

![image](https://user-images.githubusercontent.com/88995459/158029700-72da4533-c587-4f63-8895-0f97ba54ebd2.png)


### AUTO ARIMA MODELS
This library automatically discovers the optimal order for an ARIMA model with stepwise execution of hyperparameters and parallel fitting of models.

This objective of this library (auto_arima) is to identify the most optimal parameters for an ARIMA/SARIMA and return a fitted ARIMA model. It does not depend on the PACF/Auto-Correlation (manual computation of differencing), but instead, it conducts differencing tests (i.e., Kwiatkowski–Phillips–Schmidt–Shin, Augmented Dickey-Fuller or Phillips–Perron) on its own to determine the order of differencing.

One other difference from Stats Model’s Vanilla ARIMA is that in case the model does not converge, it raises ValueError to signify the re-evaluation of stationarity-inducing measures.

For seasonality, SARIMA identifies the optimal P and Q hyperparameters after conducting the Canova-Hansen to determine the optimal order of seasonal differencing D, so that the auto-regressive/moving average portions of the seasonal model are defined by start_P, start_Q, max_P, max_Q

The model is fitted within ranges of defined start_p order (or number of time lags for AR), max_p, start_q (order of the moving average MA), max_q ranges.

Hence it can be formulated as : (p, q, d)x(P, Q, D)

### AUTO SARIMA MODEL
The SARIMA model is tuned with hyper-parameters to find the best model with the lowest AIC. The code snippet is given below, which does parameter tuning for (p, q, d)x(P, Q, D)

Here (P, Q, D) and (m) are tuned with values (1,0,1), (1,0,0), (0,0,1) and 12 respectively.

In the equation of SARIMA(p,d,q)x(P,D,Q):

P denotes Seasonal AR, D denote Seasonal Order of seasonal differencing and Q denotes Seasonal Moving Average, x is the frequency of the time series.

















