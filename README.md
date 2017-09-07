# Predicting Total Eviction Notices by ZIP in San Francisco

The goal of this project was to use time-lagged features from disparate datasets to predict eviction notice spikes in San Francisco ZIP codes.

- As of the release of this project, San Francisco is becoming an increasingly expensive city to live in. There is a definite sense that the city's desirability and limited housing stock is leading to increases in evictions. The city keeps a public record of all eviction notices given and this was used as the core dataset for prediction.

- There are a number of hypothesis about what leads to increases in evictions and I wanted to explore possible predictors, from median home sale price in the prior year to demographic shifts based on American Community Survey data.



Ideally, this will be a resource for tenant rights' groups and city officials in terms of both planning and outreach.


# Results
I combined a top-down hierarchical ARIMAX model with a random forest regressor to achieve the lowest root mean squared error on unseen eviction data. Predicting outliers/spikes is extremely difficult, but I was able to improve upon the baseline in that regard and minimize their latent impact on future predictions in the process.

The final result of this project is code in the forecasting folder, which allows you to input the number of months into the future you'd like to see and return predicted eviction totals, by ZIP, for those months.

## My Process
I started with a baseline of predicting a rolling mean for all evictions in SF. This did surprisingly well, save for a few major outliers, so I began to look at potential ways to harness the power of this macro success to predict well on a ZIP by ZIP basis, where totals by month are small and erratic.


# Data Collection and Storage
The data used in this model comes from a variety of sources. All are open source and publicly available. Links are provided below.

Data sources used:
- Evictions - OpenDataSF [link]
- Capital Improvements Petitions- OpenDataSF
- American Community Survey
- Unemployment by Month - San Francisco

Most of the data munging/processing is done with the code in the data_processing folder. If there is additional pre-processing needed, I've listed this in the doc string.



# Using the Model
Here are the steps to run the model:

1. Follow these links to download the appropriate datasets:

- Eviction Notice OpenDataSF -
- Unemployment_Rate - https://fred.stlouisfed.org/series/CASANF0URN
- Capital Improvement -

Note : The open data version of the eviction notices

2. Import predict_evictions from linear_regression_forecast.py. Pass as parameters the datasets above as well as the months ahead you want to look and whether you want a set of plots returned by ZIP. It may take 10-15 minutes to run, as it has not yet been pickled (next on my to-do list).




# Ongoing Work
Currently, the model only works with whatever data you've provided; the next step is to pull regularly from the OpenDataSF and the Department of Labor APIs to update the model with new data and improve its ability to forecast.

I also have a few unexplored hypotheses. One is that a ZIP's proximity to ZIPs that had a large number of evictions in previous months may be an indicator of an increase in evictions in coming months in that ZIP. This, along with feature engineering around changes in rental price over time as well as digging deeper into the City of San Francisco's large building permit dataset will be my next steps.



An overview of your project.
What is the goal of your project?
How did you accomplish this goal? (Include an explanation that's not too technical)
What are your results?
How can I see what you did? (Link to your live app!)
An in-depth explanation of your process
What algorithms and techniques did you use?
How did you validate your results?
What interesting insights did you gain?
How to run on my own
Give instructions for how to run your code on their computer (e.g. Run python scraper.py to collect the data, then run...)
