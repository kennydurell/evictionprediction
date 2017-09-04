# Predicting Total Eviction Notices by ZIP in San Francisco

The goal of this project was to use time-lagged features from disparate datasets to improve planning and prediction of eviction notice spikes in San Francisco ZIP codes. As of August 2017, San Francisco is becoming an increasingly expensive city to rent and live in and there is a definite sense that the city's desirability and limited housing stock is leading to increases in evictions. The city keeps a public record of all eviction notices given and this was used as the core dataset for prediction.

There are a number of hypothesis about what leads to increases in evictions and I wanted to explore possible predictors, from median home sale price in the prior year to demographic shifts based on American Community Survey data.

[Eviction Graph Here]

Ideally, this will be a resource for tenant rights' groups and city officials in terms of both planning and outreach.

The final result of this project is code in the forecasting folder, which allows you to input the number of months into the future you'd like to see and return predicted eviction totals, by ZIP, for those months.


# Data Collection and Storage
The data used in this model comes from a variety of sources. All are open source and publicly available. Links are provided below.

Data sources used:
Evictions - OpenDataSF
Capital Improvements Petitions- OpenDataSF
American Community Survey
Unemployment by Month - San Francisco

Most of the data munging/processing is done with the code in the data_processing folder. If there is additional pre-processing needed, I've listed this in the doc string.

# Results
I combined a top-down hierarchical ARIMAX model with a random forest regressor to achieve the lowest root mean squared error on unseen eviction data. Predicting outliers/spikes is extremely difficult, but I was able to improve upon the baseline in that regard and minimize their latent impact on future predictions in the process.


# Ongoing Work
Currently, the model only works with whatever data you've provided; the next step is to pull regularly from the OpenDataSF and the Department of Labor APIs to update the model with new data and improve its ability to forecast.

Likewise, proximity to ZIPs that had a large number of evictions in previous months may be in an indicator of an increases in evictions. This, along with feature engineering around changes in rental price over time as well as digging deeper into the large building permit dataset will be my next steps.



The code in this repo  and plot upcoming evictions.  




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
