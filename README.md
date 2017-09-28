# Predicting Total Eviction Notices by ZIP in San Francisco

The goal of this project was to use time-lagged features from disparate datasets to predict eviction notice spikes in San Francisco ZIP codes.

- As of the release of this project, San Francisco is becoming an increasingly expensive city to live in. There is a definite sense that the city's desirability and limited housing stock is leading to increases in evictions. The city keeps a public record of all eviction notices given and this was used as the core dataset for prediction.

- There are a number of hypothesis about what leads to increases in evictions and I wanted to explore possible predictors, from median home sale price in the prior year to demographic shifts based on American Community Survey data.



Ideally, this will be a resource for tenant rights' groups and city officials in terms of both planning and outreach.


# Results
I combined a top-down hierarchical ARIMAX model with a random forest regressor to achieve the lowest root mean squared error on unseen eviction data by ZIP. Predicting outliers/spikes is extremely difficult, but I was able to improve upon the baseline in that regard and minimize their latent impact on future predictions in the process.

The features in each model that ultimately produced the lowest RMSE for the cross validated time series data were:

**ARIMAX Features**  
    Unemployment Rate - year prior (along with AR of 2, I of 1 and MA of 2)

**Random Forest Regressor Features**  
    Month,Year, Zipcode, Unemployment Rate - year_prior, Unemployment Rate - six_months_prior, Capital Improvement Petitions - year_prior, Capital Improvement Petitions - six_months_prior, Capital Improvement Petitions - two_years_prior, Black population - year_prior , Median Home Sale Price - year_prior, Median Home Sale Price - six_months_prior

**Feature Importance for Random Forest Regressor**
1. ZIP 94110
2. Unemployment Rate - year_prior
3. Month
4. Unemployment Rate - six_months_prior
5. Capital Improvement Petitions - two_years_prior


The final result of this project is code in the forecasting folder, which allows you to input the number of months into the future you'd like to see and return predicted eviction totals, by ZIP, for those months.

[Watch me present my process and the results here!](https://youtu.be/MZoeI4p_Hq8?t=4977)

# My Process
I started with a baseline of predicting a rolling mean for all evictions in SF. This did surprisingly well, save for a few major outliers, so I began to look at potential ways to harness the power of this macro success to predict well on a ZIP by ZIP basis, where totals by month are small and erratic.


# Data Collection and Storage
The data used in this model comes from a variety of sources. All are open source and publicly available. Links are provided below.

Data sources used:

- [Eviction Notices in SF](https://data.sfgov.org/Housing-and-Buildings/Eviction-Notices/5cei-gny5/data)
- [Petitions to the Rent Board](https://data.sfgov.org/Housing-and-Buildings/Petitions-to-the-Rent-Board/6swy-cmkq) (specifically the Capital Improvement Request petitions)
- [American Community Survey](https://factfinder.census.gov/faces/nav/jsf/pages/community_facts.xhtml?src=bkmk)
- [Unemployment by Month - San Francisco](https://fred.stlouisfed.org/series/CASANF0URN)

Most of the data munging/processing is done with the code in the data_processing folder. If there is additional pre-processing needed, I've listed this in the doc string.



# Using the Model
Here are the steps to run the model:

1. Follow the links below to download the appropriate datasets. The name of the datasets, their parameter names, and what they contain, are listed.

- [Eviction Notices in SF](https://data.sfgov.org/Housing-and-Buildings/Eviction-Notices/5cei-gny5/data) = df_eviction (* additional info below)

   Evictions notices filed with the San Francisco Rent Board from 1997 to present (usually up to 2-3 months prior to present day). An eviction notice does not necessarily mean the tenant was evicted.


- [Petitions to the Rent Board](https://data.sfgov.org/Housing-and-Buildings/Petitions-to-the-Rent-Board/6swy-cmkq) = df_capital_improvements

   Petitions filed with the Rent Board for a variety of reasons, although typically they are tied to a request for a rent increase/decrease review by the Rent Board. The only feature in the dataset used by this model is the capital improvement request column. These are requests made by landlords for substantial capital improvements on a building, typically resulting in evictions while the capital improvements are conducted.


- [American Community Survey](https://factfinder.census.gov/faces/nav/jsf/pages/community_facts.xhtml?src=bkmk) = df_census

   Population and a mountain of demographic information for each ZIP in San Francisco. The ACS is conducted on more regular intervals than the US Census so as to provide a more accurate understanding of the current demographic makeup of U.S. ZIPs/counties/cities/states.

- [Unemployment by Month - San Francisco](https://fred.stlouisfed.org/series/CASANF0URN) = df_unemployment

   Culled from the US Bureau of Labor Statistics, this dataset includes the unemployment rate in SF for each month from 1990 to present day lagged 2 months.


2. Import the predict_evictions function from linear_regression_forecast.py. Pass the datasets above as parameters as well as the months ahead you want to look and whether you want a set of plots returned by ZIP. It may take 10-15 minutes to run, as it has not yet been pickled (next on my to-do list).

3. The past evictions (predicted and actual) as well as the forecasted future evictions will be saved on your computer as 'past_evictions_by_zip_SF.csv' and 'future_evictions_by_zip_SF.csv'



   *Thanks to some additional follow-up with the OpenSF team, I was given special access to eviction notices by individual building addresses (as compared to the block-level addresses provided in the publicly available dataset) and used these addresses in my dataset, filling in any blank ZIPs via the Streety Smarts ZIP API. The columns I added are named 'Specific_Address' and 'Address_Zipcode'. The landlord ZIP matches up with the tenant ZIP about 90% of the time so if you're looking for a quick overview/forecast, using these ZIPs as 'Address_Zipcode' is probably the best option.

   Otherwise, the 'Location' column lists the lat-lon for the block level for each eviction notice and, with the help of [Geocoder](https://chrisalbon.com/python/geocoding_and_reverse_geocoding.html), this data can be used to back out the eviction notice ZIP for each entry.*


# Ongoing Work
Currently, the model only works with whatever data you've provided; the next step is to pull regularly from the OpenDataSF and the Department of Labor APIs to update the model with new data and improve its ability to forecast.

I also have a few unexplored hypotheses. One is that a ZIP's proximity to ZIPs that had a large number of evictions in previous months may be an indicator of an increase in evictions in coming months in that ZIP. This, along with feature engineering around changes in rental price over time as well as digging deeper into the City of San Francisco's large building permit dataset will be my next steps.
