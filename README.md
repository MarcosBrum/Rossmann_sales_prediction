# Rossmann_sales_prediction

Sales forecast for the stores of the european drugstore chain Rossmann.

<img src="https://github.com/MarcosBrum/Rossmann_sales_prediction/blob/master/rossmann_drogeriemarkt.jpg" height="378" width="504">


## 1 Business problem

The stores of the Rossmann drugstore chain need to be restored and the CEO needs to decide how much is going to be dedicated to the restoration of each one. To support this decision, the Analytics team is asked to present a sales forecast for each store during a period of six weeks, alongside with the total income expected in the chain. This forecast also informs the CEO which store is able to account for its own restoration with the income within this period.

## 2 Business results

The gross expected income of the majority of stores is in the range between R$5000.00 and R$22000.00. The chain is expected to obtain R$289,822,112.00, with best and worst case scenarios of R$290,808,412.17 and R$288,835,860.27, respectively. These scenarios are predicted using the mean absolute percentage error. The same statistical error is applied to each store, individually.


## 3 Business Assumptions

* All stores contain a basic sortment, but some of them contain (different kinds of) extra sortments.
* The store's opening on weekends and holidays vary from place to place.
* The stores participate in seasonal promotions. In some of these cases, the promotion is continued for a longer time.


## 4 Solution Strategy

The strategy adopted was the following:

__Step 01. Data Description__: I searched for NAs, checked data types (and adapted some of them for analysis) and presented a statistical description.

__Step 02. Feature Engineering__: New features were created to make possible a more thorough analysis.

__Step 03. Data Filtering__: Entries containing no information or containing information which does not match the scope of the project were filtered out.

__Step 04. Exploratory Data Analysis__: I performed univariate, bivariate and multivariate data analysis, obtaining statistical properties of each of them, correlations and testing hypothesis (the most important of them are detailed in the following section).

__Step 05. Data Preparation__: Numerical data was rescaled, categorical data was transformed and cyclic data (such as months, weeks and days) was transformed using mathematical trigonometrical functions.

__Step 06. Feature selection__: The statistically most relevant features were selected using the Boruta package.

__Step 07. Machine learning modelling__: Some machine learning models were trained. The one that presented best results after cross-validation went through a further stage of hyperparameter fine tunning to optimize the model's generalizability.

__Step 08. Model-to-business__: The models performance is converted into business values.

__Step 09. Deploy Model to Production__: The model is deployed on a cloud environment to make possible that other stakeholders and services access its results.


## 5 Top 3 Data insights

1. Stores with larger assortment do not sell more.
2. Stores with closer competitors do sell more.
3. Stores sell less at school holidays (except during summer).


## 6 Machine Learning Model Applied

The following machine learning models were trained:
* Linear Regression;
* Regularized Linear Regression;
* Random Forest Regressor;
* XGBoost Regressor.

All of them were cross-validated and their performance was compared against a random model.


## 7 Machine Learning Model Performance

The performance of every trained model, after cross-validation. The columns correspond to the metrics: Mean Absolute Error, Mean Absolute Percentage Error and Root Mean Squared Error.

![picture alt](https://github.com/MarcosBrum/Rossmann_sales_prediction/blob/master/cv_performance_to_readme.jpg)

## 8 Conclusions

The sales forecast and the generated insights provide the CEO with valuable tools to decide the amount of budget that is going to be dedicated to the restoration of each store.


## 9 Lessons Learned

* The exploratory data analysis provides important insights to the business problem, many of which contradict the initial hypothesis. This information is valuable for the understanding of business and for planning future actions. This step also provides a preview of the result of the feature selection step.
* The machine learning model performance must be evaluated in the learning and generalization stages. A balance between bias and variance must be achieved based on the uniqueness of the problem.


## 10 Next steps and improvements

Some hypothesis made when filling missing data would be reviewed in a following CRISP cycle, and other ones would be tested in the exploratory data analysis step. Also, other machine learning models would be employed (in particular, gradient boost models).

Besides, the model is deployed to production in an App at Heroku. One can send a request from an external application (such as Postman, for example). The app receives a JSON file and returns the sales forecast for the following six weeks (the amount is displayed in the Brazilian currency BRL). This app is also receiving requests from another app hooked to a Telegram Bot. In this case, one must only pass the number of the store to the Bot to obtain the sales forecast. This second App is also hosted at Heroku.
