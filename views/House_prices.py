

import streamlit as st
st.title(' Predicting Boston house prices')  

st.image('Boston_houses_USA.png', width=800)

st.write('''In this project, I will evaluate the performance and predictive power of a model
          that has been trained and tested on data collected from homes in suburbs of Boston,
          Massachusetts. A model trained on this data that is seen as a *good fit* could then
          be used to make certain predictions about a home — in particular, its monetary value.''')

st.write('''Each of the entries in the dataset represent aggregated data 
         about 14 features for these homes. ''')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from IPython.display import display, Markdown

code='''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from IPython.display import display, Markdown
'''
st.code(code, language='python')


code='''data=pd.read_csv('./housing.csv')
data.head()'''
st.code(code)

data=pd.read_csv('./housing.csv')
st.dataframe(data.head())

prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

st.write('''
I will be using three features from the Boston housing dataset: `'RM'`, `'LSTAT'`, and `'PTRATIO'`. For each data point:
- `'RM'` is the average number of rooms among homes in the neighborhood.
- `'LSTAT'` is the percentage of lower income level households in neighborhood.
- `'PTRATIO'` is the ratio of students to teachers in primary and secondary schools in the neighborhood.''')
 
st.write('## Data Exploration')

st.write('''In this first section of this project, I will make a cursory investigation about the Boston housing data and provide my observations. 
         Familiarizing with the data through an explorative process is a fundamental practice to help better understand and justify the results.''')

st.write('''Since the main goal of this project is to 
         construct a working model which has the capability
          of predicting the value of houses, I will need to
          separate the dataset into **features** and the
          **target variable**. The **features**,
          `'RM'` , `'LSTAT'` , and `'PTRATIO'`,
          give us quantitative information about each data point. 
         The **target variable**, `'MEDV'`, will be the variable to predict. These are stored in `features` and `prices`,
          respectively.''')


code='''
data.sort_values(by=['MEDV'], ascending=False).head(10))
'''
st.code(code, language='python')
st.dataframe(data.sort_values(by=['MEDV'], ascending=False).head(10))



st.write('''**Insight**: There are only 3 houses valued over $1m. The most expensive houses have 7 to 8 rooms, a low ratio of students to teacher and a very small % of low income families in the neighborhood.''')



code='''
data.describe()
'''
st.code(code, language='python')
st.dataframe(data.describe())






code='''
params = {
    'features':['RM', 'LSTAT', 'PTRATIO'],
    'title':['Number of rooms distribution', "% Low Income families distribution",
             "Pupils to teacher ratio distribution"],
    'x_label': ['RM', 'LSTAT', 'PTRATIO'],
    'color': ['blue', 'pink', 'purple']
}

fig=plt.figure(figsize=(15,4))
for i in range(3):
        
        ax=plt.subplot(1,3,i+1)
        ax.hist(data[params['features'][i]], bins = 12, color=params['color'][i])
        plt.title(params['title'][i])
        plt.xlabel(params['x_label'][i])
        plt.ylabel("frequency")
'''
st.code(code, language='python')

# Plot parameters
params = {
    'features': ['RM', 'LSTAT', 'PTRATIO'],
    'title': ['Number of rooms distribution', "% Low Income families distribution",
              "Pupils to teacher ratio distribution"],
    'x_label': ['RM', 'LSTAT', 'PTRATIO'],
    'color': ['blue', 'pink', 'purple']
}

# Create the figure and subplots
fig = plt.figure(figsize=(15, 4))
for i in range(3):
    ax = plt.subplot(1, 3, i + 1)
    ax.hist(data[params['features'][i]], bins=12, color=params['color'][i])
    plt.title(params['title'][i])
    plt.xlabel(params['x_label'][i])
    plt.ylabel("Frequency")

# Display the plot in Streamlit
st.pyplot(fig)

st.write('**Insight**: The distribution of the number of rooms resembles a normal distribution. The LSTAT data is skewed to the right, while the PTRATIO data is skewed to the left.')

st.write('''### Implementation: Calculate Statistics
I will calculate descriptive statistics about the Boston housing prices. These statistics will be extremely important later on to analyze various prediction results from the constructed model.
''')


code='''
minimum_price = data.MEDV.min()

maximum_price = data.MEDV.max()

mean_price = round(data.MEDV.mean(),1)

median_price = data.MEDV.median() 

std_price = round(data.MEDV.std(),1)

print("**Statistics for Boston housing dataset:**")
print(f"Minimum price: ${minimum_price:,.0f}" )
print(f"Maximum price: ${maximum_price:,.0f}")
print(f"Mean price: ${mean_price:,.0f}")
print(f"Median price: ${median_price:,.0f}")  
print(f"Standard deviation of prices: ${std_price:,.0f}")'''
st.code(code)


# TODO: Minimum price of the data
minimum_price = data.MEDV.min()

# TODO: Maximum price of the data
maximum_price = data.MEDV.max()

# TODO: Mean price of the data
mean_price = round(data.MEDV.mean(),1)

# TODO: Median price of the data
median_price = data.MEDV.median()

# TODO: Standard deviation of prices of the data
std_price = round(data.MEDV.std(),1)

# Show the calculated statistics
st.write("**Statistics for Boston housing dataset:**")
st.write(f"Minimum price: ${minimum_price:,.0f}" )
st.write(f"Maximum price: ${maximum_price:,.0f}")
st.write(f"Mean price: ${mean_price:,.0f}")
st.write(f"Median price: ${median_price:,.0f}")  
st.write(f"Standard deviation of prices: ${std_price:,.0f}")



st.write('''---

## Developing a Model
In this second section of the project, I will develop the tools and
          techniques necessary for a model to make a prediction. Being 
         able to make accurate evaluations of each model's performance
          through the use of these tools and techniques helps to greatly 
         reinforce the confidence in my predictions.''') 


st.write('''### Implementation: Define a Performance Metric
It is difficult to measure the quality of a given model without quantifying its performance over
 training and testing.
I will be calculating the *coefficient of determination*
(http://stattrek.com/statistics/dictionary.aspx?definition=coefficient_of_determination), $R^2$, 
to quantify the model's performance. The coefficient of determination describes how
"good" that model is at making predictions. 
The values for $R^2$ range from 0 to 1, which captures the percentage of squared
 correlation between the predicted and actual values of the **target variable**.
A model with an $R^2$ of 0 is no better than a model that always predicts the *mean* 
of the target variable, whereas a model with an $R^2$ of 1 perfectly predicts the target variable.
Any value between 0 and 1 indicates what percentage of the target variable, 
using this model, can be explained by the **features**.''')


code='''
from sklearn.metrics import r2_score
def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    
    return score
'''
st.code(code, language='python')

from sklearn.metrics import r2_score
def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)
    
    # Return the score
    return score


st.write('''### Implementation: Shuffle and Split Data
''')


code='''
from sklearn.cross_validation import train_test_split
X = np.array(data.iloc[:,:3])
y=np.array(data.loc[:,'MEDV'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)'''
st.code(code, language='python')

from sklearn.model_selection import train_test_split
X = np.array(data.iloc[:,:3])
y=np.array(data.loc[:,'MEDV'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



st.write('''
## Evaluating Model Performance
In this final section of the project, I will construct a model and make
 a prediction on the client's feature set using an optimized model from `fit_model`.''')


st.write('''### Implementation: Fitting a Model
I will train a model using the **decision tree algorithm**.
To ensure that I create an optimized model, I will train the model using the **grid search** technique to optimize the
 `'max_depth'` parameter for the decision tree. 
The `'max_depth'` parameter can be thought of as how many questions
 the decision tree algorithm is allowed to ask about the data before making a prediction.

In addition, I will use
 `ShuffleSplit()` for an alternative form of cross-validation
  (see the `'cv_sets'` variable). The `ShuffleSplit()` implementation below will
 create 10 (`'n_splits'`) shuffled sets, and for each shuffle,
  20% (`'test_size'`) of the data will be used as the *validation set*.

For the `fit_model` function in the code cell below, i will implement the following:
- Use [`DecisionTreeRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) from `sklearn.tree` to create a decision tree regressor object.
  - Assign this object to the `'regressor'` variable.
- Create a dictionary for `'max_depth'` with the values from 1 to 10, and assign this to the `'params'` variable.
- Use [`make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html) from `sklearn.metrics` to create a scoring function object.
  - Pass the `performance_metric` function as a parameter to the object.
  - Assign this scoring function to the `'scoring_fnc'` variable.
- Use [`GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) from `sklearn.grid_search` to create a grid search object.
  - Pass the variables `'regressor'`, `'params'`, `'scoring_fnc'`, and `'cv_sets'` as parameters to the object. 
  - Assign the `GridSearchCV` object to the `'grid'` variable.''')




code='''

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV




def fit_model(X, y):
    
    cv_sets = ShuffleSplit( n_splits = 10, test_size = 0.20, random_state = 0)

    regressor = DecisionTreeRegressor()

    parameters={'max_depth': [1,2,3,4,5,6,7,8,9,10]}

    scoring_fnc = make_scorer(performance_metric)

    grid = GridSearchCV(regressor, parameters, scoring=scoring_fnc, cv=cv_sets)

    grid = grid.fit(X, y)

    return grid
    '''
st.code(code, language='python')


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score
from sklearn.datasets import load_diabetes


def fit_model(X, y):

    cv_sets = ShuffleSplit( n_splits = 10, test_size = 0.20, random_state = 0)

    regressor = DecisionTreeRegressor()

    parameters={'max_depth': [1,2,3,4,5,6,7,8,9,10]}

    scoring_fnc = make_scorer(performance_metric)

    grid = GridSearchCV(regressor, parameters, scoring=scoring_fnc, cv=cv_sets)

    grid = grid.fit(X, y)

    return grid





st.write('''### Making Predictions
Once a model has been trained on a given set of data, it can now be used to make predictions on new sets of input data. In the case of a *decision tree regressor*, the model has learned *what the best questions to ask about the input data are*, and can respond with a prediction for the **target variable**. You can use these predictions to gain information about data where the value of the target variable is unknown — such as data the model was not trained on.''')



code = '''reg = fit_model(X_train, y_train)'''
st.code(code, language='python')



reg = fit_model(X_train, y_train)
code = '''print(f'The $R^2$ score is {reg.best_score_:.2}.') 
'''
st.code(code)

st.write(f'''The model's $R^2$ score is {reg.best_score_:.2}.''') 

st.write(f'''An R²  value of {reg.best_score_:.2} indicates
          that the model explains {reg.best_score_:.0%} of the variance in the dependent variable 
         using the independent variables. This is a fairly strong indication that the
          model has a good fit, as it captures a significant portion of the data's variability.''')

# st.write(f"Parameter 'max_depth' is {reg.get_params()['max_depth']} for the optimal model.")



st.write('''### Predicting Selling Prices
Imagine that a real estate agent in the Boston area looking to use this model to help price homes owned by
          her clients that they wish to sell. 
         She has collected the following information from three of her clients:

         

| Feature | Client 1 | Client 2 | Client 3 |
| :---: | :---: | :---: | :---: |
| Total number of rooms in home | 5 rooms | 4 rooms | 8 rooms |
| Neighborhood income level (as %) | 17% | 32% | 3% |
| Student-teacher ratio of nearby schools | 15-to-1 | 22-to-1 | 12-to-1 |
''')


code='''client_data = [[5, 17, 15],
              [4, 32, 22],
              [8, 3, 12]]

for i, number in enumerate(reg.predict(client_data)):
                          print(f'Predicted selling price for Client{i+1}: ${number:,.0f}')'''
st.code(code)


client_data = [[5, 17, 15],
              [4, 32, 22],
              [8, 3, 12]]

for i, number in enumerate(reg.predict(client_data)):
                          st.write(f'''Predicted selling price
                                    for Client {i+1}: ${number:,.0f}''')


code='''

ii=['Client 1','Client 2','Client 3']
xx=['red','orange', 'pink']
vv=range(3)

for i, x, v in zip(ii,xx,vv):

    fig= plt.figure(figsize=(15,4))
    aa=range(131, 134)
    bb=params['features']
    cc=range(3)
    dd=params['x_label']


    for a, b ,c, d in zip(aa, bb, cc, dd):
        ax2=fig.add_subplot(a)
        ax2.scatter(data['MEDV'], data[b])
        ax2.scatter(list(reg.predict(client_data))[v],client_data[v][c], color=x, marker='o',s=350)
        plt.ylabel(d)
        plt.xlabel('MEDV')
    fig.suptitle(i, color=x, fontsize=20)
'''
st.code(code, language='python')




params = {
    'features': ['RM', 'LSTAT', 'PTRATIO'],
    'title': ['Number of rooms distribution', "% Low Income families distribution",
              "Pupils to teacher ratio distribution"],
    'x_label': ['RM', 'LSTAT', 'PTRATIO'],
    'color': ['blue', 'pink', 'purple']
}

ii = ['Client 1', 'Client 2', 'Client 3']
xx = ['red', 'orange', 'pink']
vv = range(3)

# Plotting
for i, x, v in zip(ii, xx, vv):
    fig = plt.figure(figsize=(15, 4))
    aa = range(131, 134)
    bb = params['features']
    cc = range(3)
    dd = params['x_label']

    for a, b, c, d in zip(aa, bb, cc, dd):
        ax2 = fig.add_subplot(a)
        ax2.scatter(data['MEDV'], data[b])
        ax2.scatter([reg.predict([client_data[v]])[0]], [client_data[v][c]], color=x, marker='o', s=350)
        plt.ylabel(d)
        plt.xlabel('MEDV')
    fig.suptitle(i, color=x, fontsize=20)

    # Display the plot in Streamlit
    st.pyplot(fig)



st.write('''The characteristics of Client 1's house (red dots) closely match the average features of the dataset. Additionally, the average price in the dataset aligns with the model's suggested price, making it a well-aligned prediction.

For Client 2, the suggested price aligns with expectations. The low number of rooms, high percentage of low-income families, and elevated pupil-to-teacher ratio (orange dots) justify the lower predicted price.

Finally, the predicted price for Client 3's house is well-supported by the neighborhood's high-quality attributes, including a significant proportion of high-income families, a very low pupil-to-teacher ratio, and a large number of rooms (pink dots).''')


st.write('### Conclusion')

st.write('''This project has demonstrated the process of predicting
          Boston housing prices using a decision tree regression model.
          By thoroughly exploring the dataset and leveraging key features such
          as 'RM', 'LSTAT', and 'PTRATIO', we established a solid foundation for
          creating a predictive model. The insights gained through data analysis and
          visualizations provided a deeper understanding of how different factors influence 
         house prices in Boston.

The constructed model achieved a reasonably 
         high $R^2$ score, indicating that it effectively captures the 
         variability in housing prices based on the given features.
          Predictions for the test cases aligned well with expectations,
          underscoring the model's practical applicability.
          This project illustrates the importance of careful data exploration,
          model evaluation, and interpretability in building robust predictive tools
          that can aid real-world decision-making, such as pricing homes accurately for clients.''')
