import streamlit as st
import numpy as np
import pandas as pd
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, Markdown # Allows the use of display() for DataFrames


st.image('donors_image.png')

st.write('''## Getting Started

In this project, I employ several supervised algorithms to accurately model individuals' income using data.
          I will then choose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data.
          The goal with this implementation is to construct a model that accurately predicts whether an individual makes more than $50,000.
          This sort of task can arise in a non-profit setting, where organizations survive on donations.  Understanding an individual's income 
         can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with.
           While it can be difficult to determine an individual's general income bracket directly from public sources, I can
          infer this value from other publically available features. 

The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income).''')

################################################
################################################

st.write('''## Exploring the Data
 `'Income'`, will be the target label (whether an individual makes more than, or at most, $50,000 annually).
          All other columns are features about each individual in the census database.''')

code='''
import streamlit as st
import numpy as np
import pandas as pd
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
%matplotlib inline
'''
st.code(code, language='python')


# Load the Census dataset
data = pd.read_csv("census.csv")

code='''
data = pd.read_csv("census.csv")

'''
st.code(code, language='python')
st.dataframe(data)


################################################
################################################

st.write('''### Data Exploration
A cursory investigation of the dataset will determine how many individuals fit into either group, and will tell us about 
         the percentage of these individuals making more than \$50,000. I will compute the following:
- The total number of records, `'n_records'`
- The number of individuals making more than \$50,000 annually, `'n_greater_50k'`.
- The number of individuals making at most \$50,000 annually, `'n_at_most_50k'`.
- The percentage of individuals making more than \$50,000 annually, `'greater_percent'`.''')



code='''data.income.unique()
data.income = pd.Categorical(data.income, categories=['<=50K', '>50K'], ordered=True)
'''
st.code(code, language='python')




data.income = pd.Categorical(data.income, categories=['<=50K', '>50K'],
                              ordered=True)

code='''n_records = data.shape[0]
n_greater_50k = data.query('income ==">50K"').income.count()
n_at_most_50k = data.query('income =="<=50K"').income.count()
greater_percent = round(n_greater_50k/n_records,1)

print(f"###### Total number of records: {n_records:,}")
print(f"###### Individuals making more than $50,000: {n_greater_50k:,}")
print(f"###### Individuals making at most $50,000: {n_at_most_50k:,}")
print(f"###### Percentage of individuals making more than $50,000: {greater_percent:.0%}")'''
st.code(code)

n_records = data.shape[0]
n_greater_50k = data.query('income ==">50K"').income.count()
n_at_most_50k = data.query('income =="<=50K"').income.count()
greater_percent = round(n_greater_50k/n_records,1)

st.write(f"###### Total number of records: {n_records:,}")
st.write(f"###### Individuals making more than $50,000: {n_greater_50k:,}")
st.write(f"###### Individuals making at most $50,000: {n_at_most_50k:,}")
st.write(f"###### Percentage of individuals making more than $50,000: {greater_percent:.0%}")


st.write('''### **Feature set exploration**

* **age**: continuous. 
* **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
* **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
* **education-num**: continuous. 
* **marital-status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
* **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
* **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
* **race**: Black, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other. 
* **sex**: Female, Male. 
* **capital-gain**: continuous. 
* **capital-loss**: continuous. 
* **hours-per-week**: continuous. 
* **native-country**: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.''')


st.write('''### Preparing the Data
Data must be cleaned, formatted, and restructured — this is typically known as **preprocessing**. 
         Fortunately, for this dataset,
          there are no invalid or missing values, however, 
         there are some qualities about certain features
          that must be adjusted. This preprocessing can help tremendously with the outcome
          and predictive power of learning algorithms.''')

st.write('''#### Transforming Skewed Continuous Features
A dataset may sometimes contain at least one feature whose values tend to lie near a single number,
          but will also have a non-trivial number of vastly larger or smaller values than that single number.
           Algorithms can be sensitive to such distributions of values and can underperform if the range is not properly normalized. 
         With the census dataset two features fit this description: '`capital-gain'` and `'capital-loss'`. 

This is a histogram of the two features. Note the range of the values present and how they are distributed.''')



income_raw = data['income']
features_raw = data.drop('income', axis = 1)


code='''# Visualize skewed continuous features of original data
def distribution(data, transformed=False):
    """
    Visualization code for displaying skewed distributions of features
    """
    
    # Create figure
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))

    # Skewed feature plotting
    for i, feature in enumerate(['capital-gain', 'capital-loss']):
        ax = axes[i]
        ax.hist(data[feature], bins=25, color='#00A0A0')
        ax.set_title(f"'{feature}' Feature Distribution", fontsize=14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Census Data Features", fontsize=16, y=1.03)
    else:
        fig.suptitle("Skewed Distributions of Continuous Census Data Features", fontsize=16, y=1.03)

    fig.tight_layout()
    plt.show()
distribution(data)'''

st.code(code, line_numbers=True)

# Visualize skewed continuous features of original data
def distribution(data, transformed=False):
    """
    Visualization code for displaying skewed distributions of features
    """
    
    # Create figure
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))

    # Skewed feature plotting
    for i, feature in enumerate(['capital-gain', 'capital-loss']):
        ax = axes[i]
        ax.hist(data[feature], bins=25, color='#00A0A0')
        ax.set_title(f"'{feature}' Feature Distribution", fontsize=14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Census Data Features", fontsize=16, y=1.03)
    else:
        fig.suptitle("Skewed Distributions of Continuous Census Data Features", fontsize=16, y=1.03)

    fig.tight_layout()
    return fig

# Assuming 'data' is your DataFrame
fig = distribution(data)
st.pyplot(fig)



st.write('''For highly-skewed feature distributions such as `'capital-gain'` and `'capital-loss'`, it is common practice to apply a
         logarithmic transformation on the data so that the very large and very small values do not negatively affect the performance
          of a learning algorithm. Using a logarithmic transformation significantly reduces the range of values caused by outliers. 
         Care must be taken when applying this transformation however: The logarithm of `0` is undefined, so we must translate the values by a small amount above
          `0` to apply the the logarithm successfully.

This is the code and the result after transformation:''')


code='''
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

fig = distribution(features_log_transformed, transformed = True)
fig.show()

'''
st.code(code, language='Python')


skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

fig = distribution(features_log_transformed, transformed = True)
st.pyplot(fig)


################################################
################################################



st.write('---')
st.write('''### Normalizing Numerical Features
In addition to performing transformations on features that are highly skewed, it is often good practice to
          perform some type of scaling on numerical features.
          Applying a scaling to the data does not change the shape of each feature's distribution
          (such as `'capital-gain'` or `'capital-loss'` above);
          however, normalization ensures that each feature is treated equally when applying supervised learners.

''')


code='''
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])


# An example of a record with scaling applied
display(features_log_minmax_transform.head(n = 5))'''
st.code(code, language='Python')

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler


# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])


# Show an example of a record with scaling applied
st.dataframe(features_log_minmax_transform.head(n = 5))

st.write('---')

st.write('''### Implementation: Data Preprocessing

From the table in **Exploring the Data** above, there are several features for each record
          that are non-numeric. Typically, learning algorithms expect input to be numeric,
          which requires that non-numeric features (i.e. *categorical variables*) be converted. 
         One popular way to convert categorical variables is by using the **one-hot encoding** scheme.
          One-hot encoding creates a _"dummy"_ variable for each possible category of each non-numeric feature.
  
Additionally, as with the non-numeric features, I need to convert the non-numeric target label,
          `'income'` to numerical values for the learning algorithm to work. Since there are only two 
         possible categories for this label ("<=50K" and ">50K"),
          I will simply encode these two categories as `0` and `1`, respectively.
         ''')

code='''
features_final = pd.get_dummies(features_log_minmax_transform)

income = income_raw.apply(lambda x: 1 if x==">50K" else 0)

encoded = list(features_final.columns)
print(f"There are {len(encoded)} total features after one-hot encoding.")'''
st.code(code)


features_final = pd.get_dummies(features_log_minmax_transform)

income = income_raw.apply(lambda x: 1 if x==">50K" else 0)


# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
st.write(f"There are {len(encoded)} total features after one-hot encoding.")


st.write('---')

st.write('''### Shuffle and Split Data
Now all _categorical variables_ have been converted into numerical features, and all numerical features have been normalized. 
         I will now split the data (both features and their labels) into training and test sets. 80% of the data will be used for training and 20% for testing.
''')


code='''# Import train_test_split
from sklearn.model_selection import train_test_split


# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)'''
st.code(code)

# Import train_test_split
from sklearn.model_selection import train_test_split


# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)





st.write('''----
### Evaluating Model Performance
In this section, I will investigate four different algorithms, 
         and determine which is best at modeling the data.
        Three of these algorithms will be supervised learners,
          and the fourth algorithm is known as a *naive predictor*.''')


st.write('''#### Metrics and the Naive Predictor
*CharityML*, equipped with their research, knows individuals that make more than
          50,000 USD are most likely to donate to their charity. 
         Because of this, *CharityML* is particularly interested in predicting who makes more than 
         50,000 USD accurately. It would seem that using **accuracy** as a metric for evaluating 
         a particular model's performace would be appropriate.
         Additionally, identifying someone that *does not* make more than 50,000 USD as someone
          who does would be detrimental to *CharityML*, since they are looking to find individuals
          willing to donate. Therefore, a model's ability to precisely predict (**precision**)
          those that make more than 50,000 USD is *more important* than the model's ability to **recall** 
         those individuals. We can use **F-beta score** as 
         a metric that considers both precision and recall.''')


st.write('''### Question 1 - Naive Predictor Performace
* If we chose a model that always predicted an individual made more than $50,000, what would  that model's accuracy and F-score be on this dataset? 
        

The purpose of generating a naive predictor is simply to show what a base model without any intelligence would look like.
          In the real world, ideally your base model would be either the results of a previous model
          or could be based on a research paper upon which you are looking to improve. 
         When there is no benchmark model set, getting a result better than random choice is a place you could start from.



* When we have a model that always predicts '1' (i.e. the individual makes more than 50k) then our model will have no True Negatives(TN) or False Negatives(FN) as we are not making any negative('0' value) predictions. Therefore our Accuracy in this case becomes the same as our Precision(True Positives/(True Positives + False Positives)) as every prediction that we have made with value '1' that should have '0' becomes a False Positive; therefore our denominator in this case is the total number of records we have in total. 
* Our Recall score(True Positives/(True Positives + False Negatives)) in this setting becomes 1 as we have no False Negatives.''')



income=income.astype('int')
TP = np.sum(income) 
# Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data 
#encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case

code='''income=income.astype('int')
TP = np.sum(income) 
# Counting the ones as this is the naive case. 
# Encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case'''
st.code(code)



code = '''
accuracy = TP/income.shape[0]
recall = TP/(TP+FN)
precision = TP/(TP+FP)

# Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
B=0.5
fscore = (1+B**2)*(precision*recall)/((B**2*precision)+recall)


# Print the results 
print(f"Naive Predictor: [Accuracy score: {accuracy:.2%}, F-score: {fscore:.2%}]")'''
st.code(code, line_numbers=True)

# TODO: Calculate accuracy, precision and recall
accuracy = TP/income.shape[0]
recall = TP/(TP+FN)
precision = TP/(TP+FP)

# TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
B=0.5
fscore = (1+B**2)*(precision*recall)/((B**2*precision)+recall)
st.write(f"Naive Predictor: [Accuracy score: {accuracy:.2%}, F-score: {fscore:.2%}]")



st.write('---')

st.write('''###  Supervised Learning Models
I considered these supervised learning models that are currently available in [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) to choose from:
- Gaussian Naive Bayes (GaussianNB)
- Decision Trees
- Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
- K-Nearest Neighbors (KNeighbors)
- Stochastic Gradient Descent Classifier (SGDC)
- Support Vector Machines (SVM)
- Logistic Regression''')




st.write('''**My picks:**
         

#### 1.**AdaBoost Classifier:**


 **Advantages**: 

1. Its ease of use stems from minimal hyperparameter tuning requirements. Unlike many other algorithms, it requires minimal hyperparameter tuning, making it user-friendly.

2. AdaBoost demonstrates reduced vulnerability to overfitting. It displays a lower tendency to overfitting, enhancing its robustness in model training.

**Disadvantages**:

1. AdaBoost's susceptibility to noisy data and outliers means that the presence of outliers can notably influence the algorithm's performance, as these data points may receive substantial weight in the training process.

2.  The process of training an AdaBoost model can be computationally demanding, requiring significant computing resources.

3. AdaBoost is primarily tailored for addressing linear classification problems. In cases where the data exhibits intricate, non-linear relationships, AdaBoost may not perform as effectively as other algorithms specialized in capturing such non-linear patterns.

**A good candidate for the problem:**
AdaBoost is good to use for this kind of problem, because it can utilize other good classification algorithms, like  decision trees, to build the weak learners and deliver good results. 


#### 2.**Decision Tree Classifier:**
         
**Advantages**:
1. The primary benefit of decision trees is their exceptional ability to interpret and visually represent non-linear data patterns with ease.
2. Furthermore, they exhibit high speed, particularly during exploratory data analysis.
3. Able to handle both numerical and categorical data.

**Disadvantages**: 
1. They are prone to overfitting, especially when they are deep and highly complex.

2. A slight variation can lead to significantly different tree structures, making them less robust.

3. Greedy nature: Decision trees use a greedy algorithm that makes splits based on the best feature at each node. This can lead to suboptimal splits further down the tree. 

4. Complexity:  decision trees can become complex and hard to interpret for large datasets or deep trees.

**A good candidate for the problem**: This problem is a categorical one, and a decision tree should work well. 
         A decision tree classifier can assist us in pinpointing the most influential factor and a decision path that we can examine for a 
         given set of variables to determine whether an individual's income exceeds 50,000 USD. The model evaluates the information gain of 
         each split within the dataset, allowing us to efficiently identify the variables that play a pivotal role in determining whether a 
         person earns more than 50,000 USD and those that conclusively indicate otherwise.


#### 3.**Logistic Regression:**

**Advantages**:
1. Easy to implement and interpret yet efficient in training. Performs well on low-dimensional data.
2. Performs well on low-dimensional data.


**Disadvantages**: 
Logistic regression has limitations when dealing with non-linear problems.

**A good candidate for the problem:**
This type of statistical model  is often used for classification and predictive analytics. ''')



st.write('''### Implementation - Creating a Training and Predicting Pipeline
To properly evaluate the performance of each model I've chosen, I will create a training and predicting pipeline that allows to quickly 
         and effectively train models using various sizes of training data and perform predictions on the testing data.
In the code block below, I will :
 - Import `fbeta_score` and `accuracy_score` from [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).
 - Fit the learner to the sampled training data and record the training time.
 - Perform predictions on the test data `X_test`, and also on the first 300 training points `X_train[:300]`.
   - Record the total prediction time.
 - Calculate the accuracy score for both the training subset and testing set.
 - Calculate the F-score for both the training subset and testing set.
   ''')


code='''
# Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
  
    
    results = {}
    
    # Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end - start
        
    #  Get the predictions on the test set(X_test),
    #  Then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # alculate the total prediction time
    results['pred_time'] = end - start
            
    # Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)
        
    # Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)
       
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    return results
'''
st.code(code)

# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score


def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
   
    start = time() # Get start time
    learner = learner
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
 
    results['train_time'] = end - start
        

    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
  
    results['pred_time'] = end - start
            

    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        

    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)
        
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)
       
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    return results



st.write('''### Implementation: Initial Model Evaluation
In the code cell, I will :
- Import the three supervised learning models I've chosen in the previous section.
- Initialize the three models and store them in `'clf_A'`, `'clf_B'`, and `'clf_C'`.
  - Use a `'random_state'` for each model I use, if provided.
- Calculate the number of records equal to 1%, 10%, and 100% of the training data.
  - Store those values in `'samples_1'`, `'samples_10'`, and `'samples_100'` respectively.''')


code='''# Import the three supervised learning models from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

'''
st.code(code)

# Import the three supervised learning models from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

code='''# Initialize the three models
clf_A = AdaBoostClassifier(random_state=40)
clf_B = DecisionTreeClassifier(random_state=40)
clf_C = LogisticRegression(random_state=40)

# Calculate the number of samples for 1%, 10%, and 100% of the training data
#  samples_100 is the entire training set i.e. len(y_train)
#  samples_10 is 10% of samples_100 )
#  samples_1 is 1% of samples_100)
samples_100 = int(len(y_train)) 
samples_10 = int(samples_100 * 0.1)
samples_1 = int(samples_100 * 0.01)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)


def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - results: a dictionary of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """
  
    # Create figure
    fig, ax = plt.subplots(2, 3, figsize=(11, 7))
    
    # Constants
    bar_width = 0.3
    colors = ['#A00000', '#00A0A0', '#00A000']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                # Creative plot code
                ax[j//3, j%3].bar(i + k * bar_width, results[learner][i][metric], width=bar_width, color=colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size")
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[0, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))
    
    # Create patches for the legend
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=colors[i], label=learner) for i, learner in enumerate(results.keys())]
    # plt.legend(handles=patches, bbox_to_anchor=(-.80, 2.53), loc='upper center', borderaxespad=0., ncol=3, fontsize='x-large')
    
    # Aesthetics
    plt.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize=16, y=1.10)
    plt.tight_layout()
    
    return fig
    
  evaluate(results, accuracy, fscore)'''
st.code(code, line_numbers=True)




# Initialize the three models
clf_A = AdaBoostClassifier(random_state=40)
clf_B = DecisionTreeClassifier(random_state=40)
clf_C = LogisticRegression(random_state=40)

# Calculate the number of samples for 1%, 10%, and 100% of the training data
#  samples_100 is the entire training set i.e. len(y_train)
#  samples_10 is 10% of samples_100 (set the count of the values to be `int` and not `float`)
#  samples_1 is 1% of samples_100 (set the count of the values to be `int` and not `float`)
samples_100 = int(len(y_train)) 
samples_10 = int(samples_100 * 0.1)
samples_1 = int(samples_100 * 0.01)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)



def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - results: a dictionary of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """
  
    # Create figure
    fig, ax = plt.subplots(2, 3, figsize=(11, 7))
    
    # Constants
    bar_width = 0.3
    colors = ['#A00000', '#00A0A0', '#00A000']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                # Creative plot code
                ax[j//3, j%3].bar(i + k * bar_width, results[learner][i][metric], width=bar_width, color=colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size")
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[0, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))
    
    # Create patches for the legend
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=colors[i], label=learner) for i, learner in enumerate(results.keys())]
    # plt.legend(handles=patches, bbox_to_anchor=(-.80, 2.53), loc='upper center', borderaxespad=0., ncol=3, fontsize='x-large')
    
    # Aesthetics
    plt.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize=16, y=1.10)
    plt.tight_layout()
    
    return fig


fig = evaluate(results, accuracy, fscore)
st.pyplot(fig)


st.write('''----
## Improving Results
In this final section, I choose from the three supervised learning models the *best* model to use on the student data. 
         I will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at
          least one parameter to improve upon the untuned model's F-score. ''')


st.write('''### Question 3 - Choosing the Best Model''')

st.write('''
I select the AdaBoostClassifier as the top-performing model among the three options for the classification task of distinguishing individuals with incomes exceeding 50,000 USD.

`Accuracy`: Examining the bottom-middle plot, the AdaBoost Classifier achieved the highest accuracy score, surpassing 80\%, which was notably
          superior to both the Logistic Regression and the Decision Tree Classifier. Additionally, the AdaBoost Classifier maintained a
          superior accuracy score when using either 1\% or 10\% of the testing set. It's worth noting that, although the Decision Tree
          Classifier achieved the highest accuracy on training sets of all sizes, it exhibited a significantly lower accuracy score, indicating potential overfitting.

`F-score`: Most importantly, the AdaBoost algorithm delivered the highest F-score of approximately 70\% on the test dataset,
          consistently outperforming the other two models across all three dataset sizes. The bottom-right plot illustrates
          the AdaBoostClassifier (highlighted in red) achieving the highest F-score on 1\%, 10\%, or 100\% of the test dataset.

While the AdaBoost Classifier required slightly more time than the other two models for both training and prediction,
          the absolute time costs were minimal: less than 2 seconds for training and less than 1/10 of a second for prediction. 
         Ultimately, the AdaBoost Classifier emerged as the most accurate and precise model among the three options.

''')




st.write('''### Implementation: Model Tuning
         
I will fine tune the chosen model. I will use grid search (`GridSearchCV`) with two important parameters tuned with
          at a few different values. I will need to use the entire training set for this.''')


code='''#  Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  make_scorer


# Initialize the classifier
clf = AdaBoostClassifier(estimator = DecisionTreeClassifier(), random_state=40)

# Create the parameters list you wish to tune, using a dictionary if needed.
parameters = {'n_estimators': [4, 10]
             ,'learning_rate': [0.1, 1.]}



# The fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score,beta=0.5)

start = time()

# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf,parameters,scoring=scorer)

# Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train,y_train)


# Get the estimator
best_clf = grid_fit.best_estimator_


# Make predictions using the unoptimized and optimized model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

end = time()



# Report the before-and-afterscores
print("Unoptimized model")
print("Accuracy score on testing data: {:.2%}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.2%}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model")
print("Final accuracy score on the testing data: {:.2%}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.2%}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))

print("It took ",round((end - start)/60,2),"minutes to run `GridSearchCV`.")

print(best_clf)'''
st.code(code, line_numbers=True)



#  Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  make_scorer


# Initialize the classifier
clf = AdaBoostClassifier(estimator = DecisionTreeClassifier(), random_state=40)

# Create the parameters list you wish to tune, using a dictionary if needed.
parameters = {'n_estimators': [4, 10]
             ,'learning_rate': [0.1, 1.]}



# The fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score,beta=0.5)

start = time()

# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf,parameters,scoring=scorer)

# Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train,y_train)


# Get the estimator
best_clf = grid_fit.best_estimator_


# Make predictions using the unoptimized and optimized model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

end = time()





# Report the before-and-afterscores
st.write('''##### Unoptimized model''')
st.write(''' `Accuracy` score on `testing` data: {:.2%}'''.format(accuracy_score(y_test, predictions)))
st.write(''' F-score on testing data: {:.2%}'''.format(fbeta_score(y_test, predictions, beta = 0.5)))
st.write('''##### Optimized Model''')
st.write(''' Final accuracy score on the testing data: {:.2%}'''.format(accuracy_score(y_test, best_predictions)))
st.write(''' Final F-score on the testing data: {:.2%}'''.format(fbeta_score(y_test, best_predictions, beta = 0.5)))

st.write('''It took ''',round((end - start)/60,2),'''minutes to run `GridSearchCV`.''')
st.write('''The optimized model:''')
st.write(best_clf)



st.write('''### Question 5 - Final Model Evaluation

* What is the optimized model's accuracy and F-score on the testing data? 
* Are these scores better or worse than the unoptimized model? 
* How do the results from the optimized model compare to the naive predictor benchmarks?''')


st.write(f'''#### Results:

|     Metric     | Unoptimized Model | Optimized Model |
| :------------: | :---------------: | :-------------: | 
| Accuracy Score | {accuracy_score(y_test, predictions):.2%}  |     {accuracy_score(y_test, best_predictions):.2%}      |
| F-score        |         {fbeta_score(y_test, predictions, beta = 0.5):.2%}     |     {fbeta_score(y_test, best_predictions, beta = 0.5):.2%}       |
''')


st.write(f'''
By using GridSearchCV, I optimised the AdaBoost algorithm by testing different number of learners created.
          Specifically, GridSearchCV created 4 and 10 learners  and used 0.1 and 1 learning rates and investigated
          when AdaBoost achieved its best performance by looking at the accuracy score and the F-score. 
The optimised model achieved {accuracy_score(y_test, best_predictions):.1%}  ( against {accuracy_score(y_test, predictions):.1%} ) accuracy score
 and an F-score {fbeta_score(y_test, best_predictions, beta = 0.5):.1%} vs the unoptimized model's {fbeta_score(y_test, predictions, beta = 0.5):.1%}.

As a reminder, the naive predictor's accuracy score was {accuracy:.2%} and its F-score was {fscore:.2%}, thus there is a huge improvement when using the AdaBoost algorithm.
''') 


st.write('''## Feature Importance

An important task when performing supervised learning on a dataset  is determining
          which features provide the most predictive power. By focusing on the relationship between only a few crucial
          features and the target label we simplify our understanding of the phenomenon, which is most always a useful 
         thing to do. In the case of this project, I will identify a small number of features that most strongly
          predict whether an individual makes at most or more than 50,000 USD.

I will choose a scikit-learn classifier (e.g., AdaBoost, Random Forests) that has a `feature_importance_` attribute, which is a function that ranks
          the importance of features according to the chosen classifier.
           I will fit this classifier to training set and use this attribute to determine the top 5 most important features for the census dataset.''')



st.write('''### Implementation - Extracting Feature Importance
I will choose a `scikit-learn` supervised learning algorithm that has a `feature_importance_`
          attribute available for it. This attribute is a function 
         that ranks the importance of each feature when making predictions based on the chosen algorithm.

In the code cell below, I will :
 - Import a supervised learning model from sklearn.
 - Train the supervised model on the entire training set.
 - Extract the feature importances using `'.feature_importances_'`.''')


code='''# Import a supervised learning model that has 'feature_importances_'
from sklearn.tree import DecisionTreeClassifier 

# Train the supervised model on the training set using .fit(X_train, y_train)
model = DecisionTreeClassifier()
clf=model.fit(X_train, y_train)

# Extract the feature importances using .feature_importances_ 
importances = clf.feature_importances_


def feature_plot(importances, X_train, y_train):
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Create the plot
    fig = plt.figure(figsize=(9, 5))
    plt.title("Normalized Weights for First Five Most Predictive Features", fontsize=16)
    plt.bar(np.arange(5), values, width=0.6, align="center", color='#00A000', label="Feature Weight")
    plt.bar(np.arange(5) - 0.3, np.cumsum(values), width=0.2, align="center", color='#00A0A0', label="Cumulative Feature Weight")
    plt.xticks(np.arange(5), columns)
    plt.xlim((-0.5, 4.5))
    plt.ylabel("Weight", fontsize=12)
    plt.xlabel("Feature", fontsize=12)
    plt.legend(loc='upper center')
    plt.tight_layout()
    return fig

fig = feature_plot(importances, X_train, y_train)
plt.pyplot(fig)'''
st.code(code, line_numbers=True)



# Import a supervised learning model that has 'feature_importances_'
from sklearn.tree import DecisionTreeClassifier 

# Train the supervised model on the training set using .fit(X_train, y_train)
model = DecisionTreeClassifier()
clf=model.fit(X_train, y_train)

# Extract the feature importances using .feature_importances_ 
importances = clf.feature_importances_


def feature_plot(importances, X_train, y_train):
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Create the plot
    fig = plt.figure(figsize=(9, 5))
    plt.title("Normalized Weights for First Five Most Predictive Features", fontsize=16)
    plt.bar(np.arange(5), values, width=0.6, align="center", color='#00A000', label="Feature Weight")
    plt.bar(np.arange(5) - 0.3, np.cumsum(values), width=0.2, align="center", color='#00A0A0', label="Cumulative Feature Weight")
    plt.xticks(np.arange(5), columns)
    plt.xlim((-0.5, 4.5))
    plt.ylabel("Weight", fontsize=12)
    plt.xlabel("Feature", fontsize=12)
    plt.legend(loc='upper center')
    plt.tight_layout()
    return fig

fig = feature_plot(importances, X_train, y_train)
st.pyplot(fig)



st.write('''### Feature Selection
How does a model perform if we only use a subset of all the available
          features in the data? With less features required to train,
          the expectation is that training and prediction time is much lower
          — at the cost of performance metrics. From the visualization above,
          I see that the top five most important features contribute more than
          half of the importance of **all** features present in the data. This 
         hints that I can attempt to *reduce the feature space* and simplify the
          information required for the model to learn. The code cell below will use 
         the same optimized model, and train it on the same training 
         set *with only the top five important features*. ''')



code='''# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.2%}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.2%}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.2%}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.2%}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))'''
st.code(code)

# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
st.write("##### Final Model trained on full data")
st.write("Accuracy on testing data: {:.2%}".format(accuracy_score(y_test, best_predictions)))
st.write("F-score on testing data: {:.2%}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
st.write("##### Final Model trained on reduced data")
st.write("Accuracy on testing data: {:.2%}".format(accuracy_score(y_test, reduced_predictions)))
st.write("F-score on testing data: {:.2%}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))


st.write('''### Effects of Feature Selection

* How does the final model's F-score and accuracy score
          on the reduced data using only five features
          compare to those same scores when all features are used?
* If training time was a factor, would I consider using the reduced data as your training set?''')

acc_red=accuracy_score(y_test, reduced_predictions)
acc_full=accuracy_score(y_test, best_predictions)
f_red=fbeta_score(y_test, reduced_predictions, beta=0.5)
f_full=fbeta_score(y_test, best_predictions, beta=0.5)


st.write(f'''**Answer:**
The final model experienced a slight decrease in accuracy
          ({acc_red-acc_full:.2%}) compared to the original model, which utilized
          all available features. This decline in accuracy is relatively minimal.
          Similarly, the final model demonstrated a decrease of {f_red-f_full:.2%} in its F-score.
          Consequently, the model that utilized only five selected features exhibited both
          a slightly reduced accuracy and F-score. Evaluating the results
         , particularly when considering the potential cost of training time, it becomes
          worthwhile to seriously contemplate deploying the reduced-feature model given the 
         modest impact on accuracy and F-score.''')

