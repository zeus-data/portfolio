import streamlit as st

st.title('''Lab: Titanic Survival Exploration with Decision Trees''')


st.write('''## Getting Started
Implementing a decision tree algorithm on the Titanic dataset is one way to predict whether
          a passenger survived or not. The Titanic dataset contains various features such as age,
          gender, passenger class, and more, which can be used to build a predictive model. 
         By training a decision tree on this data, we can identify patterns and relationships 
         between these features and the survival outcome. This approach not only helps in understanding
          the factors that influenced
          survival but also demonstrates the power of decision trees in classification tasks.

I start by loading the dataset and display the table.''')

code='''import numpy as np
import pandas as pd
import random
# Set a random seed
random.seed(42)

# Load the dataset
in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)'''
st.code(code)

import numpy as np
import pandas as pd
import random
# Set a random seed
random.seed(42)

# Load the dataset
in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)

st.dataframe(full_data)


st.write('''These are the various features present for each passenger on the ship:
- **Survived**: Outcome of survival (0 = No; 1 = Yes)
- **Pclass**: Socio-economic class (1 = Upper class; 2 = Middle class; 3 = Lower class)
- **Name**: Name of passenger
- **Sex**: Sex of the passenger
- **Age**: Age of the passenger (Some entries contain `NaN`)
- **SibSp**: Number of siblings and spouses of the passenger aboard
- **Parch**: Number of parents and children of the passenger aboard
- **Ticket**: Ticket number of the passenger
- **Fare**: Fare paid by the passenger
- **Cabin** Cabin number of the passenger (Some entries contain `NaN`)
- **Embarked**: Port of embarkation of the passenger (C = Cherbourg; Q = Queenstown; S = Southampton)

Since I'm interested in the outcome of survival for each passenger or crew member, I can remove the 
         **Survived** feature from this dataset and store it as its own separate variable `outcomes`.
          These will be the prediction targets.''')


code='''# Survived' feature stored in a new variable and removed from the dataset
outcomes = full_data['Survived']
features_raw = full_data.drop('Survived', axis = 1)

# The new dataset with 'Survived' removed
features_raw.head()
'''
st.code(code)

# Survived' feature stored in a new variable and removed from the dataset
outcomes = full_data['Survived']
features_raw = full_data.drop('Survived', axis = 1)

# Show the new dataset with 'Survived' removed
st.dataframe(features_raw.head())

st.write('''The sample of the RMS Titanic data now shows the **Survived** feature removed from the DataFrame.
          Note that `data` 
         (the passenger data) and `outcomes` (the outcomes of survival) are now *paired*. 
         That means for any passenger `data.loc[i]`, they have the survival outcome `outcomes[i]`.

## Preprocessing the data

First, I will remove the names of the passengers,
          and then `one-hot encode` the features.
         `One-hot encoding` is a technique used to convert categorical variables into a numerical
          format that machine learning algorithms can process. It works by creating a new binary 
         column for each unique category in the original variable. 
         Each column corresponds to a specific category,
          and the presence of a category is marked with a 1, while all other columns are marked with 0. 

**Question:** Why would it be a terrible idea to one-hot encode the data without removing the names?

**Answer:** If we one-hot encode the names columns, then there would be one column for
          each name, and the model would be learn the names of the survivors, and make 
         predictions based on that. This would lead to some serious overfitting!''')

code='''# Removing the names
features_no_name = features_raw.drop(['Name'], axis=1)

# One-hot encoding
features = pd.get_dummies(features_no_name)'''
st.code(code)

# Removing the names
features_no_name = features_raw.drop(['Name'], axis=1)

# One-hot encoding
features = pd.get_dummies(features_no_name)

st.write('''Now, I will fill in any blanks with zeroes.''')

code='''features = features.fillna(0.0)
'''
st.code(code)
features = features.fillna(0.0)


st.write('''## Training the model

It's time to train a model in sklearn. 
         First, I split the data into training and testing sets. Then I will train the model 
         on the training set.''')

code='''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, outcomes, test_size=0.2, random_state=42)'''
st.code(code)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, outcomes, test_size=0.2, random_state=42)

code='''# Import the classifier from sklearn
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

model.fit(X_train, y_train)'''
st.code(code)

# Import the classifier from sklearn
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

model.fit(X_train, y_train)


st.write('''## Testing the model
How did the model perform? I will calculate the accuracy over both the training and the testing set.''')

code='''# Making predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Accuracy calculation
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f' The training accuracy is {train_accuracy:.1%}')
print(f'The testing accuracy is {test_accuracy:.1%}:')'''
st.code(code)

# Making predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Accuracy calculation
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
st.write(f' The training accuracy is {train_accuracy:.1%}')
st.write(f'The testing accuracy is {test_accuracy:.1%}:')


st.write('''## Improving the model''')

st.write('''The model shows very high accuracy on the training data but lower accuracy on the testing data,
          indicating some overfitting. Overfitting occurs when the model learns the training data too well, 
         including its noise and outliers, which makes it perform poorly on new, unseen data. 
         Essentially, the model becomes too specialized to the training data and loses
          its ability to generalize to other datasets.

I will now train a new model, and try to specify some parameters
          in order to improve the testing accuracy, such as:
- `max_depth`:This parameter sets the maximum depth of the tree. Limiting the depth helps prevent
          the model from becoming too complex and overfitting the training data. A shallower
          tree might not capture all the patterns in the data, but it will be better at generalizing to new data.
- `min_samples_leaf`: This parameter specifies the minimum number of samples that a leaf node must have.
         A low value increases the risk of overfitting.
- `min_samples_split`:This parameter determines the minimum number of samples required 
         to split an internal node. If a node has fewer samples than this number, it will not be split
          further. This helps control the growth of the tree and can prevent
          overfitting by ensuring that splits are only made when there is enough data to justify them.

I will try to get 85% accuracy on the testing set.''')

code='''# Training the model
model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=6, min_samples_split=10)
model.fit(X_train, y_train)

# Making predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculating accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print('The training accuracy is', train_accuracy)
print('The test accuracy is', test_accuracy)'''
st.code(code)

# Training the model
model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=6, min_samples_split=10)
model.fit(X_train, y_train)

# Making predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculating accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

st.write(f'The training accuracy is {train_accuracy:.1%}.')
st.write(f'The test accuracy is {test_accuracy:.1%}.')


