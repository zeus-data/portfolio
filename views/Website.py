import streamlit as st

# st.write(f'''#### Work in Progress . . . ''')

# st.image('images/atlantis.png')


st.title('Evaluating the Impact of a New Website on Conversion Rates: An A/B Test Analysis')


st.write('''### Introduction

A/B tests are a fundamental tool for data analysts and data scientists, frequently 
         used to compare two versions of a webpage or product to determine which performs better.
          These tests help in making data-driven decisions by analyzing user interactions and 
         preferences. For this project, I will be delving into the results of an A/B test conducted
          by an e-commerce website. The primary objective is to analyze the
          data to provide insights that will guide the company
          in deciding whether to implement the new webpage, retain the old one, or extend
          the duration of the experiment to gather more conclusive evidence.

The analysis will involve a thorough examination of various metrics and statistical tests
          to understand user behavior and the impact of the changes. By working through this
          notebook, I aim to offer a comprehensive evaluation that considers all possible outcomes
          and their implications. This will enable the company to make an informed decision that
          aligns with their business goals and enhances user experience. The findings from this
          project could potentially lead to significant improvements in the website's performance
          and user satisfaction.
''')

code='''
# import the libraries
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm
np.random.seed(42)
import io

# read the file
df = pd.read_csv('ab_data.csv')
'''
st.code(code)

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import statsmodels.api as sm
np.random.seed(42)
import io

df = pd.read_csv('ab_data.csv')
st.dataframe(df)

# code='''df.query('group =="treatment" & landing_page != "new_page"').count()'''
# st.code(code)
# st.dataframe(df.query('group =="treatment" & landing_page != "new_page"').count())

st.write('''
|Data columns|Purpose|Valid values|
| ------------- |:-------------| -----:|
|user_id|Unique ID|Int64 values|
|timestamp|Time stamp when the user visited the webpage|-|
|group|In the current A/B experiment, the users are categorized into two broad groups. The `control` group users are expected to be served with `old_page`; and `treatment` group users are matched with the `new_page`. However, **some inaccurate rows** are present in the initial data, such as a `control` group user is matched with a `new_page`. |`['control', 'treatment']`|
|landing_page|It denotes whether the user visited the old or new webpage.|`['old_page', 'new_page']`|
|converted|It denotes whether the user decided to pay for the company's product. Here, `1` means yes, the user bought the product.|`[0, 1]`|
''')


st.write(''' In a particular row, the **group** and **landing_page** columns should have
          either of the following acceptable values:

|user_id| timestamp|group|landing_page|converted|
|---|---|---|---|---|
|XXXX|XXXX|`control`| `old_page`|X |
|XXXX|XXXX|`treatment`|`new_page`|X |

However, for the rows where `treatment` does not match with `new_page` or `control`
          does not match with `old_page`, there is no certainty if such rows truly received
          the new or old wepage.''')


code='''# Remove the inaccurate rows, and store the result in a new dataframe df2
df2=df.query('(group == "control" & landing_page == "old_page") | (group=="treatment" &
 landing_page == "new_page")')

df2.head(10) '''
st.code(code)

# Remove the inaccurate rows, and store the result in a new dataframe df2
df1=df.query('(group == "control" & landing_page == "old_page") | (group=="treatment" & landing_page == "new_page")')
# df2 = df.drop(df1.index)
df2=df1.copy()
st.dataframe(df2.head()) 


code='''check1 = df2.query('group=="treatment" & landing_page =="old_page"').user_id.count()
check2 = df2.query('group=="control" & landing_page =="new_page"').user_id.count()'''
st.code(code)

check1 = df2.query('group=="treatment" & landing_page =="new_page"').user_id.count()
check2 = df2.query('group=="control" & landing_page =="old_page"').user_id.count()
check1, check2


st.write('''**a.** How many are the unique `users`?''')

code='''print(f'There are {df2.user_id.nunique():,} unique users: {check1} users in the treatment
 group and {check2} users in the control group.')'''
st.code(code)

st.write(f'''There are {df2.user_id.nunique():,} unique users: {check1:,}  in the treatment
 group and {check2:,}  in the control group.''')

code='''df2.head()'''
st.code(code)
st.dataframe(df2.head())


st.write('''**b.** There is one **user_id** repeated in **df2**.  What is it?''') 

code='''df2.groupby('user_id')['timestamp'].count().reset_index().sort_values(by=['timestamp'], ascending=False).head()'''
st.code(code)


st.dataframe(df2.groupby('user_id')['timestamp'].count().reset_index().sort_values(by=['timestamp'], ascending=False).head())

st.write('''**c.** Display the rows for the duplicate **user_id**? ''')

code='''df2.query('user_id =="773192"')'''
st.code(code)
st.dataframe(df2.query('user_id ==773192'))


st.write('''**d.** Remove **one** of the rows with a duplicate **user_id**, from the **df2** dataframe.''')

code='''# Remove one of the rows with a duplicate user_id
df2.drop([1899], inplace=True)'''
st.code(code)


# Remove one of the rows with a duplicate user_id..
# Hint: The dataframe.drop_duplicates() may not work in this case because the rows with duplicate user_id are not entirely identical. 
df2.drop([1899], inplace=True)

# Check again if the row with a duplicate user_id is deleted or not
df2.query('user_id =="773192"')


#The final table, after cleaning the data  


code='''df2.shape'''
st.code(code)

df2.shape

st.write('''**a.** What is the probability of an individual converting
          regardless of the page they receive?

The probability  I will compute represents the overall "converted" success rate
          in the population and it is $p_{population}$.
''')

code='''df2.converted.mean()'''
st.code(code)
st.write(f'{df2.converted.mean():.4%}')

st.write('''**b.** Given that an individual was in the `control` group,
          what is the probability they converted?''') 

code='''df2.query('group=="control"').converted.mean()'''
st.code(code)
con= df2.query('group=="control"').converted.mean()
st.write(f'{con:.4%}')

st.write('''**c.** Given that an individual was in the `treatment`
          group, what is the probability they converted?''')
st.code(code)

treat= df2.query('group=="treatment"').converted.mean()
st.write(f'{treat:.4%}')


st.write('''Calculate the actual difference (obs_diff) between the conversion rates for the two groups. ''')
code='''
obbs_diff = treat - con
obbs_diff'''
st.code(code)

# Calculate the actual difference (obs_diff) between the conversion rates for the two groups.
obs_diff = treat - con
obs_diff

st.write('''**d.** What is the probability that an individual received the new page?''')

code='''df2.query('landing_page=="new_page"')['user_id'].count()/df2.shape[0]'''
st.code(code)

st.write(f'{df2.query("landing_page=='new_page'")["user_id"].count()/df2.shape[0]:.4%}')

st.write('''**e.** Consider the results and discuss whether there is sufficient evidence to 
         conclude that the new treatment page leads to more conversions.''')

st.write('''Out of a total of 290,584 participants, approximately half were 
         shown the new page (referred to as the `treatment group`), while the other half
          were directed to the old page (referred to as the `control group`).

The conversion rate, which is the percentage of people who made a purchase after 
         landing on the new page, was 11.88%. In comparison, the conversion rate for those
          who used the old page was slightly higher at 12.03%.

This data suggests that there were marginally fewer sales (or conversions) when
          users were directed to the new page compared to the old page.''')





st.write('''## A/B Test

Since a timestamp is associated with each event, I could run a hypothesis test
          continuously as long as I observe the events. 

However, then the hard questions would be: 
- Do I stop as soon as one page is considered significantly better than another
          or does it need to happen consistently for a certain amount of time?  
- How long do I run to render a decision that neither page is better than another?  

These questions are the difficult parts associated with A/B tests in general.  

> I calculated earlier that the "converted" probability
          (or rate) for the old page is *slightly* higher than that of the new page. 

 My null hypothesis **$H_0$** asserts that the difference in conversion rates between the new 
         and old websites is less than or equal to zero, meaning that the new design does not 
         lead to higher conversions. Conversely, the alternative hypothesis (**$H_1$**)
          suggests that the difference is greater than zero, indicating that the new website indeed
          converts more visitors into buyers.         


 Under the null hypothesis, I will assume that $p_{new}$ and $p_{old}$ both have "true" success rates equal
          to the **converted** success rate regardless of page
          - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume
          they are equal to the **converted** rate in **ab_data.csv** regardless of the page.         



**$p_{old}$:** conversion rate , when user accesses the *old* page
         
**$p_{new}$:** conversion rate, when user accesses *new* page 
         
**$H_0$**: Null Hypothesis:  **$p_{new}$** - **$p_{old}$** <= 0  
**$H_1$**: Alternative Hypothesis:  **$p_{new}$** - **$p_{old}$** > 0
''')

st.write('''#### Null Hypothesis $H_0$ Testing
I will: 

- Simulate (bootstrap) sample data set for both groups, and compute the  "converted" probability $p$ for those samples. 

- Use a sample size for each group equal to the ones in the `df2` data.

- Compute the difference in the "converted" probability for the two samples above. 

- Perform the sampling distribution for the "difference in the converted probability"
          between the two simulated-samples over 10,000 iterations; and calculate an estimate. 
''')

st.write('''**a.** What is the **conversion rate** for $p_{old}$ under the null hypothesis? ''')

code='''p_old = len(df2.query( 'converted==1'))/len(df2.index)'''
st.code(code)

p_old = len(df2.query( 'converted==1'))/len(df2.index)
st.write(f'{p_old:.4%}')





st.write('''**a.** What is the **conversion rate** for $p_{new}$ under the null hypothesis? ''')

code='''p_new = len(df2.query( 'converted==1'))/len(df2.index)'''
st.code(code)

p_new = len(df2.query( 'converted==1'))/len(df2.index)  
st.write(f'{p_new:.4%}')


st.write('''**c.** What is $n_{new}$, the number of individuals in the treatment group?''')

code='''n_new = df2.query('group=="treatment"').user_id.count()'''
st.code(code)

n_new = df2.query('group=="treatment"').user_id.count()
st.write(f'{n_new:,}')

st.write('''**d.** What is $n_{old}$, the number of individuals in the control group?''')

code='''n_old = len(df2.query('group != "treatment"'))'''
st.code(code)

n_old = len(df2.query('group != "treatment"'))
st.write(f'{n_old:,}')

st.write('''**e. Simulate Sample for the `treatment` Group**
Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null
          hypothesis.
''')



code='''# Simulate a Sample for the treatment Group
new_page_converted = np.random.choice([1,0], n_new,p=[p_new, 1-p_new])'''
st.code(code)

# Simulate a Sample for the control Group
new_page_converted = np.random.choice([1,0], n_new,p=[p_new, 1-p_new])


st.write('''**f. Simulate Sample for the `control` Group**
Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null hypothesis.
          Store these $n_{old}$ 1's and 0's in the `old_page_converted` numpy array.''')

code='''# Simulate a Sample for the control Group
old_page_converted = np.random.choice([1,0], n_old,p=[p_old, 1-p_old])'''
st.code(code)

# Simulate a Sample for the control Group
old_page_converted = np.random.choice([1,0], n_old,p=[p_old, 1-p_old])


st.write('''**g.** Find the difference in the "converted" probability
          $(p{'}_{new}$ - $p{'}_{old})$  ''')

code='''obs_diff_v2= new_page_converted.mean() - old_page_converted.mean()
obs_diff_v2'''
st.code(code)

obs_diff_v2= new_page_converted.mean() - old_page_converted.mean()
st.write(obs_diff_v2)  


st.write('''
**h. Sampling distribution**
Re-create `new_page_converted` and `old_page_converted` and find the
          $(p{'}_{new}$ - $p{'}_{old})$ value 10,000 times using the same simulation
          process you used in parts (a) through (g) above. 
I will store all  $(p{'}_{new}$ - $p{'}_{old})$  values in a NumPy array called `p_diffs`.''')

code='''# Sampling distribution 
p_diffs = []
for s in range(10000):
    new_page_converted =  np.random.choice([0,1], n_new, p = [p_new, 1-p_new])
    old_page_converted =  np.random.choice([0,1], n_old, p = [p_old, 1-p_old])
    obs_diff_v3 = new_page_converted.mean() - old_page_converted.mean()
    p_diffs.append(obs_diff_v3)
    '''
st.code(code)


# Sampling distribution 
p_diffs = []
for s in range(100):
    new_page_converted =  np.random.choice([0,1], n_new, p = [p_new, 1-p_new])
    old_page_converted =  np.random.choice([0,1], n_old, p = [p_old, 1-p_old])
    obs_diff_v3 = new_page_converted.mean() - old_page_converted.mean()
    p_diffs.append(obs_diff_v3)
    

st.write('''**i. Histogram**
''')

code='''plt.hist(p_diffs)
plt.title('Graph of the converted probability difference p_new - p_old')
plt.xlabel('Difference')
plt.ylabel('Frequency')
plt.axvline(x = obs_diff, color = 'red', linewidth = 2);'''
st.code(code)


fig, ax = plt.subplots()

ax.hist(p_diffs)
ax.set_title('Graph of the converted probability difference p_new - p_old')
ax.set_xlabel('Difference')
ax.set_ylabel('Frequency')
ax.axvline(x = obs_diff, color = 'red', linewidth = 2)
st.pyplot(fig)


st.write('''**j.** What proportion of the **p_diffs** are greater
          than the actual difference observed in the `df2` data?''')

code='''(p_diffs > obs_diff).mean()'''
st.code(code)

st.write((p_diffs > obs_diff).mean())
 


st.write('''The value calculated is the p-value. This p-value is greater than 90%.
          It is also significantly higher than the Type I error threshold of 5%. 
         This high p-value suggests that we should not reject the null hypothesis.
          In statistical terms, we do not have enough evidence to move away from the null
          hypothesis. The null hypothesis posits that fewer people will complete a purchase
          after viewing the new page. Therefore, we should adhere to what the null hypothesis
          states. The data indicates that the new page is less effective in converting visitors
          into buyers. This conclusion is based on the high p-value. It implies that the observed
          difference in conversion rates is not statistically significant. Consequently,
          we should not implement the new page.
          Instead, we should continue using the old page to maintain higher conversion rates.''')

# st.write('''

# **l. Using Built-in Methods for Hypothesis Testing**
         
# We could also use a built-in to achieve similar results.
#            Though using the built-in might be easier to code, the above 
#          portions are a walkthrough of the ideas that are critical to correctly
#           thinking about statistical significance. 

# Fill in the statements below to calculate the:
# - `convert_old`: number of conversions with the old_page
# - `convert_new`: number of conversions with the new_page
# - `n_old`: number of individuals who were shown the old_page
# - `n_new`: number of individuals who were shown the new_page
# ''')

# code='''import statsmodels.api as sm

# # number of conversions with the old_page
# convert_old = df2.query('landing_page == "old_page" & converted==1').user_id.count()

# # number of conversions with the new_page
# convert_new = df2.query('landing_page != "old_page" & converted==1').user_id.count()

# # number of individuals who were shown the old_page
# n_old = df2.query('landing_page == "old_page"').user_id.count()

# # number of individuals who received new_page
# n_new = df2.query('landing_page != "old_page"').user_id.count()

# convert_old, convert_new, n_old, n_new'''
# st.code(code)




# # number of conversions with the old_page
# convert_old = df2.query('landing_page == "old_page" & converted==1').user_id.count()

# # number of conversions with the new_page
# convert_new = df2.query('landing_page != "old_page" & converted==1').user_id.count()

# # number of individuals who were shown the old_page
# n_old = df2.query('landing_page == "old_page"').user_id.count()

# # number of individuals who received new_page
# n_new = df2.query('landing_page != "old_page"').user_id.count()

# convert_old, convert_new, n_old, n_new


# st.write('''**m.** Now use `sm.stats.proportions_ztest()` to compute your test statistic and p-value.  [Here](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html) is a helpful link on using the built in.

# The syntax is: 
# ```bash
# proportions_ztest(count_array, nobs_array, alternative='larger')
# ```
# where, 
# - `count_array` = represents the number of "converted" for each group
# - `nobs_array` = represents the total number of observations (rows) in each group
# - `alternative` = choose one of the values from `[‘two-sided’, ‘smaller’, ‘larger’]` depending upon two-tailed, left-tailed, or right-tailed respectively. 
# >**Hint**: <br>
# It's a two-tailed if you defined $H_1$ as $(p_{new} = p_{old})$. <br>
# It's a left-tailed if you defined $H_1$ as $(p_{new} < p_{old})$. <br>
# It's a right-tailed if you defined $H_1$ as $(p_{new} > p_{old})$. 

# The built-in function above will return the z_score, p_value. 

# ---
# ### About the two-sample z-test
# Recall that you have plotted a distribution `p_diffs` representing the
# difference in the "converted" probability  $(p{'}_{new}-p{'}_{old})$  for your two simulated samples 10,000 times. 

# Another way for comparing the mean of two independent and normal distribution is a **two-sample z-test**. You can perform the Z-test to calculate the Z_score, as shown in the equation below:

# $$
# Z_{score} = \frac{ (p{'}_{new}-p{'}_{old}) - (p_{new}  -  p_{old})}{ \sqrt{ \frac{\sigma^{2}_{new} }{n_{new}} + \frac{\sigma^{2}_{old} }{n_{old}}  } }
# $$

# where,
# - $p{'}$ is the "converted" success rate in the sample
# - $p_{new}$ and $p_{old}$ are the "converted" success rate for the two groups in the population. 
# - $\sigma_{new}$ and $\sigma_{new}$ are the standard deviation for the two groups in the population. 
# - $n_{new}$ and $n_{old}$ represent the size of the two groups or samples (it's same in our case)


# >Z-test is performed when the sample size is large, and the population variance is known. The z-score represents the distance between the two "converted" success rates in terms of the standard error. 

# Next step is to make a decision to reject or fail to reject the null hypothesis based on comparing these two values: 
# - $Z_{score}$
# - $Z_{\alpha}$ or $Z_{0.05}$, also known as critical value at 95% confidence interval.  $Z_{0.05}$ is 1.645 for one-tailed tests,  and 1.960 for two-tailed test. You can determine the $Z_{\alpha}$ from the z-table manually. 

# Decide if your hypothesis is either a two-tailed, left-tailed, or right-tailed test. Accordingly, reject OR fail to reject the  null based on the comparison between $Z_{score}$ and $Z_{\alpha}$. We determine whether or not the $Z_{score}$ lies in the "rejection region" in the distribution. In other words, a "rejection region" is an interval where the null hypothesis is rejected iff the $Z_{score}$ lies in that region.

# >Hint:<br>
# For a right-tailed test, reject null if $Z_{score}$ > $Z_{\alpha}$. <br>
# For a left-tailed test, reject null if $Z_{score}$ < $Z_{\alpha}$. 




# Reference: 
# - Example 9.1.2 on this [page](https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Book%3A_Introductory_Statistics_(Shafer_and_Zhang)/09%3A_Two-Sample_Problems/9.01%3A_Comparison_of_Two_Population_Means-_Large_Independent_Samples), courtesy www.stats.libretexts.org

# ---

# >**Tip**: You don't have to dive deeper into z-test for this exercise. **Try having an overview of what does z-score signify in general.** ''')


# code='''import statsmodels.api as sm
# # sm.stats.proportions_ztest() method 
# z_score, p_value = sm.stats.proportions_ztest([convert_new, convert_old], [n_new, n_old],alternative='larger') 
# z_score, p_value'''
# st.code(code)

# import statsmodels.api as sm
# # ToDo: Complete the sm.stats.proportions_ztest() method arguments
# z_score, p_value = sm.stats.proportions_ztest([convert_new, convert_old], [n_new, n_old],alternative='larger') 
# z_score, p_value


# st.write('''
# $Z_{score}$ is calculated at -1.31 and the  $Z_{a}$=$Z_{0.05}$ is 1.645. This is a right-tailed test, and because $Z_{score}$  < $Z_{a}$, we fail to reject the null.<br>
# The p-value is similar to the one we calculated at **j.** , which says that we have 90% probability of observing the difference given that the null is true.Thus, I fail to reject the null hypothesis.<br><br>
# Both the $Z_{score}$ and **p-value** confirm that we fail to reject the null.

# ''')


st.write('''
#### A regression approach


The result that was achieved
          in the A/B test above can also be achieved by performing logistic regression.
             ''')


code='''df2['intercept'] = 1
df2['ab_page'] = pd.get_dummies(df2['group'])['treatment']
df2.head(15)'''
st.code(code)


df2['intercept'] = 1
df2['ab_page'] = pd.get_dummies(df2['group'])['treatment'].astype('int')
st.dataframe(df2.head(7))


code='''import scipy.stats as stats
lreg_model = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
program_ = lreg_model.fit()
program_.summary2()'''
st.code(code)


# buffer = io.StringIO()
# df2.info(buf=buffer)
# s = buffer.getvalue()

# st.text(s)
import scipy.stats as stats
lreg_model = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
program_ = lreg_model.fit()



# Get the summary as a string
summary_str = program_.summary2().as_text()

# Display the summary in Streamlit
st.text(summary_str)



st.write('''What is the p-value associated with **ab_page**? 
         Why does it differ from the A/B test value ?''')

st.write('''

The `ab_page`'s **p-value** is 0.189. Our Type I error is 5%; **p-value** > 
         Type I error, so we fail to reject the null.
In **A/B test**, the hypothesis was a one-tailed test; the null hypothesis was
          the new page has fewer or equal conversions,and the alternative was
          the new page would result in more convesrions.
In the regression approach, the null hypothesis is that the new page would not
          change the conversions, and the alternative is that the coversion rates
          are different betwen the old and new page. 
Thus, the two hypotheses are different , and thus the **p-values** are different as well.
''')


st.write('''**Adding countries: **
Now along with testing if the conversion rate changes for different pages,
          I will also add an effect based on which country a user lives in. 
         To enhance the robustness of my findings, 
         I will incorporate additional data from a file named countries.csv and rerun the 
         regression analysis. This additional data will provide a broader context
          and help me assess whether the initial results held true under different conditions,
          ultimately guiding the company towards a more informed decision.''')

code='''# Read the countries.csv
df_countries = pd.read_csv('countries.csv')
df_countries.head()'''
st.code(code)


# Read the countries.csv
df_countries = pd.read_csv('countries.csv')
st.dataframe(df_countries.head())

code='''# Join with the df2 dataframe
df_merged = df_countries.set_index('user_id').join(df2.set_index('user_id'), how='inner')
df_merged.head()'''
st.code(code)


# Join with the df2 dataframe
df_merged = df_countries.set_index('user_id').join(df2.set_index('user_id'), how='inner')
st.dataframe(df_merged.head())

code='''# Create the necessary dummy variables
new_df = pd.get_dummies(df_merged['country']).join(df_merged, how='inner')
new_df.head()'''
st.code(code)


# Create the necessary dummy variables

new_df = pd.get_dummies(df_merged['country']).astype('int').join(df_merged, how='inner')
st.dataframe(new_df.head())

code='''log_1 = sm.Logit(new_df['converted'], new_df[['intercept', 'UK', 'CA']])
result_1 = log_1.fit()
result_1.summary2()'''
st.code(code)

log_1 = sm.Logit(new_df['converted'], new_df[['intercept', 'UK', 'CA']])
result_1 = log_1.fit()
summary_html = result_1.summary2().as_html()

# Display the summary in Streamlit
st.markdown(summary_html, unsafe_allow_html=True)

 
# summary_str = result_1.summary2().as_text()

# # Display the summary in Streamlit
# st.text(summary_str)


code='''log_2 = sm.Logit(new_df['converted'], new_df[['intercept', 'ab_page', 'UK', 'CA']])
result_2 = log_2.fit()
result_2.summary2()'''
st.code(code)

log_2 = sm.Logit(new_df['converted'], new_df[['intercept', 'ab_page', 'UK', 'CA']])
result_2 = log_2.fit()
result_2.summary2()


summary_str = result_2.summary2().as_text()

# Display the summary in Streamlit
st.text(summary_str)

st.write('''The statistical results show high p-values, specifically greater than 0.05. 
         This indicates that the country parameter does not significantly impact
          the conversion rate. In other words, the country variable is not 
         a crucial factor in determining conversions.
          The data suggests that other variables might be more influential.''')


st.write('''
Though I have examined the individual effects of country and page on conversion,
          I now want to explore their interaction. Specifically, 
         I aim to see if there are significant effects on conversion when 
         considering both factors together. This approach will help determine
          if the combination of page and country influences conversion rates.
          By analyzing this interaction, I can uncover any potential synergistic effects.
''')

code='''# Fit your model, and summarize the results
new_df['UK_abpage'] = new_df['ab_page'] * new_df['UK']
new_df['CA_abpage'] = new_df['ab_page'] * new_df['CA']
new_df.head()
'''
st.code(code)

# Fit your model, and summarize the results
new_df['UK_abpage'] = new_df['ab_page'] * new_df['UK']
new_df['CA_abpage'] = new_df['ab_page'] * new_df['CA']
st.dataframe(new_df.head())

code='''log_3 = sm.Logit(new_df['converted'], new_df[['intercept', 'UK', 'CA', 'UK_abpage', 'CA_abpage']])
result3 = log_3.fit()
result3.summary2()'''
st.code(code)


log_3 = sm.Logit(new_df['converted'], new_df[['intercept', 'UK', 'CA', 'UK_abpage', 'CA_abpage']])
result3 = log_3.fit()
summary_str=result3.summary2().as_text()

# Display the summary in Streamlit
st.text(summary_str)

code='''log_4 = sm.Logit(new_df['converted'], new_df[['intercept','ab_page', 'UK', 'CA', 'UK_abpage', 'CA_abpage']])
result4 = log_4.fit()
result4.summary2()'''
st.code(code)


log_4 = sm.Logit(new_df['converted'], new_df[['intercept','ab_page', 'UK', 'CA', 'UK_abpage', 'CA_abpage']])
result4 = log_4.fit()
summary_str = result4.summary2().as_text()
st.text(summary_str)

code='''print(np.exp(0.0314)), print(np.exp(-0.0469));'''
st.code(code)

st.write(np.exp(0.0314)), st.write(np.exp(-0.0469))

st.write('''The analysis shows that the interaction between country and page was statistically 
         insignificant as well. This means that neither
          the country nor the page, when combined, had a meaningful impact on conversion rates. For example,
          when holding all other variables constant, a UK user who receives
          the new page is only 1.03 times more likely to convert, and a Canadian user is
          0.95 times more likely to convert.
          Both of these conversion rates are practically 
         insignificant.
          In conclusion, the analysis revealed that all p-values were above the Type I 
         error rate of α=0.05. This means we fail to reject the null hypothesis, 
         Consequently, the new page should not be implemented based on these findings.
        ''')