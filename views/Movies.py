import streamlit as st

# st.title('# k-means Clustering of Movie Ratings')

# st.write('''Movie time!''')

# st.write('https://learn.udacity.com/nanodegrees/nd229-ent/parts/cbac748e-de4c-46ed-9a8b-9135f183913d/lessons/16900da4-4b91-470e-bd5b-d255125f850b/concepts/ec9a131c-5bf5-424c-b84a-aebed559059b?lesson_tab=lesson')


# st.write('''#### Work in Progress . . . ''')

# st.image('images/atlantis.png')


st.write('''# Movie Ratings at Netflix

As a Data Analyst at Netflix, I would be interested in examining the 
         similarities and differences in people's movie preferences based
          on their ratings. Understanding these ratings could help enhance
          a movie recommendation system for users. By the end of this project, 
         I aim to have developed a recommendation system.
## Dataset overview
         
The data I will be using comes from the [MovieLens user rating dataset](https://grouplens.org/datasets/movielens/).
The dataset has two files. I will import them both into Pandas Dataframes:''')

code='''import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import silhouette_samples, silhouette_score
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.sparse as sp
import datetime'''
st.code(code)


code='''
# Import the Movies dataset
movies = pd.read_csv('ml-latest-small/movies.csv')
movies.head()'''
st.code(code)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import silhouette_samples, silhouette_score
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.sparse as sp
import datetime


# Import the Movies dataset
movies = pd.read_csv('ml-latest-small/movies.csv')
# 
st.dataframe(movies.head())

code='''ratings = pd.read_csv('ml-latest-small/ratings.csv')
ratings.head()'''
st.code(code)

ratings = pd.read_csv('ml-latest-small/ratings.csv')
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings['timestamp'] = ratings['timestamp'].dt.strftime('%Y-%m-%d')
st.dataframe(ratings.head())



code='''print(f'The ratings table contains: {len(ratings):,} `ratings` of {len(movies):,} `movies`.')'''
st.code(code)
st.write(f'The ratings table contains {len(ratings):,} `ratings` of {len(movies):,} `movies`.')


st.write('''
## Romance vs. Scifi
I will start by taking a subset of users, and seeing what their preferred genres are.''')


code='''def get_genre_ratings(ratings, movies, genres, column_names):
    genre_ratings = pd.DataFrame()
    for genre in genres:        
        genre_movies = movies[movies['genres'].str.contains(genre) ]
        avg_genre_votes_per_user = ratings[ratings['movieId'].isin(genre_movies['movieId'])].loc[:, ['userId', 'rating']].groupby(['userId'])['rating'].mean().round(2)
        
        genre_ratings = pd.concat([genre_ratings, avg_genre_votes_per_user], axis=1)
        
    genre_ratings.columns = column_names
    return genre_ratings'''
st.code(code)

code='''  
# Calculate the average rating of romance and scifi movies
genre_ratings=get_genre_ratings(ratings, movies, ['Romance', 'Sci-Fi'], ['avg_romance_rating', 'avg_scifi_rating'])
genre_ratings.head()'''
st.code(code)

def get_genre_ratings(ratings, movies, genres, column_names):
    genre_ratings = pd.DataFrame()
    for genre in genres:        
        genre_movies = movies[movies['genres'].str.contains(genre) ]
        avg_genre_votes_per_user = ratings[ratings['movieId'].isin(genre_movies['movieId'])].loc[:, ['userId', 'rating']].groupby(['userId'])['rating'].mean().round(2)
        
        genre_ratings = pd.concat([genre_ratings, avg_genre_votes_per_user], axis=1)
        
    genre_ratings.columns = column_names
    return genre_ratings

# Calculate the average rating of romance and scifi movies
genre_ratings=get_genre_ratings(ratings, movies, ['Romance', 'Sci-Fi'], ['avg_romance_rating', 'avg_scifi_rating'])
st.write( "Number of records: ", len(genre_ratings))
st.dataframe(genre_ratings.head())
    

st.write('''The function `get_genre_ratings` calculated each user's 
         average rating of all romance movies and all scifi movies. 
         I will now bias the dataset a little by removing people who like both scifi and romance,
          just so that the clusters tend to define them as liking one genre more than the other.''')

code='''    
def bias_genre_rating_dataset(genre_ratings, score_limit_1, score_limit_2):
    biased_dataset = genre_ratings[((genre_ratings['avg_romance_rating'] < score_limit_1 - 0.2) & (genre_ratings['avg_scifi_rating'] > score_limit_2)) | ((genre_ratings['avg_scifi_rating'] < score_limit_1) & (genre_ratings['avg_romance_rating'] > score_limit_2))]
    biased_dataset = pd.concat([biased_dataset[:300], genre_ratings[:2]])
    biased_dataset = pd.DataFrame(biased_dataset.to_records())
    return biased_dataset'''
st.code(code)


code='''biased_dataset = bias_genre_rating_dataset(genre_ratings, 3.2, 2.5)

print( "Number of records: ", len(biased_dataset))
biased_dataset.head()'''
st.code(code)


    
def bias_genre_rating_dataset(genre_ratings, score_limit_1, score_limit_2):
    biased_dataset = genre_ratings[((genre_ratings['avg_romance_rating'] < score_limit_1 - 0.2) & (genre_ratings['avg_scifi_rating'] > score_limit_2)) | ((genre_ratings['avg_scifi_rating'] < score_limit_1) & (genre_ratings['avg_romance_rating'] > score_limit_2))]
    biased_dataset = pd.concat([biased_dataset[:300], genre_ratings[:2]])
    biased_dataset = pd.DataFrame(biased_dataset.to_records())
    return biased_dataset


biased_dataset = bias_genre_rating_dataset(genre_ratings, 3.2, 2.5)

st.write( "Number of records: ", len(biased_dataset))
st.dataframe(biased_dataset.head())


st.write('''There are 183 users, and for each user, 
         we have their average ratings for the romance and sci-fi movies they've watched.
I will plot this dataset:''')

code='''def draw_scatterplot(x_data, x_label, y_data, y_label):
    fig,ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.scatter(x_data, y_data, s=30)
    return fig, ax'''
st.code(code)
code='''draw_scatterplot(biased_dataset['avg_scifi_rating'],
'Avg scifi rating', biased_dataset['avg_romance_rating'], 'Avg romance rating')'''
st.code(code)


def draw_scatterplot(x_data, x_label, y_data, y_label):
    fig,ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.scatter(x_data, y_data, s=30)
    return fig, ax

fig, ax=draw_scatterplot(biased_dataset['avg_scifi_rating'],
                     'Avg scifi rating', biased_dataset['avg_romance_rating'], 'Avg romance rating')
st.pyplot(fig)

st.write('''I see come clear bias in this sample. 
         I will now break the sample down into two groups using `k-means` algorithm''')





code='''# Turn the dataset into a list
X = biased_dataset[['avg_scifi_rating','avg_romance_rating']].values'''
st.code(code)

# Let's turn our dataset into a list
X = biased_dataset[['avg_scifi_rating','avg_romance_rating']].values

code='''# Import KMeans
from sklearn.cluster import KMeans 

# This is an instance of KMeans to find two clusters
kmeans_1 = KMeans(n_clusters=2)

# Fit_predict to cluster the dataset
predictions = kmeans_1.fit_predict(X) 

# Plot
def draw_clusters(biased_dataset, predictions, cmap='viridis'):
    fig,ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel('Avg scifi rating')
    ax.set_ylabel('Avg romance rating')
    

    clustered = pd.concat([biased_dataset.reset_index(), pd.DataFrame({'group':predictions})], axis=1)
    ax.scatter(clustered['avg_scifi_rating'], clustered['avg_romance_rating'], c=clustered['group'], s=20, cmap=cmap)
    return fig, ax
draw_clusters(biased_dataset, predictions)'''
st.code(code)


# TODO: Import KMeans
from sklearn.cluster import KMeans 

# TODO: Create an instance of KMeans to find two clusters
kmeans_1 = KMeans(n_clusters=2)

# TODO: use fit_predict to cluster the dataset
predictions = kmeans_1.fit_predict(X) 

# Plot
def draw_clusters(biased_dataset, predictions, cmap='viridis'):
    fig,ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel('Avg scifi rating')
    ax.set_ylabel('Avg romance rating')


    clustered = pd.concat([biased_dataset.reset_index(), pd.DataFrame({'group':predictions})], axis=1)
    ax.scatter(clustered['avg_scifi_rating'], clustered['avg_romance_rating'], c=clustered['group'], s=20, cmap=cmap)
    return fig, ax

fig, ax = draw_clusters(biased_dataset, predictions)
st.pyplot(fig)

st.write('''I can see that the groups are mostly based on how each person
          rated romance movies. 
         If their average rating of romance movies is over 3 stars,
          then they belong to one group. Otherwise, they belong to the other group.

Now, I will break them down into three groups:''')

code='''# An instance of KMeans to find three clusters
kmeans_2 = KMeans(n_clusters=3)

# Fit_predict to cluster the dataset
predictions_2 = kmeans_2.fit_predict(X)

# Plot
fig, ax =draw_clusters(biased_dataset, predictions_2)'''
st.code(code)


kmeans_2 = KMeans(n_clusters=3)

predictions_2 = kmeans_2.fit_predict(X)

# Plot
fig, ax =draw_clusters(biased_dataset, predictions_2)
st.pyplot(fig)


st.write('''Now the average scifi rating is starting to come into play. The groups are:
 * people who like romance but not scifi
 * people who like scifi but not romance
 * people who like both scifi and romance
 
I will add one more group:''')

code='''# An instance of KMeans to find four clusters
kmeans_3 = KMeans(n_clusters=4)

# Fit_predict to cluster the dataset
predictions_3 = kmeans_3.fit_predict(X)

# Plot
fig, ax =draw_clusters(biased_dataset, predictions_3)'''
st.code(code)


# Create an instance of KMeans to find three clusters
kmeans_3 = KMeans(n_clusters=4)

# Use fit_predict to cluster the dataset
predictions_3 = kmeans_3.fit_predict(X)

# Plot
fig, ax =draw_clusters(biased_dataset, predictions_3)
st.pyplot(fig)



st.write('''As I increase the number of clusters in the dataset, the individuals within each cluster
          tend to have more similar preferences. This means that users within the same cluster exhibit 
         more homogeneous tastes in sci-fi and romance movies. Consequently, the clustering becomes more refined,
          allowing for better-targeted recommendations. However,it's important to balance
          the number of clusters to avoid overfitting and ensure meaningful groupings.

## Choosing K
In summary, it's possible to group the data points into any number of clusters. 
         How do we determine the optimal number of clusters for this dataset?

There are [several](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set) ways of
          choosing the number of clusters, k. I will look at a simple one called "the elbow method".
          The elbow method works by plotting the ascending values of k versus the total error calculated
          using that k. 

How can I determine the total error?

One approach is to calculate the squared error. For instance, with k=2, the data
          is divided into two clusters, each having one "centroid" point. For every point in the dataset,
          subtract its coordinates from its cluster centroid's coordinates. Then square the result
          to eliminate negative values and sum these squared differences. This sum gives an error
          value for each point. Adding up these error values provides the total error for all points when k=2.

The aim is to repeat this process for each k (ranging from 1 to, for example, the number
          of elements in our dataset).''')

code='''

# I implemented a stride of 5 to enhance performance,
# allowing us to skip calculating the error for every single k value
possible_k_values = range(2, len(X)+1, 5)

def clustering_errors(k, data):
    kmeans = KMeans(n_clusters=k).fit(data)
    predictions = kmeans.predict(data)
    silhouette_avg = silhouette_score(data, predictions)
    return silhouette_avg

# Error values for the selected subset of k values
errors_per_k = [clustering_errors(k, X) for k in possible_k_values]
'''
st.code(code)


def clustering_errors(k, data):
    kmeans = KMeans(n_clusters=k).fit(data)
    predictions = kmeans.predict(data)
    #cluster_centers = kmeans.cluster_centers_
    # errors = [mean_squared_error(row, cluster_centers[cluster]) for row, cluster in zip(data.values, predictions)]
    # return sum(errors)
    silhouette_avg = silhouette_score(data, predictions)
    return silhouette_avg





# Choose the range of k values to test.
# We added a stride of 5 to improve performance. We don't need to calculate the error for every k value
possible_k_values = range(2, len(X)+1, 5)

# Calculate error values for all k values we're interested in
errors_per_k = [clustering_errors(k, X) for k in possible_k_values]


# Optional: Look at the values of K vs the silhouette score of running K-means with that value of k

# for item in list(zip(possible_k_values, errors_per_k)):
#     st.write(item)


code='''# Plot each value of K vs. the silhouette score at that value
fig, ax = plt.subplots(figsize=(16, 6))
plt.plot(possible_k_values, errors_per_k)

# Ticks and grid
xticks = np.arange(min(possible_k_values), max(possible_k_values)+1, 5.0)
ax.set_xticks(xticks, minor=False)
ax.set_xticks(xticks, minor=True)
ax.xaxis.grid(True, which='both')
yticks = np.arange(round(min(errors_per_k), 2), max(errors_per_k), .05)
ax.set_yticks(yticks, minor=False)
ax.set_yticks(yticks, minor=True)
ax.yaxis.grid(True, which='both')'''
st.code(code)


# Plot each value of K vs. the silhouette score at that value
fig, ax = plt.subplots(figsize=(16, 6))
plt.plot(possible_k_values, errors_per_k)

# Ticks and grid
xticks = np.arange(min(possible_k_values), max(possible_k_values)+1, 5.0)
ax.set_xticks(xticks, minor=False)
ax.set_xticks(xticks, minor=True)
ax.xaxis.grid(True, which='both')
yticks = np.arange(round(min(errors_per_k), 2), max(errors_per_k), .05)
ax.set_yticks(yticks, minor=False)
ax.set_yticks(yticks, minor=True)
ax.yaxis.grid(True, which='both')

st.pyplot(fig)


st.write('''Analyzing the graph, optimal choices for the number of clusters (k) include 7, 17, 32, and 57,
          among other values, with slight variations between different runs. Increasing the number of clusters
          beyond this range tends to produce less effective clusters, as indicated by the Silhouette score.
          I would choose k=7 because it provides a balance between cluster quality and ease of visualization.
          This choice allows for a clear and interpretable grouping of users based on their movie preferences.
          Additionally, having fewer clusters
          simplifies the analysis and makes it easier to derive actionable insights.:''')

code='''# An instance of KMeans to find seven clusters
kmeans_4 = KMeans(n_clusters=7)

# Fit_predict to cluster the dataset
predictions_4 = kmeans_4.fit_predict(X)

# Plot
draw_clusters(biased_dataset, predictions_4, cmap='Accent')'''
st.code(code)


# Create an instance of KMeans to find seven clusters
kmeans_4 = KMeans(n_clusters=7)

# Use fit_predict to cluster the dataset
predictions_4 = kmeans_4.fit_predict(X)

# Plot
fig, ax = draw_clusters(biased_dataset, predictions_4, cmap='Accent')
st.pyplot(fig)


st.write('''
### Adding Action genre into the analysis
Until now, I've focused on how users rated romance
          and sci-fi movies. I will now include another genre: Action.

The dataset now looks like this:''')

code='''biased_dataset_3_genres = get_genre_ratings(ratings, movies, 
                                                     ['Romance', 'Sci-Fi', 'Action'], 
                                                     ['avg_romance_rating', 'avg_scifi_rating', 'avg_action_rating'])
biased_dataset_3_genres = bias_genre_rating_dataset(biased_dataset_3_genres, 3.2, 2.5).dropna()

print( "Number of records: ", len(biased_dataset_3_genres))
biased_dataset_3_genres.head()'''
st.code(code)


biased_dataset_3_genres = get_genre_ratings(ratings, movies, 
                                                     ['Romance', 'Sci-Fi', 'Action'], 
                                                     ['avg_romance_rating', 'avg_scifi_rating', 'avg_action_rating'])
biased_dataset_3_genres = bias_genre_rating_dataset(biased_dataset_3_genres, 3.2, 2.5).dropna()

st.write( "Number of records: ", len(biased_dataset_3_genres))
st.dataframe(biased_dataset_3_genres.head())

code='''X_with_action = biased_dataset_3_genres[['avg_scifi_rating',
                                                           'avg_romance_rating', 
                                                           'avg_action_rating']].values'''
st.code(code)


X_with_action = biased_dataset_3_genres[['avg_scifi_rating',
                                                           'avg_romance_rating', 
                                                           'avg_action_rating']].values


code='''def draw_clusters_3d(biased_dataset_3, predictions):
    fig, ax = plt.subplots(figsize=(8,8))

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel('Avg scifi rating')
    ax.set_ylabel('Avg romance rating')

    clustered = pd.concat([biased_dataset_3.reset_index(), pd.DataFrame({'group':predictions})], axis=1)

    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for g in clustered.group.unique():
        color = next(colors)
        for index, point in clustered[clustered.group == g].iterrows():
            if point['avg_action_rating'].astype(float) > 3: 
                size = 50
            else:
                size = 15
            plt.scatter(point['avg_scifi_rating'], 
                        point['avg_romance_rating'], 
                        s=size, 
                        color=color)
    return fig, ax'''
st.code(code)


code='''# An instance of KMeans to find seven clusters
kmeans_5 = KMeans(n_clusters=7)

# Fit_predict to cluster the dataset
predictions_5 = kmeans_5.fit_predict(X_with_action)

# plot
draw_clusters_3d(biased_dataset_3_genres, predictions_5)'''
st.code(code)

def draw_clusters_3d(biased_dataset_3, predictions):
    fig, ax = plt.subplots(figsize=(8,8))

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel('Avg scifi rating')
    ax.set_ylabel('Avg romance rating')

    clustered = pd.concat([biased_dataset_3.reset_index(), pd.DataFrame({'group':predictions})], axis=1)

    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for g in clustered.group.unique():
        color = next(colors)
        for index, point in clustered[clustered.group == g].iterrows():
            if point['avg_action_rating'].astype(float) > 3: 
                size = 50
            else:
                size = 15
            plt.scatter(point['avg_scifi_rating'], 
                        point['avg_romance_rating'], 
                        s=size, 
                        color=color)
    return fig, ax
            
# Create an instance of KMeans to find seven clusters
kmeans_5 = KMeans(n_clusters=7)

# Use fit_predict to cluster the dataset
predictions_5 = kmeans_5.fit_predict(X_with_action)

# plot
fig, ax = draw_clusters_3d(biased_dataset_3_genres, predictions_5)
st.pyplot(fig)


st.write('''The x and y axes in the graph represent the sci-fi and romance ratings, respectively,
          while the size of each dot indicates the 'action' rating. Larger dots correspond to average
          action ratings over 3, and smaller dots represent lower ratings. Including the action genre
          in the analysis influences how users are grouped. As we add more data to the k-means algorithm,
          the preferences within each cluster become more consistent. However, visualizing clusters in more
          than two or three dimensions becomes challenging with this method. This limitation makes it difficult
          to interpret the data effectively. 
In the next section, I will explore an alternative type of plot that
          can handle up to fifty dimensions. This approach will
          allow to better visualize and understand the complex relationships between multiple genres.

### Movie-level Clustering
With a clearer understanding of how `k-means` clusters users based on their genre preferences,
          let's dive into how users rated individual movies.
          To achieve this, I'll restructure the dataset to display `userId` against `user rating` for each movie.''')

code='''# The two tables will merge then pivot so there is Users X Movies dataframe
ratings_title = pd.merge(ratings, movies[['movieId', 'title']], on='movieId' )
user_movie_ratings = pd.pivot_table(ratings_title, index='userId', columns= 'title', values='rating')

print('Dataset dimensions: ', user_movie_ratings.shape, 'Subset example:')
user_movie_ratings.iloc[:6, :10]'''
st.code(code)

ratings_title = pd.merge(ratings, movies[['movieId', 'title']], on='movieId')
user_movie_ratings = pd.pivot_table(ratings_title, index='userId', columns= 'title', values='rating')

st.write(f'''Dataset dimensions: {user_movie_ratings.shape}''')
        
st.write(''' Top 6 rows:''')
st.dataframe(user_movie_ratings.iloc[:6, :10])


st.write('''The dominance of NaN values presents the first issue. Most users have not watched or rated
          most movies. Datasets like this are called "sparse" because only a small 
         number of cells have values. 

To address this, I'll sort by the movies with the highest number of ratings and the users
          who have rated the most movies. This approach will highlight a denser region
          when I examine the top of the dataset.

When selecting the most-rated movies and the most active users, it will appear as follows:

''')


code='''def get_most_rated_movies(user_movie_ratings, max_number_of_movies):
    # 1- Count
        user_movie_ratings = pd.concat([user_movie_ratings, pd.DataFrame([user_movie_ratings.count()])], ignore_index=True)    # 2- sort
    user_movie_ratings_sorted = user_movie_ratings.sort_values(len(user_movie_ratings)-1, axis=1, ascending=False)
    user_movie_ratings_sorted = user_movie_ratings_sorted.drop(user_movie_ratings_sorted.tail(1).index)
    # 3- slice
    most_rated_movies = user_movie_ratings_sorted.iloc[:, :max_number_of_movies]
    return most_rated_movies

def get_users_who_rate_the_most(most_rated_movies, max_number_of_movies):
    # Get most voting users
    # 1- Count
    most_rated_movies['counts'] = pd.Series(most_rated_movies.count(axis=1))
    # 2- Sort
    most_rated_movies_users = most_rated_movies.sort_values('counts', ascending=False)
    # 3- Slice
    most_rated_movies_users_selection = most_rated_movies_users.iloc[:max_number_of_movies, :]
    most_rated_movies_users_selection = most_rated_movies_users_selection.drop(['counts'], axis=1)


def sort_by_rating_density(user_movie_ratings, n_movies, n_users):
    most_rated_movies = get_most_rated_movies(user_movie_ratings, n_movies)
    most_rated_movies = get_users_who_rate_the_most(most_rated_movies, n_users)
    return most_rated_movies'''
st.code(code)

code='''n_movies = 30
n_users = 18
most_rated_movies_users_selection = sort_by_rating_density(user_movie_ratings, n_movies, n_users)

print('dataset dimensions: ', most_rated_movies_users_selection.shape)
most_rated_movies_users_selection.head()'''
st.code(code)



def get_most_rated_movies(user_movie_ratings, max_number_of_movies):
    # 1- Count
    user_movie_ratings = pd.concat([user_movie_ratings, pd.DataFrame([user_movie_ratings.count()])], ignore_index=True)
    # 2- Sort
    user_movie_ratings_sorted = user_movie_ratings.sort_values(len(user_movie_ratings)-1, axis=1, ascending=False)
    user_movie_ratings_sorted = user_movie_ratings_sorted.drop(user_movie_ratings_sorted.tail(1).index)
    # 3- Slice
    most_rated_movies = user_movie_ratings_sorted.iloc[:, :max_number_of_movies]
    return most_rated_movies

def get_users_who_rate_the_most(most_rated_movies, max_number_of_movies):
    # Get most voting users
    # 1- Count
    most_rated_movies['counts'] = pd.Series(most_rated_movies.count(axis=1))
    # 2- Sort
    most_rated_movies_users = most_rated_movies.sort_values('counts', ascending=False)
    # 3- Slice
    most_rated_movies_users_selection = most_rated_movies_users.iloc[:max_number_of_movies, :]
    most_rated_movies_users_selection = most_rated_movies_users_selection.drop(['counts'], axis=1)
    return most_rated_movies_users_selection

def sort_by_rating_density(user_movie_ratings, n_movies, n_users):
    most_rated_movies = get_most_rated_movies(user_movie_ratings, n_movies)
    most_rated_movies = get_users_who_rate_the_most(most_rated_movies, n_users)
    return most_rated_movies

n_movies = 30
n_users = 18
most_rated_movies_users_selection = sort_by_rating_density(user_movie_ratings, n_movies, n_users)

if most_rated_movies_users_selection is not None:
    st.write('dataset dimensions: ', most_rated_movies_users_selection.shape)
    st.dataframe(most_rated_movies_users_selection.head())
else:
    st.write("No data to display")


st.write('''This looks much improved. 
         Here's an effective way to visualize these ratings,
          making it easier to recognize patterns and clusters when examining larger subsets.

There is use of colors instead of numerical ratings:''')



def draw_movies_heatmap(most_rated_movies_users_selection, axis_labels=True):
    fig, ax = plt.subplots(figsize=(15,4)) 
    # Draw heatmap
    heatmap = ax.imshow(most_rated_movies_users_selection, interpolation='nearest', vmin=0, vmax=5, aspect='auto')
    if axis_labels:
         ax.set_yticks(np.arange(most_rated_movies_users_selection.shape[0]), minor=False)
         ax.set_xticks(np.arange(most_rated_movies_users_selection.shape[1]), minor=False)
         ax.invert_yaxis()
         ax.xaxis.tick_top()
         labels = most_rated_movies_users_selection.columns.str[:40]
         ax.set_xticklabels(labels, minor=False)
         ax.set_yticklabels(most_rated_movies_users_selection.index, minor=False)
         plt.setp(ax.get_xticklabels(), rotation=90)
    else:
         ax.get_xaxis().set_visible(False)
         ax.get_yaxis().set_visible(False)
         ax.grid(False)
         ax.set_ylabel('User id')
         # Separate heatmap from color bar
         divider = make_axes_locatable(ax)
         cax = divider.append_axes("right", size="5%", pad=0.05)
        # Color bar
         cbar = fig.colorbar(heatmap, ticks=[5, 4, 3, 2, 1, 0], cax=cax)
         cbar.ax.set_yticklabels(['5 stars', '4 stars', '3 stars', '2 stars', '1 star', '0 stars'])
    return fig 
fig = draw_movies_heatmap(most_rated_movies_users_selection)
st.pyplot(fig)


st.write('''Each column is a movie. Each row is a user. The color of the cell is how the
          user rated that movie (`Blue`: 0 stars, `Green`: 3 stars, `Yellow`: 5 stars).

When a cell is white, it indicates that the respective user did not rate that movie.
          This poses a challenge for clustering in real-world scenarios.
          Unlike the clean example we began with, actual datasets are often
          sparse and may not have a value in every cell. This makes it more complicated to cluster users based
          solely on their movie ratings, as k-means typically doesn't handle missing values well.

For performance reasons, I will use ratings for 1000 movies
          (out of the 9000+ available in the dataset).''')

code='''user_movie_ratings =  pd.pivot_table(ratings_title, index='userId', columns= 'title', values='rating')
most_rated_movies_1k = get_most_rated_movies(user_movie_ratings, 1000)'''
st.code(code)


user_movie_ratings =  pd.pivot_table(ratings_title, index='userId', columns= 'title', values='rating')
most_rated_movies_1k = get_most_rated_movies(user_movie_ratings, 1000)


st.write('''To enable sklearn to run k-means clustering on a dataset with missing values, 
         I will first convert it to a sparse CSR matrix. To achieve this, I will convert the pandas DataFrame to
          a SparseDataFrame and then use pandas' to_coo() method for the conversion.''')

code='''sparse_ratings = csr_matrix(pd.SparseDataFrame(most_rated_movies_1k).to_coo())'''
st.code(code)
user_movie_ratings = user_movie_ratings.fillna(user_movie_ratings.mean())
# Convert to sparse matrix 
sparse_ratings = sp.csr_matrix(user_movie_ratings.values)

st.write('''## Clusters
With k-means, we have to specify k, the number of clusters. I will arbitrarily try k=20 :''')

code='''# 20 clusters
predictions = KMeans(n_clusters=20, algorithm='full', random_state=40).fit_predict(sparse_ratings)'''
st.code(code)
# 20 clusters
predictions = KMeans(n_clusters=20, algorithm='lloyd', random_state=40).fit_predict(sparse_ratings)



code='''def draw_movie_clusters(clustered, max_users, max_movies):
    c=1
    for cluster_id in clustered.group.unique():
        # To improve visibility, I show most max_users users and max_movies movies per cluster.
        # You can change these values to see more users & movies per cluster
        d = clustered[clustered.group == cluster_id].drop(['index', 'group'], axis=1)
        n_users_in_cluster = d.shape[0]
        
        d = sort_by_rating_density(d, max_movies, max_users)
        
        d = d.reindex(d.mean().sort_values(ascending=False).index, axis=1)
        d = d.reindex(d.count(axis=1).sort_values(ascending=False).index)
        d = d.iloc[:max_users, :max_movies]
        n_users_in_plot = d.shape[0]
        
        # I will select to show clusters that have more than 9 users
        if len(d) > 9:
            print('cluster # {}'.format(cluster_id))
            print('# of users in cluster: {}.'.format(n_users_in_cluster), '# of users in plot: {}'.format(n_users_in_plot))
            fig = plt.figure(figsize=(15,4))
            ax = plt.gca()

            ax.invert_yaxis()
            ax.xaxis.tick_top()
            labels = d.columns.str[:40]

            ax.set_yticks(np.arange(d.shape[0]) , minor=False)
            ax.set_xticks(np.arange(d.shape[1]) , minor=False)

            ax.set_xticklabels(labels, minor=False)
                        
            ax.get_yaxis().set_visible(False)

            # Heatmap
            heatmap = plt.imshow(d, vmin=0, vmax=5, aspect='auto')

            ax.set_xlabel('movies')
            ax.set_ylabel('User id')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            # Color bar
            cbar = fig.colorbar(heatmap, ticks=[5, 4, 3, 2, 1, 0], cax=cax)
            cbar.ax.set_yticklabels(['5 stars', '4 stars','3 stars','2 stars','1 stars','0 stars'])

            plt.setp(ax.get_xticklabels(), rotation=90, fontsize=9)
            plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', labelbottom='off', labelleft='off') 
            print('cluster # {} (Showing at most {} users and {} movies)'.format(cluster_id, max_users, max_movies))

            plt.show()


'''
st.code(code)


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
from mpl_toolkits.axes_grid1 import make_axes_locatable

def draw_movie_clusters(clustered, max_users, max_movies):
    c = 1
    for cluster_id in clustered.group.unique():
        # To improve visibility, we're showing at most max_users users and max_movies movies per cluster.
        d = clustered[clustered.group == cluster_id].drop(['index', 'group'], axis=1)
        n_users_in_cluster = d.shape[0]

        d = sort_by_rating_density(d, max_movies, max_users)
        
        d = d.reindex(d.mean().sort_values(ascending=False).index, axis=1)
        d = d.reindex(d.count(axis=1).sort_values(ascending=False).index)
        d = d.iloc[:max_users, :max_movies]
        n_users_in_plot = d.shape[0]

        # We're only selecting to show clusters that have more than 9 users
        if len(d) > 9:
            st.write(f'Cluster # {cluster_id}')
            st.write(f'Users in cluster: {n_users_in_cluster}.  Users in plot: {n_users_in_plot}')
            
            fig, ax = plt.subplots(figsize=(15, 4))

            ax.invert_yaxis()
            ax.xaxis.tick_top()
            labels = d.columns.str[:40]

            ax.set_yticks(np.arange(d.shape[0]), minor=False)
            ax.set_xticks(np.arange(d.shape[1]), minor=False)

            ax.set_xticklabels(labels, minor=False)
                        
            ax.get_yaxis().set_visible(False)

            # Heatmap
            heatmap = ax.imshow(d, vmin=0, vmax=5, aspect='auto')

            ax.set_xlabel('Movies')
            ax.set_ylabel('User ID')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            # Color bar
            cbar = fig.colorbar(heatmap, ticks=[5, 4, 3, 2, 1, 0], cax=cax)
            cbar.ax.set_yticklabels(['5 stars', '4 stars', '3 stars', '2 stars', '1 star', '0 stars'])

            plt.setp(ax.get_xticklabels(), rotation=90, fontsize=9)
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
            
            st.write(f'Cluster # {cluster_id} (Showing at most {max_users} users and {max_movies} movies)')
            st.pyplot(fig)
        
        c += 1
        if c > 6:
            break


code='''max_users = 70
max_movies = 50

clustered = pd.concat([most_rated_movies_1k.reset_index(), pd.DataFrame({'group':predictions})], axis=1)
draw_movie_clusters(clustered, max_users, max_movies)
'''
st.code(code)


max_users = 70
max_movies = 50

clustered = pd.concat([most_rated_movies_1k.reset_index(), pd.DataFrame({'group':predictions})], axis=1)
draw_movie_clusters(clustered, max_users, max_movies)


st.write('''There are several things to note here:
* The more similar the ratings in a cluster are, the more **vertical** lines in similar colors you'll be able to trace in that cluster. 
* Some clusters are more sparse than others, containing people who probably watch and rate less movies than in other clusters.
* Some clusters are mostly yellow and bring together people who really love a certain group of movies. Other clusters are mostly green or navy blue meaning they contain people who agree that a certain set of movoies deserves 2-3 stars.
* The movies change in every cluster. The graph filters the data to only show the most rated movies, and then sorts them by average rating.
* It's easy to spot **horizontal** lines with similar colors, these are users without a lot of variety in their ratings. 
         This is likely one of the reasons for Netflix switching from a stars-based ratings to a thumbs-up-thumbs-down rating. A rating of four stars means different things to different people.

## Prediction
I will pick a cluster and a specific user and see what useful things this clustering will allow us to do.

Cluster 6:''')




# Pick a cluster ID from the clusters above
cluster_number = 6

# Let's filter to only see the region of the dataset with the most number of values 
n_users = 75
n_movies = 300
cluster = clustered[clustered.group == cluster_number].drop(['index', 'group'], axis=1)

cluster = sort_by_rating_density(cluster, n_movies, n_users)
fig=draw_movies_heatmap(cluster, axis_labels=False)
st.pyplot(fig)

st.write('''And the actual ratings in the cluster look like this:''')

st.dataframe(cluster.fillna('').head(10))

st.write('''Let's look, for example, at `Matrix, The (1999)`''')

st.dataframe(cluster['Matrix, The (1999)'])

st.write(''' In this column, there are empty cells where users did not rate that movie. 
         Can I predict how these users would rate the movie ? Since the users are in a cluster of users that 
         seem to have similar taste, I can take the average of the votes for that movie in this cluster,
          and that would be a reasonable 
         prediction.''')

code='''# Example
movie_name = "Matrix, The (1999)"

cluster[movie_name].mean()'''
st.code(code)

movie_name = "Matrix, The (1999)"


st.write(f'''{cluster[movie_name].mean():.1f}''')


st.write(''' And this would be my prediction for how people who didn't vote, would rate the movie.

## Recommendation
So far, I've utilized k-means to group users based on their ratings,
          resulting in clusters of users with similar ratings and, consequently,
          a similar taste in movies. Using this method, when a user didn't have a 
         rating for a particular movie, I estimated their preference by averaging the
          ratings of the other users in the same cluster.

By applying this logic, calculating the average score for each movie within a cluster allows
        to understand how this 'taste cluster' collectively feels about each movie
          in the dataset.

''')



code='''
# Calculate mean and rename columns
cluster_mean = cluster.mean().head(20).reset_index()
cluster_mean = round(cluster_mean,1)
cluster_mean.columns = ['Movie', 'Average Rating']
# Display the DataFrame 
cluster_mean'''
st.code(code)

# The average rating of 20 movies as rated by the users in the cluster
# Calculate mean and rename columns
cluster_mean = cluster.mean().head(20).reset_index()
cluster_mean = round(cluster_mean,1)
cluster_mean.columns = ['Movie', 'Average Rating']
#  # Display the DataFrame with new column names
st.dataframe(cluster_mean, hide_index=True)


st.write('''I can can now use it as a **recommendation engine**
          that enables users to discover movies they're likely to enjoy.
When a user logs in to this app, she can find recommendations that are appropriate
          to her taste. The formula for these recommendations is to select the cluster's highest-rated
          movies that the user did not rate yet.
''')

code='''# A random user 
user_id = 29

# This user's ratings
user_29_ratings  = cluster.loc[user_id, :]

# Movies she didn't rate (I don't want to recommend movies she has already rated!)
user_29_unrated_movies =  user_29_ratings[user_29_ratings.isnull()]

# What are the ratings of these movies the user did not rate?
avg_ratings = pd.concat([user_29_unrated_movies, cluster.mean()], axis=1, join='inner').loc[:,0]

# Sort by highest rated movies
avg_ratings.sort_values(ascending=False)[:20])'''
st.code(code)

# A random user 
user_id = 29

# Get all this user's ratings
user_2_ratings  = cluster.loc[user_id, :]

# Movies she didn't rate (I don't want to recommend movies she has already rated!)
user_2_unrated_movies =  user_2_ratings[user_2_ratings.isnull()]

# What are the ratings of these movies the user did not rate?
avg_ratings = pd.concat([user_2_unrated_movies, cluster.mean()], axis=1, join='inner').loc[:,0]

# Sort by rating so the highest rated movies are presented first
ratez=round(avg_ratings.sort_values(ascending=False)[:20],1)
ratez.columns=['Movie', 'Average Rating']
st.dataframe(ratez)



st.write('''And these are my top 20 recommendations to the user!''')

