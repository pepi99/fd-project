import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from MovieRecommender import MovieRecommender

movies = pd.read_csv("data/archive(1)/movies.csv")
ratings = pd.read_csv("data/archive(1)/ratings.csv")


recommender = MovieRecommender(movies, ratings)
recommender.preprocess()
recommender.fit()
recommendations = recommender.get_movie_recommendation('Memento')
print(recommendations)
