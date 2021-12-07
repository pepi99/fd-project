import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class MovieRecommender:
    def __init__(self, movies, ratings):
        self.movies = movies
        self.ratings = ratings

        # Create matrix with moviesIds and ratings(userids)
        final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating')
        # Fill NaNs with 0s
        final_dataset.fillna(0, inplace=True)
        self.ds = final_dataset

    def preprocess(self):
        # Movie IDS and total number of votes for that movie
        no_user_voted = self.ratings.groupby('movieId')['rating'].agg('count')

        # User ids and total number of votes for each user
        no_movies_voted = self.ratings.groupby('userId')['rating'].agg('count')

        # Filter
        final_dataset = self.ds.loc[no_user_voted[no_user_voted > 10].index, :]
        final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]

        self.reduced_matrix = csr_matrix(final_dataset.values)

        final_dataset.reset_index(inplace=True)
        self.ds = final_dataset

    def fit(self, n_neighbours=20):
        knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbours, n_jobs=-1)
        knn.fit(self.reduced_matrix)
        self.knn = knn

    def get_movie_recommendation(self, movie_name):
        n_movies_to_reccomend = 10
        movie_list = self.movies[self.movies['title'].str.contains(movie_name)]

        if len(movie_list):
            movie_idx = movie_list.iloc[0]['movieId']
            movie_idx = self.ds[self.ds['movieId'] == movie_idx].index[0]
            distances, indices = self.knn.kneighbors(self.reduced_matrix[movie_idx],
                                                     n_neighbors=n_movies_to_reccomend + 1)
            rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
                                       key=lambda x: x[1])[:0:-1]
            recommend_frame = []

            for val in rec_movie_indices:
                movie_idx = self.ds.iloc[val[0]]['movieId']
                idx = self.movies[self.movies['movieId'] == movie_idx].index
                recommend_frame.append({'Title': self.movies.iloc[idx]['title'].values[0], 'Distance': val[1]})
            df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_reccomend + 1))
            return df
        else:
            return "No movies found. Please check your input"
