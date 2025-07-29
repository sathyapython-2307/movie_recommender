from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_sqlalchemy import SQLAlchemy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
db = SQLAlchemy(app)
api = Api(app)

# Database Models
class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), unique=True, nullable=False)
    genre = db.Column(db.String(100))

class Rating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    movie = db.relationship('Movie')

# Recommendation Engine
class Recommender:
    @staticmethod
    def get_recommendations(user_id, num_recommendations=5):
        # Get all ratings
        ratings = Rating.query.all()
        
        # Create ratings matrix
        data = [[r.user_id, r.movie_id, r.rating] for r in ratings]
        df = pd.DataFrame(data, columns=['user_id', 'movie_id', 'rating'])
        
        # Pivot table (users x movies)
        ratings_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating')
        ratings_matrix = ratings_matrix.fillna(0)
        
        # Normalize ratings
        scaler = MinMaxScaler()
        normalized_ratings = scaler.fit_transform(ratings_matrix)
        
        # Calculate similarity
        similarity = cosine_similarity(normalized_ratings)
        similarity_df = pd.DataFrame(similarity, index=ratings_matrix.index, columns=ratings_matrix.index)
        
        # Get similar users
        similar_users = similarity_df[user_id].sort_values(ascending=False)[1:6]
        
        # Get recommended movies
        similar_users_ratings = ratings_matrix.loc[similar_users.index]
        recommended_movies = similar_users_ratings.mean(axis=0).sort_values(ascending=False)
        
        # Filter out movies already rated by user
        user_rated = ratings_matrix.loc[user_id]
        unrated_movies = recommended_movies[user_rated == 0]
        
        return unrated_movies.head(num_recommendations).index.tolist()

# API Resources
class MovieRecommendation(Resource):
    def get(self, user_id):
        recommendations = Recommender.get_recommendations(user_id)
        movies = [Movie.query.get(mid).title for mid in recommendations]
        return {'recommendations': movies}

class RateMovie(Resource):
    def post(self):
        data = request.get_json()
        rating = Rating(
            user_id=data['user_id'],
            movie_id=data['movie_id'],
            rating=data['rating']
        )
        db.session.add(rating)
        db.session.commit()
        return {'message': 'Rating added successfully'}

api.add_resource(MovieRecommendation, '/recommend/<int:user_id>')
api.add_resource(RateMovie, '/rate')

@app.route('/')
def home():
    return "Movie Recommendation Engine"

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        # Add sample movies if empty
        if Movie.query.count() == 0:
            movies = [
                Movie(title="The Shawshank Redemption", genre="Drama"),
                Movie(title="The Godfather", genre="Crime"),
                Movie(title="The Dark Knight", genre="Action"),
                Movie(title="Pulp Fiction", genre="Crime"),
                Movie(title="Fight Club", genre="Drama")
            ]
            db.session.add_all(movies)
            db.session.commit()
    
    app.run(debug=True)