import csv
import re

def load_movies(filename):
    movies = []

    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            movies.append(row)

    return movies

def tokenize(text):
    text = text.lower()
    return re.findall(r'\b[a-z0-9]+\b', text)

def create_profile(movie):
    text = (
        movie["genre"] + " " +
        movie["keywords"] + " " +
        movie["description"]
    )

    return set(tokenize(text))

def calculate_similarity(profile1, profile2):
    if len(profile1) == 0 or len(profile2) == 0:
        return 0

    common_words = profile1.intersection(profile2)

    return len(common_words) / len(profile1.union(profile2))

def recommend_movies(movie_name, movies, number_of_recommendations=5):
    selected_movie = None

    for movie in movies:
        if movie["title"].lower() == movie_name.lower():
            selected_movie = movie
            break

    if selected_movie is None:
        print("\nMovie not found!")
        return

    selected_profile = create_profile(selected_movie)

    recommendations = []

    for movie in movies:
        if movie["title"].lower() == movie_name.lower():
            continue

        movie_profile = create_profile(movie)

        similarity = calculate_similarity(
            selected_profile,
            movie_profile
        )

        recommendations.append(
            (movie["title"], similarity)
        )

    recommendations.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("\nRecommended Movies")
    print("--------------------------")

    for i, (title, score) in enumerate(
        recommendations[:number_of_recommendations],
        start=1
    ):
        print(f"{i}. {title} (Similarity: {score:.2f})")

def display_movies(movies):
    print("\nAvailable Movies")
    print("--------------------------")

    for i, movie in enumerate(movies, start=1):
        print(f"{i}. {movie['title']}")

def main():
    filename = "movies.csv"

    movies = load_movies(filename)

    print("================================")
    print("   MOVIE RECOMMENDATION SYSTEM")
    print("================================")

    display_movies(movies)

    while True:
        movie_name = input(
            "\nEnter a movie name or type 'exit': "
        )

        if movie_name.lower() == "exit":
            print("Thank you for using the system!")
            break

        recommend_movies(
            movie_name,
            movies,
            5
        )

if __name__ == "__main__":
    main()
