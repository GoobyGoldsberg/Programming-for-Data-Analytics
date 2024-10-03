


import pandas as pd
import matplotlib.pyplot as plt

MOVIES = pd.read_csv('Movies-1.csv')

def task1():
    
    unique_main_genres = len(MOVIES['main_Genre'].unique())
    
    most_popular_mgenre = MOVIES['main_Genre'].mode().values[0].capitalize()
    
    least_popular_mgenre = MOVIES['main_Genre'].value_counts().idxmin().capitalize()
    
    
    print(f'Number of unique main genres: {unique_main_genres}')
    
    print(f'Most popular genre is: {most_popular_mgenre}')
    
    print(f'Least popular genre is: {least_popular_mgenre}')
    
    
    most_popular_mgenres = MOVIES['main_Genre'].value_counts().nlargest(8)
    genres_names = most_popular_mgenres.index
    genres_count = most_popular_mgenres.values
    
    
    plt.bar(genres_names, genres_count, color='lightgreen')
    
    plt.title('8 Most Popular Genres')
    plt.ylabel('Number of Movies')
    plt.xlabel('Main Genre')
    plt.show()
    
    
def task2():
    
    
    most_common_genre = MOVIES['Genre'].value_counts().idxmax()

    least_common_genre = MOVIES['Genre'].value_counts().idxmin()
    
    print(f'Most common genre is: {most_common_genre}')
    
    print(f'Least common genre is: {least_common_genre}')


def task3():
    
    runtime_data = MOVIES['Runtime']
    
    numerical_runtime_data = pd.to_numeric(runtime_data.str.replace(' min', ''), errors='coerce')

    numerical_runtime_data = numerical_runtime_data.dropna()
    
    plt.boxplot(numerical_runtime_data)
    

    plt.title('Runtime Data on Boxplot')
    plt.ylabel('Runtime in minutes')
    

    
    plt.show()  


    
def task4():
    
    votes_mean = MOVIES['Number of Votes'].mean()
    
    ratings_mean = MOVIES['Rating'].mean()

    if MOVIES['Number of Votes'].isnull().any():
        MOVIES['Number of Votes'].fillna((votes_mean), inplace=True)
        
    if MOVIES['Rating'].isnull().any():
        MOVIES['Rating'].fillna((ratings_mean), inplace=True)
        
    
    
    plt.figure(figsize=(12, 6))
    plt.scatter(MOVIES['Number of Votes'], MOVIES['Rating'])
    
    plt.text(1, 11, "1.0 means 1million", fontsize=8)

    plt.xlabel('Number of Votes')
    plt.ylabel('Rating')
    plt.title('Relationship between Number of Votes and Rating')
    
    
    plt.show()

    # By looking at the scatter plot we can see that the movies with the highest number of votes tend to clump toward the
    # rating of between 7 and 9, most commonly 8.
    
    # The existence of null values in the ratings/votes attributes could have couple of reasons:
    # 1. The movie came out recently and the data has not yet been released
    # 2. Nobody watched the movie
    
def task5():
    
    file_path = 'main_genre.csv'
    
    main_genres = pd.read_csv(file_path, encoding='ISO-8859-1')
    main_genres = main_genres.map(lambda x: x.lower() if isinstance(x, str) else x)
    main_genres.columns = main_genres.columns.str.lower()
    
    synopsis = MOVIES['Synopsis'].str.lower()
    
    synopsis = synopsis.str.replace('[,\'\.\-]', '', regex=True)
    
    
    most_frequent_term = {}
    
    for genre in main_genres.columns:
        
        term_frequency = {}

        
        for term in main_genres[genre]:
            
            for text in synopsis:
                if term in text:
                    term_frequency[term] = term_frequency.get(term, 0) + 1
        
        most_frequent_term = max(term_frequency, key=term_frequency.get)
        
        print(f'{genre} - {most_frequent_term}')  
    
def task6():
    
    # This function creates a bar chart of top 5 most average rated movies
    # The expected output is 5 barcharts, with the average rating on the Y axis and 5 Popular genres in the X axis
    
    average_rating_for_genre = MOVIES.groupby('main_Genre')['Rating'].mean()
    
    sorted_genres = average_rating_for_genre.sort_values(ascending=False)
    
    top_5_genres = sorted_genres.head(5)
    
    plt.figure(figsize=(10, 6))
    top_5_genres.plot(kind='bar', color='lightgreen')
    plt.title('Top 5 Most Popular Main Genres by Average Rating')
    plt.xlabel('Main Genre')
    plt.ylabel('Average Rating')
    plt.ylim(0, 10)
    
    plt.show()
    
#task1()
#task2()
#task3()
#task4()
#task5()
#task6()
    
        
