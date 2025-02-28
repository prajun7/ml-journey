# Machine Learning: Movie Recommender System | Nearest vector to find the similarities

# I should do this and deploy the project to Heroku.

# YT link: https://www.youtube.com/watch?v=1xtrIEwY_zY&list=PPSV
# YT Title: Movie Recommender System Project | Content Based Recommender System with Heroku Deployment
# YT Channel: CampusX


############# Data Processing and cleaning #####################

# Reading the csv file
movies = pd.read_csv('rmdb_5000_movies.csv');
credits = pd.read_csv('rmdb_5000_credits.csv');

# Displays all the data
movies.head()


# Displays the first movie
movies.head(1)

# Displays the credits cast value. cast is the column
credits.head(1)['cast'].values

# Merging two datasets that is movies and credits together and storing it into mavies based on the title of the movie. Title coulmn was present on both movies and credits.
movies = movies.merge(credits, on = 'title').shape

# counts the values that are in the movies for the column original_langiage
movies['original_language'].value_counts()
# output: 
# en 4510
# fr 70
# es 32

# We will only need movie_id, title, overview, genres, keywords, cast and crew from the movie dataset to create the movie recomendation. So we are extracting the columns.
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# To check if anyone of the columns have empty or null value
movies.isnull().sum()

# To drop the empty ones or null
movies.dropna(inplace=True)

# Check duplicate data. Should result 0
movies.duplicated().sum()

# To see the values of the genres which is a column in the movies data
movies.iloc[0].genres

# Helper function to clean the data
# I have this data, obj = [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]
# And I want in this format, ["Action", "Adventure"]
import ast
def cleanData(obj):
	L = []
	for i in ast.literal_eval(obj):
		L.append(i['name'])
	return L

# To apply above cleanData function we can use apply
movies['genres'].apply(cleanData)

# To save this data into genres column in the movies
movies['genres'] = movies['genres'].apply(cleanData)

# To convert sentences into list we can do,
# Input = 'In the 22nd century, a Marine is dispatched to the moon'
# Output =  ["In", "the", "22nd", "centiry," "a", "Marine", "is", "dispatched", "to", "the", "moon"]
movies['overview'] = movies['overview'].apply(lambda x:x.split())

# To remove spaces from words which are in list
# Input = [John Wick, James Bond]
# Outout = [JohnWich, JamesBond]
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])

# Create a new column called tags based on different columns
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create a new data that only consisits of movie_id, title and tags. That is we are creating new data frame and removing unwanted columns
new_df = movies[['movie_id', 'title', 'tags']]


############# Recommending the movies #####################

# Now based on the similarities of the tags column we need to recommend the movies. Tags columns are sentences
# We need to use vectorization
# Our data format, movie_id | name | tags and we have around 5000 data
# We will convert Each text in the tag column to vector. So, each movie will become vectors. So, when we vectorize all the movies we need to find the closest vector and recommend those movies.
# Text Vectorization: Process of converting text into vector
# Different techniques to perform this, Bag of words, df idf, word to vec. We will use simple technique that is Bag or words
# Bag of words: We will convert all the words that is in the tags columns. That is, largeTexts = tag1 + tag2 + .... Now we need to find the 5000 most repeated words with highest frequency from that largeTexts. Now we will go through each tag of the movies and check how many times those most 5000 common words are repeating. We will create a table of those 5000 words and for each word we will populate the number of times those words are repeating in the tags for each movies.
# We will also not consider stop words like in, are, to, from, if. We will remove these words and than perform the vectorization

# Here the tags should be a sentences(string) and in lowercase

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_feature = 5000, stop_words = 'english')
# Here in the above code we are creating 5000 feature words and we are removing the stop words for the language english

# converting the values into numpy array, that is vectors
vectores = cv.fit_transform(new_df['tags']).toarray()

# Displays the most common repeated words, in our case it is 5000
cv.get_feature_names()

# len returns 5000 words in our corpes
len(cv.get_feature_names())

# Some words like actor, actors should be same. So, we need to make those words as one.
# we will use stem that is,
# Input: ['loved', 'loving', 'love']
# Output: ['love', 'love', 'love']
# Basically stem is converting same words to just 1 word. in our case it is love

import nltk
# to install this library, we can do !pip install nltk in jupyter
from nlth.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
	y = []
	for i in text.split():
		y.append(ps.stem(i))
	return " ".join(y)

# Here 
# ps.stem('loving') => return love
# ps.stem('loved') => return love and so on

# Applying stem to the tags column
new_df['tags'] = new_df['tags'].apply(stem)

# Now when I run the same above code that is,

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_feature = 5000, stop_words = 'english')
# Here in the above code we are creating 5000 feature words and we are removing the stop words for the language english

# converting the values into numpy array, that is vectors
vectors = cv.fit_transform(new_df['tags']).toarray()

# Displays the most common repeated words, in our case it is 5000
cv.get_feature_names()
# Here since this time we used stem, we wont will see word like love instead of loving, loved and so on

# Now since all the moovies are vectorized, now for each movie we need to find the distance of other movie. And we will do this for all the movies. The less distance represents they are more similar movies but higher distance will represents they are not similar movies.
# We will calculate consine distance(We will calculate the angle or theta between two vectors) not Ecudarian distance. Since we have higher dimension vectore that is (5000, 5000), it is better to use consine distance as it is more accurate compared to ecudarian distance.
# Less Theta or angle means it is more similar movie.

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors) 

# here we are calucating the distance for each moviee with other movies that is, for first movie we calculate the distance of the first movie with 5000 other movies, and same for second and third. 
# similarity is the array of array
# If the siimilarity score is 1, it is same movie, that is the range is 0 to 1

# function to recommend other 5 similar movies based on the given movie
# Fetch the movie from the similarity array and sort that. 
# We need to store the index of the similarity because based on that we are finding which movie is similar   
# We cannot sort directly as it will change the index of the element. So, we will use enumarate to create a pair of index and the value. This will help to track the index. That is list(enumarate(siimilarity[0]))
# Now we can sort the similarity based on the second pair, that is, sorted(list(enumarate(similarity[0])), reverse=True, key=lambda x:x[1])
# Input: movie : title of the movie (string)
def recommend(movie):
	 movie_index = new_df[new_df['title'] == movie].index[0]
	 distances = similarity[movie_index] 
	 movies_list = sorted(list(enumarate(distances)), reverse=True, key=lambda x:x[1])[1:6]

	 for i in movies_list:
	 	print(new_df.iloc[i[0]].title)


# Now we can use the recommend methodto recommend 5 different movies based on the given title.


# To get the data while creating the web application from jupyter to Pycharm we can use pickle.
# We are using pycharm, and also using streamlit to create the web application.










