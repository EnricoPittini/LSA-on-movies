from typing import Union
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from textblob import TextBlob
import random



########## FUNCTIONS FOR THE CONTENT-BASED PART


def lemmatize_with_postag(sentence : str):
    """Lemmatize the given string, i.e. the sequence of words is transformed into the corresponding sequences of lemmas.

    Parameters
    ----------
    sentence : str
        Sequence of words to lemmatize

    Returns
    -------
    str
        Sequence of corresponding lemmas
    """
    sent = TextBlob(sentence)
    tag_dict = {"J": 'a', 
                "N": 'n', 
                "V": 'v', 
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return " ".join(lemmatized_list)


def process_text(s : str): 
    """Process the given string.

    The operations performed are the following.
    - Only the alhpabetic characters are kept (the punctuation and the numbers are removed).
    - All the words are transformed into lowercase.
    - The stopwords are removed 
    - The words with unitary length are removed 
    - Some particular wrong words are removed (e.g. 'äì').

    Parameters
    ----------
    s : str
        Input string

    Returns
    -------
    str
        Processed string 
    """
    return ' '.join([word for word in ''.join(ch if ch.isalpha() else ' ' for ch in str(s)).lower().replace('äì',' ').replace('äôs',' ').replace('äù',' ').split() 
            if word not in stopwords.words('english') and len(word)>1])


def preprocess_textual_column(df : pd.DataFrame, text_col : str, lemmatize : bool = False):
    """Preprocess the specified textual column in the given dataframe. 

    The following preprocessing operations are performed.
    - The missing values of the textual column and the empty strings are removed.
    - Only the alhpabetic characters are kept (the punctuation and the numbers are removed).
    - All the words are transformed into lowercase.
    - The stopwords are removed 
    - The words with unitary length are removed 
    - Some particular wrong words are removed (e.g. 'äì').
    - Optionally, the words can be lemmatized.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    text_col : str
        Name of the column containing the textual data to process
    lemmatize : bool, optional
        Whether to lemmatize or not the works, by default False

    Returns
    -------
    pd.DataFrame
        New dataframe, equal to the one in input, except for two things.
        - The column with name `text_col` now contains the preprocessed text, not the original one.
        - The new column with name `original {text_col}` contains the original text.
    """
    df = df.copy()
    df[f'original {text_col}'] = df[text_col]
    df = df[~df[text_col].isna()]
    df = df[df[text_col].map(lambda s: s!='')]
    df = df.reset_index(drop=True)
    df[text_col] = df[text_col].map(process_text)
    if lemmatize:
        df[text_col] = df[text_col].map(lemmatize_with_postag)
    return df


def build_voc(df : pd.DataFrame, text_col : str, docs_freq_thresh : float = None):
    """Build the vocabulary from the specified textual column in the given dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    text_col : str
        Name of the column containing the textual data to process
    docs_freq_thresh : float, optional
        Threshold to use for the document frequency of the words in the vocabulary, by default None.
        More precisely, if specified, it must a number in [0,1]. The meaning is that the words with a documents frequency 
        lower than that threshold are deleted from the vocabulary.

    Returns
    -------
    voc : np.ndarray
        Vocabulary, i.e. list of words. It represents the mapping from integer ids into words.
        This list of words is sorted in descending order with respect to the documents frequency.
    """
    voc = np.array(list(set([word for text in df[text_col] for word in text.split()])))
    #docs_freq = np.array([df[text_col].map(lambda s: f' {voc[word_id]} ' in s).sum() for word_id in range(voc.shape[0])])
    docs_freq = np.array([df[text_col].map(lambda s: f' {word} ' in s).sum() for word in voc])
    voc = voc[np.argsort(docs_freq)[::-1]]
    docs_freq = np.sort(docs_freq)[::-1]
    if docs_freq_thresh is not None:
        print(f'Original number of words: {len(voc)}')
        voc = voc[(docs_freq/df.shape[0])>=docs_freq_thresh]
        print(f'Number of words after thresholding: {len(voc)}')
    return voc


def build_termsDocs_matrix(df : pd.DataFrame, text_col : str, voc : np.ndarray):
    """Build the terms-documents matrix from the specified textual column in the given dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    text_col : str
        Name of the column containing the textual data to process
    voc : np.array
        Vocabulary of the specified textual column

    Returns
    -------
    np.ndarray
        Bidimensional array, representing the terms-documents matrix. Namely, the rows correspon to the terms, while the 
        columns correspond to the documents. Each cell contains the number of occurances of that term in that document.
    """
    terms_docs = np.zeros(shape=(len(voc), df.shape[0]))
    word2id = {word:id for id,word in enumerate(voc)}
    for doc_id in range(df.shape[0]):
        s = df.loc[doc_id,text_col]
        for word in s.split():
            if word in voc:
                terms_docs[word2id[word],doc_id] += 1
    return terms_docs


def build_tfidf(df : pd.DataFrame, text_col : str, voc : np.ndarray, terms_docs : np.ndarray):
    """Build the tf-idf (i.e. 'term frequency - inverse document frequency) from the specified textual column of the given dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    text_col : str
        Name of the column containing the textual data to process
    voc : np.ndarray
        Vocabulary of the specified textual column
    terms_docs : np.ndarray
        Bidimensional array, representing the terms-documents matrix of the specified textual column

    Returns
    -------
    np.ndarray
        Bidimensional array, representing the tf-idf matrix. Namely, the rows correspon to the terms, while the 
        columns correspond to the documents. Each cell contains the tf-idf weight of that term w.r.t. that document.
    """
    docs_freq = np.array([df[text_col].map(lambda s: voc[word_id] in s).sum() for word_id in range(terms_docs.shape[0])])
    tfidf = np.log(terms_docs+1) * np.reshape(np.log(df.shape[0]/docs_freq), newshape=(len(voc),1)) 
    return tfidf


def preprocess_genres(df : pd.DataFrame, genres_col : str):
    """Preprocess the specified column containing the movie genres in the given dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    genres_col : str
        Name of the column containing the genres of the movies

    Returns
    -------
    pd.DataFrame
        New dataframe, equal to the one in input, except for the fact that the column `genres_col` now contain sets of genres
    """
    df_p = df.copy()
    df_p[genres_col] = df_p[genres_col].map(lambda genres_set: set([g['name'] for g in eval(str(genres_set))]) 
                                                                    if str(genres_set)!='nan' else '')
    return df_p



########## FUNCTIONS FOR THE COLLABORATIVE PART


def preprocess_dfMovies_100k_coll(df_movies : pd.DataFrame):
    """Process the 100k movies dataframe.

    The MovieLens 100k movies dataset is different from the others, and it recquires more preprocessing. 
    In particular, the genres are not encoded as a single column containing the set of genres of each film. But they are 
    encoded in a one-hot encoding way.
    Therefore, we want to transform that one-hot encoding into the single set column representation.

    Parameters
    ----------
    df_movies : pd.DataFrame
        MovieLens 100k movies dataset

    Returns
    -------
    pd.DataFrame
        Processed 100k movies dataset
    """
    df_movies = df_movies.copy()
    genres_list = np.array(df_movies.columns[5:])
    df_movies_genres = []
    for movie_id in range(df_movies.shape[0]):
        genres_set = set(genres_list[df_movies.loc[movie_id][genres_list].to_numpy().astype(bool)])
        df_movies_genres.append(genres_set)
    df_movies['genres'] = df_movies_genres
    df_movies = df_movies[['movie id', 'movie title', 'genres']]
    return df_movies 


def preprocess_movies_ratings_coll(df_movies : pd.DataFrame, df_ratings : pd.DataFrame):
    """Process the movies and ratings dataframe.

    Parameters
    ----------
    df_movies : pd.DataFrame
        Movies dataframe
    df_ratings : pd.DataFrame
        Ratings dataframe

    Returns
    -------
    df_movies : pd.DataFrame
        Processed movies dataframe
    df_ratings : pd.DataFrame
        Processed movies dataframe
    """
    # We transform the users indices from [1..n_users] into [0..n_users-1]
    df_movies, df_ratings = df_movies.copy(), df_ratings.copy()
    df_ratings['userId'] = df_ratings['userId'].map(lambda u: u-1)

    # We map the movies indices in order to be equal to the indices of the movies dataframe
    moviesIds_map = np.array([df_movies.loc[i, 'movieId'] for i in range(df_movies.shape[0])])
    moviesIds_map_rev = {id2:id1 for id1,id2 in enumerate(moviesIds_map)}
    df_ratings['movieId'] = df_ratings['movieId'].map(moviesIds_map_rev)
    df_movies = df_movies.drop(['movieId'], axis=1)

    return df_movies, df_ratings


def build_rating_matrix(df_movies : pd.DataFrame, df_ratings : pd.DataFrame, n_users : int, fill_na : str = 'zero', 
                        subtract_mean : str = 'zero', test_prop : float = 0.0, random_seed : int = 44):
    """Build the rating matrix.

    The rating matrix is a bidimensional matrix with shape (n_users,n_movies): so, the rows correspond to the users, while 
    the columns correspond to the movies. Each cell contains the rate given from that user to that movie. 

    A very important point is the handling of the missing values: a missing value is a cell (user_id,movie_id) s.t. that user
    has never rated that movie.
    These missing values must be replaced with something. 
    The four following strategies can be specified.
    - 'zero'. The missing values are simply replaced with 0.
    - 'items_means'. Each missing value is replaced with the mean of the column in which it belongs, i.e. the mean of the
       rates of that movie. If a movie has no rates, all its values are replaced with the global mean of all rates.
    - 'users_means'. Each missing value is replaced with the mean of the row in which it belongs, i.e. the mean of the
       rates given by that user. If a user has never given a rate, all its values are replaced with the global mean of all 
       rates.
    - 'items_users_means'. Each missing value is replaced with the mean between the mean of the column (i.e. item mean) and
       the mean of the row (i.e. user mean). 

    Another important point is that it is typically useful to subtract the values in the rating matrix by certain means. 
    Basically, it is typically beneficial to do not work directly with the ratings, but working with the differences of the 
    ratings from certain means.
    The four following strategies can be specified.
    - 'zero'. Zero values are subtracted from the rating matrix values. Basically, the matrix remains as it is.
    - 'items_means'. The items means (i.e. the means of the columns) are subtracted from the rating matrix values. Basically,
       each rate is subtracted with the average rate given to that movie.
       The purpose of this operation is to have a more balanced view of the ratings from an items point of view. Indeed, the 
       less popular movies have lower ratings w.r.t. the popular movies, and so they will have much lower similarity scores,
       even in cases in which they are more appropriate.
    - 'users_means'. The users means (i.e. the means of the rows) are subtracted from the rating matrix values.  Basically,
       each rate is subtracted with the average rate given by that user.
       The purpose of this operation is to have a more balanced view of the ratings from an users point of view. Indeed, there
       can be generous users which are more prone to give high rates, and greedy users which instead give often low rates.
       In simple terms, a 3 rating given by a generous user has not the same importance of a 3 rating given by a greedy one: 
       the former is less important and significative.
       Therefore, we want to normalize with respect to the users average rating.
    - 'items_users_means'. The means between the items means and the user means are subtracted from the rating matrix values.
       Basically,  each rate is subtracted with the average between the average rate given to that movie and the average rate 
       given by that user.
       The purpose of this operation is to normalize w.r.t. both items and users.
    
    The last important point is about the test set. Our approach is the following.
    A random subset of the ratings is taken and considered as test set: the proportion of this test set is given in input by
    the user (i.e. `test_prop`). 
    These test set ratings are 'masked out', in the sense that they are treated as missing values (handled in the same way 
    of all the others missing values, as specified above).
    Of course, the true test set ratings are kept and returned to the user, because they will be necessary for evaluation
    purposes.    

    On the whole, the rating matrix cells which are considered as missing values are both the original missing ratings and the
    test set cells.

    Parameters
    ----------
    df_movies : pd.DataFrame
        Movies dataframe
    df_ratings : pd.DataFrame
        Ratings dataframe
    n_users : int
        Number of users. The users ids are integer in the interval [0..n_users]
    fill_na : str, optional
        Strategy for filling the missing values, by default 'zero'.
        It must be a value among ['zero','items_means','users_means','items_users_means']
    subtract_mean : str, optional
        Strategy for subtracting the values in the rating matrix, by default 'zero'.
        It must be a value among ['zero','items_means','users_means','items_users_means']
    test_prop : float, optional
        Proportion of the test set, by default 0.0
    random_seed : int, optional
        Random seed used for finding the random test set sample, by default 44

    Returns
    -------
    rating_matrix : np.ndarray
        Rating matrix. It either contains the actual ratings or the ratings after subtraction by certain means.
    no_rating_mask : np.ndarray
        Boolean mask with the same shape of the rating matrix. It tells us the cells which didn't have any rate at the 
        beginning (original missing values).
    test_dict : dict
        Dictionary containing the information regarding the test set. It contains the following keys.
        - 'test_mask' : np.ndarray
            Boolean mask with the same shape of the rating matrix. It tells us the cells which are part of the test set.
        - 'test_target' : np.ndarray
            Monodimensional array containing the true test set ratings.
    means_dict : dict
        Dictionary containing the information regarding the means. It contains the following keys.
        - 'items_means' : np.ndarray
            Bidimensional array with the same shape of the rating matrix, containing the items means. This implies that each 
            column contains all equal values.
        - 'users_means' : np.ndarray
            Bidimensional array with the same shape of the rating matrix, containing the users means. So, each row contains
            all equal values.
        - 'items_users_means' : np.ndarray
            Bidimensional array with the same shape of the rating matrix, containing the average between the items and users 
            means. 
    """
    random.seed(random_seed)

    # Filling the matrix with the ratings
    rating_matrix = -1*np.ones(shape=(n_users,df_movies.shape[0]))
    for u in range(n_users):
        user_ratings_entries = df_ratings[df_ratings['userId']==u]
        moviesIds = user_ratings_entries['movieId']
        ratings = user_ratings_entries['rating']
        rating_matrix[u, moviesIds] = ratings

    # Mask of the matrix cells which do not have any rating
    no_rating_mask = rating_matrix < 0
    # We set Nan
    rating_matrix[no_rating_mask] = np.nan

    # Now, we mask some of the matrix cells: they are the test set cells.
    # We randomly select `test_prop` cells (among the ones which have a rate, i.e. ~no_rating_mask), and we mask out these
    # cells: this means that they are set as Nan.
    # Of course, we have to save the true original ratings, for evaluation purposes
    test_mask, test_target = None, None
    if test_prop>0:
        n_ratings = df_ratings.shape[0]
        test_size = int(n_ratings*test_prop)
        condition = np.zeros(shape=(n_ratings,)).astype(bool)
        condition[np.array(random.sample(range(n_ratings), test_size))] = True
        rating_matrix_copy = rating_matrix.copy()
        rating_matrix_copy[~no_rating_mask] = np.where(condition, -1*np.ones(shape=(n_ratings,)), rating_matrix[~no_rating_mask])
        # Mask for the test set cells
        test_mask = rating_matrix_copy<0
        # True original test set ratings
        test_target = rating_matrix[test_mask]
        # We set to Nan the test set cells
        rating_matrix[test_mask] = np.nan 
    if test_mask is None:
        test_mask = np.zeros(shape=rating_matrix.shape).astype(bool)

    # Mask for the Nan cells. A cell is Nan if either it originally has no rating or if it is a test set cell. These two 
    # conditions are exclusive: or one or the other.
    nan_mask = np.isnan(rating_matrix)

    # Means of the items (i.e. one mean for each column). In computing the means, we of course do not condier the Nan
    items_means = np.mean(rating_matrix, axis=0, where=~nan_mask)
    # We transform that in order to have the same shape of the rating matrix
    items_means = np.tile(items_means, (rating_matrix.shape[0],1))
    # If one item has no values (i.e. the mean is None) we replace it with the global mean
    items_means[np.isnan(items_means)] = np.mean(rating_matrix, where=~np.isnan(rating_matrix))

    # Means of the users (i.e. one mean for each row). In computing the means, we of course do not condier the Nan
    users_means = np.mean(rating_matrix, axis=1, where=~nan_mask)
    # We transform that in order to have the same shape of the rating matrix
    users_means = np.tile(users_means, (rating_matrix.shape[1],1)).T
    # If one user has no values (i.e. the mean is None) we replace it with the global mean
    users_means[np.isnan(users_means)] = np.mean(rating_matrix, where=~np.isnan(rating_matrix))

    # Matrix with same shape of the rating matrix, containing, for each cell, the mean between the item mean and the user mean
    items_users_means = (users_means+items_means)/2

    # Fill the Nan cells
    if fill_na=='zero':
        rating_matrix[nan_mask] = 0
    elif fill_na=='items_means':
        rating_matrix[nan_mask] = items_means[nan_mask]
    elif fill_na=='users_means':
        rating_matrix[nan_mask] = users_means[nan_mask]
    elif fill_na=='items_users_means':
        rating_matrix[nan_mask] = items_users_means[nan_mask]
    else:
        raise ValueError('`fill_na`: the value must be among ["zero","items_means","users_means","items_users_means"]')
    
    # optionally, subtract a certain mean from each cell
    if subtract_mean=='zero': 
        rating_matrix = rating_matrix
    elif subtract_mean=='items_means': 
        rating_matrix = rating_matrix - items_means
    elif subtract_mean=='users_means': 
        rating_matrix = rating_matrix - users_means
    elif subtract_mean=='items_users_means': 
        rating_matrix = rating_matrix - items_users_means
    else:
        raise ValueError('`subtract_mean`: the value must be among ["zero","items_means","users_means","items_users_means"]')

    # Preparing the output dictionaries
    test_dict = {'test_mask': test_mask, 'test_target':test_target}
    means_dict = {'items_means':items_means, 'users_means':users_means, 'items_users_means':items_users_means}

    return rating_matrix, no_rating_mask, test_dict, means_dict
