import numpy as np
import pandas as pd
from colorama import Fore, Style
from .preprocessing import process_text, lemmatize_with_postag



def compute_cos_similarities(vector : np.ndarray, vectors : np.ndarray, vector_norm : float = None, 
                             vectors_norms : np.ndarray = None):
    """Compute the cosine similarities between the given vectors and each one of the column vectors in the given matrix.

    The specified `vector` is a row vector (d,), while the specified matrix `vectors` is (d,n): we want to compute the cosine
    similarity between the row vector and each one of the column vectors of the matrix.

    Parameters
    ----------
    vector : np.ndarray
        Input row vector
    vectors : np.ndarray
        Input matrix
    vector_norm : np.ndarray, optional
        Norm of the row vector, by default None
    vectors_norms : np.ndarray
        Norms of the column vectors of the matrix, by default None

    Returns
    -------
    np.ndarray
        Flat array containing the cosine similarities
    """
    if vector_norm is None:
        vector_norm = np.sqrt(np.sum(vector*vector))
    if vectors_norms is None:
        vectors_norms = np.sqrt(np.sum(vectors*vectors, axis=0))

    cos_similarities = np.sum(np.reshape(vector, newshape=(vector.shape[0],1))*vectors, axis=0)/(vector_norm*vectors_norms)

    return cos_similarities



########## FUNCTIONS FOR THE CONTENT-BASED PART


def _color_movie_text(movie1_vector : np.ndarray, movie2_vector : np.ndarray, movie2_text : str, voc : np.ndarray):
    """Given two movies, color the words in the text of the second movie based on their importance in the similarity with 
    the second movie.

    Basically, for each word in the text of the second movie, its impact in the cosine similarity with the first movie is 
    measured, and that word is colored accordingly.
    - If the word is not present in the vocabulary, it remains white.
    - If the word has no impact in the similarity, it is colored as green.
    - If the word has middle impact in the similarity, it is colored as yellow.
    - If the word has high impact in the similarity, it is colored as red.

    Parameters
    ----------
    movie1_vector : np.ndarray
        Vector representing the first movie
    movie2_vector : np.ndarray
        Vector representing the second movie
    movie2_text : str
        Text of the second movie
    voc : np.ndarray
        Vocabulary, i.e. mapping from integer ids into words

    Returns
    -------
    str
        Text of the second movie, after that each word has been colored
    """
    norm_movie_vector = np.sqrt(np.sum(np.square(movie1_vector)))
    norm_similarMovie_vector = np.sqrt(np.sum(np.square(movie2_vector)))
    
    words_importances = (movie1_vector * movie2_vector)/(norm_movie_vector*norm_similarMovie_vector)

    word2id = {word:id for id,word in enumerate(voc)}
    words_ids_inText = np.array(list(set([word2id[word] for word in movie2_text.split() if word in word2id])))
    words_importances_inText = words_importances[words_ids_inText]

    middle_importance = (max(words_importances_inText)-min(words_importances_inText))/2

    highImportance_color = Fore.RED  
    noEvaluate_color = Style.RESET_ALL  
    lowImportance_color = Fore.GREEN  
    middleImportance_color = Fore.YELLOW

    similarMovie_text_processed = ''
    for word in movie2_text.split():
        if word not in word2id:
            color = noEvaluate_color
        else:
            word_id = word2id[word]
            if words_importances[word_id]>=middle_importance:  # TODO: tune that?
                color = highImportance_color 
            elif words_importances[word_id]>0:
                color = middleImportance_color
            else: 
                color = lowImportance_color
        similarMovie_text_processed += f'{color}{word} '

    return similarMovie_text_processed + f'{Style.RESET_ALL}'


def _color_mostRelevantWords_cosSimilarity(movie1_vector : np.ndarray, movie2_vector : np.ndarray, voc : np.ndarray, n : int):
    """Given two movies, return the string containing the `n` vocabulary words with biggest impact in the similarity between
    these two movies.

    Each word is colored according to its level of impact in the cosine similarity between the two words: yellow if middle 
    impact, red if high impact.
    Only the words with at least some impact in the similarity are shown.

    Parameters
    ----------
    movie1_vector : np.ndarray
        Vector representing the first movie
    movie2_vector : np.ndarray
        Vector representing the second movie
    voc : np.ndarray
        Vocabulary, i.e. mapping from integer ids into words
    n : int
        Number of words to return

    Returns
    -------
    str
        String containing the `n` words with biggest impact in the cosine similarity between the two movies. Each word is 
        colored according to its level of impact.
    """
    highImportance_color = Fore.RED   
    lowImportance_color = Fore.GREEN  
    middleImportance_color = Fore.YELLOW

    norm_movie_vector = np.sqrt(np.sum(np.square(movie1_vector)))
    norm_similarMovie_vector = np.sqrt(np.sum(np.square(movie2_vector)))
    
    words_importances = (movie1_vector * movie2_vector)/(norm_movie_vector*norm_similarMovie_vector)

    middle_importance = (max(words_importances)-min(words_importances))/2

    most_relevant_words = voc[np.argsort(words_importances)[::-1]]
    most_relevant_words = most_relevant_words[np.sort(words_importances)[::-1]>0]
    most_relevant_words = most_relevant_words[:n]

    word2id = {word:id for id,word in enumerate(voc)}
    mostRelevantWords_colored_string = ''
    for word in most_relevant_words:
        word_id = word2id[word]
        if words_importances[word_id]>=middle_importance:  # TODO: tune that?
            color = highImportance_color 
        elif words_importances[word_id]>0:
            color = middleImportance_color
        else: 
            color = lowImportance_color
        mostRelevantWords_colored_string += f'{color}{word} '

    return mostRelevantWords_colored_string + f'{Style.RESET_ALL}'


def _print_legend():
    """Print the legend explaining the meaning of the colors for the words: 'white' means out of the vocabulary, 'green' means 
    no impact in the similarity; 'yellow' means middle impact; 'red' means high impact.
    """
    highImportance_color = Fore.RED  
    noEvaluate_color = Style.RESET_ALL  
    lowImportance_color = Fore.GREEN  
    middleImportance_color = Fore.YELLOW
    print('Legend')
    print('The words in the overview are colored in different ways, with the following meaning.')
    print(f'\t- {noEvaluate_color}[word]{Style.RESET_ALL}: word not present in the vocabulary')
    print(f'\t- {lowImportance_color}[word]{Style.RESET_ALL}: word with no impact in the similarity')
    print(f'\t- {middleImportance_color}[word]{Style.RESET_ALL}: word with middle impact in the similarity')
    print(f'\t- {highImportance_color}[word]{Style.RESET_ALL}: word with high impact in the similarity')
    print()


def compute_most_similar_movies_tfidf(movie_title : str, df : pd.DataFrame, text_col : str, tfidf : np.ndarray, 
                                      show_similar_movies : int = None, underline_words : bool = False, 
                                      show_most_relevant_words : int = None, voc : np.ndarray = None, 
                                      print_legend : bool = False):
    """Compute the most similar movies to the one given in input, according to tf-idf.

    It returns a data structure containing such movies. In addition, an intuitive and easy visualization of these most 
    similar movies is given.

    Regarding the visualization, the `show_similar_movies` most similar movies are shown.
    For each one of them, the following information is shown.
    1. Title of the movie.
    2. Similarity score (computed as cosine similarity)
    3. Original overview (i.e. not preprocessed)
    4. Preprocessed overview. Optionally, each word in this overview can be colored according to the level of impact of that
       word in the cosine similarity.
       - If the word is not present in the vocabulary, it remains white.
       - If the word has no impact in the similarity, it is colored as green.
       - If the word has middle impact in the similarity, it is colored as yellow.
       - If the word has high impact in the similarity, it is colored as red.
    5. List of the first `show_most_relevant_words` vocabulary words with highest impact in the cosine similarity, still 
       colored according to the level of impact of each word in the cosine similarity.
       Only the words with at least some impact in the similarity are shown, therefore only the yellow and red colors are 
       used.

    Remark: the first most similar movie is for sure the movie itself. This entry is not useless, because is shows which are
    the most important and relevant words. They are the words that the model thinks are the most representative and that it 
    will search in other movies for finding similarities. 

    Remark: since we are computing simply syntactic similarity (indeed, we are using tf-idf), the words shown at point 5 are 
    for sure words contained in the movie text, i.e. words shown at point 4.

    Parameters
    ----------
    movie_title : str
        Title of the movie of interest. The most similar movies to that movie are computed.
    df : pd.DataFrame
        Input dataframe.
    text_col : str
        Name of the column containing the textual data to processs
    tfidf : np.ndarray
        Bidimensional array, representing the tf-idf matrix
    show_similar_movies : int, optional
        Number of most similar movies to show, by default None
    underline_words : bool, optional
        Whether to color or not the words in the texts based on their impact on the similarity, by default False
    show_most_relevant_words : int, optional
        Number of most relevant words w.r.t. to the similarities to show, by default None
    voc : np.ndarray, optional
        Vocabulary, i.e. mapping from integer ids into words, by default None.
        It must be specified only if either `show_similar_movies` or `show_most_relevant_words` are True.
    print_legend : bool, optional
        Whether to print or not the legend describing the meaning of the different colors, by default False.
        This is used only if either `show_similar_movies` or `show_most_relevant_words` are True.

    Returns
    -------
    pd.DataFrame
        Data structure containing the most similar movies. Basically, it contains all movies, sorted in descending order by 
        their similarity.
        So, each row is a movie.
        The columns are the following: `index`, `title`, `similarity`, `text_col`, `original text_col`.
    """
    if underline_words and voc is None:
        raise ValueError('`underline_words` is True but `voc` is None')
    if show_most_relevant_words is not None and voc is None:
        raise ValueError('`show_most_relevant_words` is not None but `voc` is None')

    movie_row = df[df['title'].map(lambda t: movie_title==t)]
    if movie_row.shape[0]==0:
        raise ValueError(f'No movie with that title "{movie_title}"')
    elif movie_row.shape[0]>1:
        raise ValueError(f'Too many movies with that title "{movie_title}"')
    movie_id = movie_row.index[0]

    movie_vector = tfidf[:,movie_id]

    cos_similarities = compute_cos_similarities(vector=movie_vector, vectors=tfidf)

    df_most_similar = df.loc[np.argsort(cos_similarities)[::-1], ['title', text_col, f'original {text_col}']]
    df_most_similar['similarity'] = np.sort(cos_similarities)[::-1]
    df_most_similar = df_most_similar[~df_most_similar['similarity'].isna()]
    df_most_similar = df_most_similar.reset_index()

    if show_similar_movies is not None:
        print(f'Most similar films to "{movie_title}"')
        print()
        if (underline_words or show_most_relevant_words is not None) and print_legend:
            _print_legend()
        for i in range(show_similar_movies):
            print(f'{i}) Title: "{df_most_similar.loc[i,"title"]}", similarity: {df_most_similar.loc[i,"similarity"]:.2f}')
            print(f'\t- original {text_col}: {df_most_similar.loc[i,f"original {text_col}"]}')
            similarMovie_vector = tfidf[:,df_most_similar.loc[i,f"index"]]
            similarMovie_overview = df_most_similar.loc[i,f"{text_col}"]
            if underline_words:
                similarMovie_overview = _color_movie_text(movie_vector, similarMovie_vector, similarMovie_overview, voc=voc)
            print(f'\t- {text_col}: {similarMovie_overview}')
            if show_most_relevant_words is not None:
                mostRelevantWords_colored_string = _color_mostRelevantWords_cosSimilarity(movie_vector, similarMovie_vector, 
                                                                                        voc=voc, n=show_most_relevant_words)
                print(f'\t- Most relevant words: {mostRelevantWords_colored_string}')
            print()

    return df_most_similar


def compute_most_similar_movies_lsa(movie_title : str, df : pd.DataFrame, text_col : str, vh : np.ndarray, s : np.ndarray, 
                                    k : int, show_similar_movies : int = None, underline_words : bool = False, 
                                    show_most_relevant_words : int = None, u : np.ndarray = None, voc : np.ndarray = None,
                                    print_legend : bool = False):
    """Compute the most similar movies to the one given in input, according to latent semantic analysis (LSA).

    It returns a data structure containing such movies. In addition, an intuitive and easy visualization of these most 
    similar movies is given.

    Regarding the visualization, the `show_similar_movies` most similar movies are shown.
    For each one of them, the following information is shown.
    1. Title of the movie.
    2. Similarity score (computed as cosine similarity)
    3. Original overview (i.e. not preprocessed)
    4. Preprocessed overview. Optionally, each word in this overview can be colored according to the level of impact of that
       word in the cosine similarity.
       - If the word is not present in the vocabulary, it remains white.
       - If the word has no impact in the similarity, it is colored as green.
       - If the word has middle impact in the similarity, it is colored as yellow.
       - If the word has high impact in the similarity, it is colored as red.
    5. List of the first `show_most_relevant_words` vocabulary words with highest impact in the cosine similarity, still 
       colored according to the level of impact of each word in the cosine similarity.
       Only the words with at least some impact in the similarity are shown, therefore only the yellow and red colors are 
       used.

    Remark: the first most similar movie is for sure the movie itself. This entry is not useless, because is shows which are
    the most important and relevant words. They are the words that the model thinks are the most representative and that it 
    will search in other movies for finding similarities. 

    Parameters
    ----------
    movie_title : str
        Title of the movie of interest. The most similar movies to that movie are computed.
    df : pd.DataFrame
        Input dataframe.
    text_col : str
        Name of the column containing the textual data to processs
    vh : np.ndarray
        Bidimensional array, obtained from the SVD.
        It's the SVD matrix related to the movies. It has dimensions (d,n_movies), where d=min{n_words,n_movies}. So, the 
        columns correspond to the movies.
    s : np.ndarray
        Monodimensional array, containing the singular values, sorted in descending order.
    k : int
        Level of approximation for the LSA: k-rank approximation. Basically, new number of dimensions.
    show_similar_movies : int, optional
        Number of most similar movies to show, by default None
    underline_words : bool, optional
        Whether to color or not the words in the texts based on their impact on the similarity, by default False
    show_most_relevant_words : int, optional
        Number of most relevant words w.r.t. to the similarities to show, by default None
    u : np.ndarray, optional
        Bidimensional array, obtained from the SVD.
        It's the SVD matrix related to the words. It has dimensions (n_words,d), where d=min{n_words,n_movies}. So, the rows
        correspond to the words.
        It must be specified only if either `show_similar_movies` or `show_most_relevant_words` are True.
    voc : np.ndarray, optional
        Vocabulary, i.e. mapping from integer ids into words, by default None.
        It must be specified only if either `show_similar_movies` or `show_most_relevant_words` are True.
    print_legend : bool, optional
        Whether to print or not the legend describing the meaning of the different colors, by default False.
        This is used only if either `show_similar_movies` or `show_most_relevant_words` are True.

    Returns
    -------
    pd.DataFrame
        Data structure containing the most similar movies. Basically, it contains all movies, sorted in descending order by 
        their similarity.
        So, each row is a movie.
        The columns are the following: `index`, `title`, `similarity`, `text_col`, `original text_col`.
    """
    if underline_words and (u is None or voc is None):
        raise ValueError('`underline_words` is True but either `u` is None or `voc` is None or both of them are None')

    movie_row = df[df['title'].map(lambda t: movie_title==t)]
    if movie_row.shape[0]==0:
        raise ValueError(f'No movie with that title "{movie_title}"')
    elif movie_row.shape[0]>1:
        raise ValueError(f'Too many movies with that title "{movie_title}"')
    movie_id = movie_row.index[0]

    sk = s[:k]
    vhk = vh[:k,:]  

    movie_vector = sk*vhk[:,movie_id]
    movies_vectors = np.reshape(sk, newshape=(sk.shape[0],1))*vhk
    
    cos_similarities = compute_cos_similarities(vector=movie_vector, vectors=movies_vectors)

    df_most_similar = df.loc[np.argsort(cos_similarities)[::-1], ['title', text_col, f'original {text_col}']]
    df_most_similar['similarity'] = np.sort(cos_similarities)[::-1]
    df_most_similar = df_most_similar[~df_most_similar['similarity'].isna()]
    df_most_similar = df_most_similar.reset_index()

    if show_similar_movies is not None:
        if underline_words or show_most_relevant_words is not None:
            uk = u[:,:k]
            xk = np.matmul(uk * sk, vhk)
            movie_vector = xk[:,movie_id]
            if print_legend:
                _print_legend()
        print(f'Most similar films to "{movie_title}"')
        print()
        for i in range(show_similar_movies):
            print(f'{i}) Title: "{df_most_similar.loc[i,"title"]}", similarity: {df_most_similar.loc[i,"similarity"]:.2f}')
            print(f'\t- original {text_col}: {df_most_similar.loc[i,f"original {text_col}"]}')
            if underline_words or show_most_relevant_words is not None:
                similarMovie_vector = xk[:,df_most_similar.loc[i,f"index"]]
                similarMovie_overview = df_most_similar.loc[i,f"{text_col}"]
            if underline_words:
                similarMovie_overview = _color_movie_text(movie_vector, similarMovie_vector, similarMovie_overview, voc=voc)
            print(f'\t- {text_col}: {similarMovie_overview}')
            if show_most_relevant_words is not None:
                mostRelevantWords_colored_string = _color_mostRelevantWords_cosSimilarity(movie_vector, similarMovie_vector, 
                                                                            voc=voc, n=show_most_relevant_words)
                print(f'\t- Most relevant words: {mostRelevantWords_colored_string}')
            print()

    return df_most_similar


def find_most_interesting_words(vh : np.ndarray, s : np.ndarray, u : np.ndarray, k : int = 100, subset : np.ndarray = None,
                                n : int = 3, normalize : bool = False):
    """Find the `n` most interesting words with respect to the given movies, according to the latent semantic analysis (i.e. 
    LSA).

    The given movies are encoded as column vectors in the matrix `vh`; the words are encoded as row vectors in the matrix `u`. 
    Optionally, we can consider only some of the movies, and not all of them (we can consider only a subset of the columns 
    in `vh`). This can be useful, for example for focusing only to the movies with a certain genre.

    We want to find the `n` words which are the most related to the given movies. Namely, the words which are the most similar 
    to the given movies.
    To be more specific, for each word, its average cosine similarity computed w.r.t. all the specified movies is calculated.

    Parameters
    ----------
    vh : np.ndarray
        Bidimensional array, obtained from the SVD.
        It's the SVD matrix related to the movies. It has dimensions (d,n_movies), where d=min{n_words,n_movies}. So, the 
        columns correspond to the movies.
    s : np.ndarray
        Monodimensional array, containing the singular values, sorted in descending order.
    u : np.ndarray
        Bidimensional array, obtained from the SVD.
        It's the SVD matrix related to the words. It has dimensions (n_words,d), where d=min{n_words,n_movies}. So, the rows
        correspond to the words.
    k : int, optional
        Level of approximation for the LSA: k-rank approximation, by default 100. Basically, new number of dimensions.
    subset : np.ndarray, optional
        Array of integers, containing the indicies of the movies in which we want to focus on, by default None
    n : int, optional
        Number of words to retrieve, by default 3
    normalize : bool, optional
        Whether to normalize or not the movies vectors and the words vectors, by default False

    Returns
    -------
    selected_words_ids : np.ndarray
        Array containing the ids of the selected words
    mean_cos_similarities : np.ndarray
        Array containing the mean cosine similarities of the selected words w.r.t all the specified movies
    """
    sk = s[:k]

    if subset is None:
        subset = range(vh.shape[1])
    vh_k = vh[:k,subset]
    vh_ks = np.reshape(np.sqrt(sk), newshape=(sk.shape[0],1)) * vh_k
    if normalize:   
        vh_ks_normalized = vh_ks/np.sqrt(np.sum(np.square(vh_ks), axis=0))
        vh_ks = vh_ks_normalized

    u_k = u[:,:k]
    u_ks = u_k * np.reshape(np.sqrt(sk), newshape=(1,sk.shape[0]))
    if normalize:
        u_ks_normalized = u_ks/np.reshape(np.sqrt(np.sum(np.square(u_ks), axis=1)),newshape=(u_ks.shape[0],1))
        u_ks = u_ks_normalized

    vh_ks_norms = np.sqrt(np.sum(np.square(vh_ks),axis=0))
    u_ks_norms = np.sqrt(np.sum(np.square(u_ks),axis=1))

    mean_cos_similarities = np.zeros(shape=(u.shape[0],))
    for word_id in range(u_ks.shape[0]):
        cos_similarities = compute_cos_similarities(vector=u_ks[word_id,:], vectors=vh_ks, vector_norm=u_ks_norms[word_id],
                                                    vectors_norms=vh_ks_norms)
        cos_similarities = cos_similarities[~np.isnan(cos_similarities)]
        mean_cos_similarities[word_id] = np.mean(cos_similarities)
    
    selected_words_ids = np.argsort(mean_cos_similarities)[::-1]
    selected_words_ids = selected_words_ids[~np.isnan(mean_cos_similarities[selected_words_ids])]
    selected_words_ids = selected_words_ids[:n]

    return selected_words_ids, mean_cos_similarities[selected_words_ids]


def compute_mostSimilarMovies_tfidf_query(query_text : str, df : pd.DataFrame, text_col : str, tfidf : np.ndarray, 
                                          voc : np.ndarray, show_similar_movies : int = None, underline_words : bool = False, 
                                          show_most_relevant_words : int = None, print_legend : bool = False):
    """Compute the most similar movies to the textual query given in input, according to tf-idf.

    Very similar interface and code to the `compute_most_similar_movies_lsa` function.

    It returns a data structure containing such movies. In addition, an intuitive and easy visualization of these most 
    similar movies is given.

    Regarding the visualization, the `show_similar_movies` most similar movies are shown.
    For each one of them, the following information is shown.
    1. Title of the movie.
    2. Similarity score (computed as cosine similarity)
    3. Original overview (i.e. not preprocessed)
    4. Preprocessed overview. Optionally, each word in this overview can be colored according to the level of impact of that
       word in the cosine similarity.
       - If the word is not present in the vocabulary, it remains white.
       - If the word has no impact in the similarity, it is colored as green.
       - If the word has middle impact in the similarity, it is colored as yellow.
       - If the word has high impact in the similarity, it is colored as red.
    5. List of the first `show_most_relevant_words` vocabulary words with highest impact in the cosine similarity, still 
       colored according to the level of impact of each word in the cosine similarity.
       Only the words with at least some impact in the similarity are shown, therefore only the yellow and red colors are 
       used.

    Remark: the first most similar movie is the query itself, for keeping the same interface of the 
    `compute_most_similar_movies_tfidf` function. This entry shows us which are the query words that the model thinks are the
    most representative and that it will search in other movies for finding similarities. 

    Remark: since we are computing simply syntactic similarity (indeed, we are using tf-idf), the words shown at point 5 are 
    for sure words contained in the movie text, i.e. words shown at point 4.

    Parameters
    ----------
    query_text : str
        Input textual query
    df : pd.DataFrame
        Input dataframe.
    text_col : str
        Name of the column containing the textual data to processs
    tfidf : np.ndarray
        Bidimensional array, representing the tf-idf matrix
    voc : np.ndarray
        Vocabulary, i.e. mapping from integer ids into words
    show_similar_movies : int, optional
        Number of most similar movies to show, by default None
    underline_words : bool, optional
        Whether to color or not the words in the texts based on their impact on the similarity, by default False
    show_most_relevant_words : int, optional
        Number of most relevant words w.r.t. to the similarities to show, by default None    
    print_legend : bool, optional
        Whether to print or not the legend describing the meaning of the different colors, by default False.
        This is used only if either `show_similar_movies` or `show_most_relevant_words` are True.

    Returns
    -------
    pd.DataFrame
        Data structure containing the most similar movies. Basically, it contains all movies, sorted in descending order by 
        their similarity.
        So, each row is a movie.
        The columns are the following: `index`, `title`, `similarity`, `text_col`, `original text_col`.
    """

    # Process the query text
    query_text_p = process_text(query_text)
    query_text_p = lemmatize_with_postag(query_text_p)

    # Build the terms-docs vector representing the query text
    terms_docs_vector = np.zeros(shape=(len(voc), ))
    word2id = {word:id for id,word in enumerate(voc)}
    for word in query_text_p.split():
        if word in voc:
            terms_docs_vector[word2id[word]] += 1

    # Build the tf-idf vector representing the query text
    docs_freq = np.array([df[text_col].map(lambda s: voc[word_id] in s).sum() for word_id in range(len(voc))])
    tfidf_vector = np.log(terms_docs_vector+1) * np.log(df.shape[0]/docs_freq) 

    movie_vector = tfidf_vector

    # Appending that vector in the tf-idf matrix: new column vector
    tfidf = np.concatenate([tfidf, np.reshape(tfidf_vector, newshape=(len(voc),1))], axis=1)
    # Add the query in the df, in orther to compute also for the similarity with itself
    df_query = pd.DataFrame({'title': ['query'], f'{text_col}': [query_text_p], f'original {text_col}': [query_text]})
    df = pd.concat([df, df_query], ignore_index=True)

    cos_similarities = compute_cos_similarities(vector=movie_vector, vectors=tfidf)

    df_most_similar = df.loc[np.argsort(cos_similarities)[::-1], ['title', text_col, f'original {text_col}']]
    df_most_similar['similarity'] = np.sort(cos_similarities)[::-1]
    df_most_similar = df_most_similar[~df_most_similar['similarity'].isna()]
    df_most_similar = df_most_similar.reset_index()

    if show_similar_movies is not None:
        print(f'Most similar films to the given query')
        print()
        if (underline_words or show_most_relevant_words is not None) and print_legend:
            _print_legend()
        for i in range(show_similar_movies):
            print(f'{i}) Title: "{df_most_similar.loc[i,"title"]}", similarity: {df_most_similar.loc[i,"similarity"]:.2f}')
            print(f'\t- original {text_col}: {df_most_similar.loc[i,f"original {text_col}"]}')
            similarMovie_vector = tfidf[:,df_most_similar.loc[i,f"index"]]
            similarMovie_overview = df_most_similar.loc[i,f"{text_col}"]
            if underline_words:
                similarMovie_overview = _color_movie_text(movie_vector, similarMovie_vector, similarMovie_overview, voc=voc)
            print(f'\t- {text_col}: {similarMovie_overview}')
            if show_most_relevant_words is not None:
                mostRelevantWords_colored_string = _color_mostRelevantWords_cosSimilarity(movie_vector, similarMovie_vector, 
                                                                                        voc=voc, n=show_most_relevant_words)
                print(f'\t- Most relevant words: {mostRelevantWords_colored_string}')
            print()

    return df_most_similar


def compute_mostSimilarMovies_lsa_query(query_text : str, df : pd.DataFrame, text_col : str, vh : np.ndarray, s : np.ndarray, 
                                        u : np.ndarray, k : int, voc : np.ndarray, show_similar_movies : int = None, 
                                        underline_words : bool = False, show_most_relevant_words : int = None,   
                                        print_legend : bool = False):
    """Compute the most similar movies tothe textual query given in input, according to latent semantic analysis (LSA).

    Very similar interface and code to the `compute_most_similar_movies_tfidf` function.

    It returns a data structure containing such movies. In addition, an intuitive and easy visualization of these most 
    similar movies is given.

    Regarding the visualization, the `show_similar_movies` most similar movies are shown.
    For each one of them, the following information is shown.
    1. Title of the movie.
    2. Similarity score (computed as cosine similarity)
    3. Original overview (i.e. not preprocessed)
    4. Preprocessed overview. Optionally, each word in this overview can be colored according to the level of impact of that
       word in the cosine similarity.
       - If the word is not present in the vocabulary, it remains white.
       - If the word has no impact in the similarity, it is colored as green.
       - If the word has middle impact in the similarity, it is colored as yellow.
       - If the word has high impact in the similarity, it is colored as red.
    5. List of the first `show_most_relevant_words` vocabulary words with highest impact in the cosine similarity, still 
       colored according to the level of impact of each word in the cosine similarity.
       Only the words with at least some impact in the similarity are shown, therefore only the yellow and red colors are 
       used.

    Remark: the first most similar movie is the query itself, for keeping the same interface of the 
    `compute_most_similar_movies_lsa` function. This entry shows us which are the query words that the model thinks are the
    most representative and that it will search in other movies for finding similarities. 

    Parameters
    ----------
    movie_title : str
        Title of the movie of interest. The most similar movies to that movie are computed.
    df : pd.DataFrame
        Input dataframe.
    text_col : str
        Name of the column containing the textual data to processs
    vh : np.ndarray
        Bidimensional array, obtained from the SVD.
        It's the SVD matrix related to the movies. It has dimensions (d,n_movies), where d=min{n_words,n_movies}. So, the 
        columns correspond to the movies.
    s : np.ndarray
        Monodimensional array, containing the singular values, sorted in descending order.
    u : np.ndarray, optional
        Bidimensional array, obtained from the SVD.
        It's the SVD matrix related to the words. It has dimensions (n_words,d), where d=min{n_words,n_movies}. So, the rows
        correspond to the words.
    k : int
        Level of approximation for the LSA: k-rank approximation. Basically, new number of dimensions.
    voc : np.ndarray, optional
        Vocabulary, i.e. mapping from integer ids into words, by default None.
    show_similar_movies : int, optional
        Number of most similar movies to show, by default None
    underline_words : bool, optional
        Whether to color or not the words in the texts based on their impact on the similarity, by default False
    show_most_relevant_words : int, optional
        Number of most relevant words w.r.t. to the similarities to show, by default None
    print_legend : bool, optional
        Whether to print or not the legend describing the meaning of the different colors, by default False.
        This is used only if either `show_similar_movies` or `show_most_relevant_words` are True.

    Returns
    -------
    pd.DataFrame
        Data structure containing the most similar movies. Basically, it contains all movies, sorted in descending order by 
        their similarity.
        So, each row is a movie.
        The columns are the following: `index`, `title`, `similarity`, `text_col`, `original text_col`.
    """

    # Process the query text
    query_text_p = process_text(query_text)
    query_text_p = lemmatize_with_postag(query_text_p)

    # Build the terms-docs vector representing the query text
    terms_docs_vector = np.zeros(shape=(len(voc), ))
    word2id = {word:id for id,word in enumerate(voc)}
    for word in query_text_p.split():
        if word in voc:
            terms_docs_vector[word2id[word]] += 1

    # Build the tf-idf vector representing the query text
    docs_freq = np.array([df[text_col].map(lambda s: voc[word_id] in s).sum() for word_id in range(len(voc))])
    tfidf_vector = np.log(terms_docs_vector+1) * np.log(df.shape[0]/docs_freq) 

    sk = s[:k]
    vhk = vh[:k,:] 
    uk = u[:,:k] 

    # FOLD-IN of the query vector
    movie_vector = (1/sk) * np.matmul(uk.T,tfidf_vector) #(uk.T * tfidf_vector) 

    # Appending that vector in the vhk matrix: new column vector
    vhk = np.concatenate([vhk, np.reshape(movie_vector, newshape=(k,1))], axis=1)
    movies_vectors = np.reshape(sk, newshape=(sk.shape[0],1))*vhk
    # Add the query in the df, in orther to compute also for the similarity with itself
    df_query = pd.DataFrame({'title': ['query'], f'{text_col}': [query_text_p], f'original {text_col}': [query_text]})
    df = pd.concat([df, df_query], ignore_index=True)
    movie_id = df.shape[0]-1
    
    cos_similarities = compute_cos_similarities(vector=movie_vector, vectors=movies_vectors)

    df_most_similar = df.loc[np.argsort(cos_similarities)[::-1], ['title', text_col, f'original {text_col}']]
    df_most_similar['similarity'] = np.sort(cos_similarities)[::-1]
    df_most_similar = df_most_similar[~df_most_similar['similarity'].isna()]
    df_most_similar = df_most_similar.reset_index()

    if show_similar_movies is not None:
        if underline_words or show_most_relevant_words is not None:
            uk = u[:,:k]
            xk = np.matmul(uk * sk, vhk)
            movie_vector = xk[:,movie_id]
            if print_legend:
                _print_legend()
        print(f'Most similar films to the given query')
        print()
        for i in range(show_similar_movies):
            print(f'{i}) Title: "{df_most_similar.loc[i,"title"]}", similarity: {df_most_similar.loc[i,"similarity"]:.2f}')
            print(f'\t- original {text_col}: {df_most_similar.loc[i,f"original {text_col}"]}')
            if underline_words or show_most_relevant_words is not None:
                similarMovie_vector = xk[:,df_most_similar.loc[i,f"index"]]
                similarMovie_overview = df_most_similar.loc[i,f"{text_col}"]
            if underline_words:
                similarMovie_overview = _color_movie_text(movie_vector, similarMovie_vector, similarMovie_overview, voc=voc)
            print(f'\t- {text_col}: {similarMovie_overview}')
            if show_most_relevant_words is not None:
                mostRelevantWords_colored_string = _color_mostRelevantWords_cosSimilarity(movie_vector, similarMovie_vector, 
                                                                            voc=voc, n=show_most_relevant_words)
                print(f'\t- Most relevant words: {mostRelevantWords_colored_string}')
            print()

    return df_most_similar



########## FUNCTIONS FOR THE COLLABORATIVE PART

def compute_most_similar_movies_lsa_coll(movie_title : str, df_movies : pd.DataFrame, vh : np.ndarray, s : np.ndarray, 
                                         k : int):
    """Compute the most similar movies to the one given in input, according to latent semantic analysis (LSA) applied in the 
    collaborative context.

    Each movie is represented as a k-dimensional vector in the LSA space. The cosine similarity is used between these vectors
    for computing the similarity scores.
    
    It returns a data structure containing such movies.

    Parameters
    ----------
    movie_title : str
        Title of the movie
    df_movies : pd.DataFrame
        Movies dataframe
    vh : np.ndarray
        Bidimensional array, obtained from the SVD.
        It's the SVD matrix related to the movies. It has dimensions (d,n_movies), where d=min{n_users,n_movies}. So, the 
        columns correspond to the movies.
    s : np.ndarray
        Monodimensional array, containing the singular values, sorted in descending order.
    k : int
        Level of approximation for the LSA: k-rank approximation. Basically, new number of dimensions.

    Returns
    -------
    pd.DataFrame
        Data structure containing the most similar movies. Basically, it contains all movies, sorted in descending order by 
        their similarity.
        So, each row is a movie.
        The columns are the following: `movie index`, `title`, `similarity`.
    """
    movie_row = df_movies[df_movies['title'].map(lambda t: movie_title==t)]
    if movie_row.shape[0]==0:
        raise ValueError(f'No movie with that title "{movie_title}"')
    elif movie_row.shape[0]>1:
        raise ValueError(f'Too many movies with that title "{movie_title}"')
    movie_id = movie_row.index[0]

    sk = s[:k]
    vhk = vh[:k,:]  

    movie_vector = sk*vhk[:,movie_id]
    movies_vectors = np.reshape(sk, newshape=(sk.shape[0],1))*vhk
    
    cos_similarities = compute_cos_similarities(vector=movie_vector, vectors=movies_vectors)

    df_most_similar = df_movies.loc[np.argsort(cos_similarities)[::-1], ['title']]
    df_most_similar['similarity'] = np.sort(cos_similarities)[::-1]
    df_most_similar = df_most_similar[~df_most_similar['similarity'].isna()]
    df_most_similar = df_most_similar.reset_index()
    df_most_similar = df_most_similar.rename(columns={'index':'movie index'})

    return df_most_similar