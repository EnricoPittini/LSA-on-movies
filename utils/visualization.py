import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Iterable, Tuple, List
from .similarities import find_most_interesting_words



def plot_movies(vh : np.ndarray, s : np.ndarray, dimensions : Tuple[int] = (0,1), labels : Iterable[str] = None, 
                subset : np.ndarray = None, normalize : bool = False, k : int = 100):
    """Plot the movies onto the two specified dimensions of the LSA space.

    The given movies are encoded as column vectors in the matrix `vh`. Optionally, we can consider only some of the movies, 
    and not all of them (we can consider only a subset of the columns in `vh`).

    This function can be used both in the content-based and collaborative contexts. 

    Parameters
    ----------
    vh : np.ndarray
        Bidimensional array, obtained from the SVD.
        It's the SVD matrix related to the movies. It has dimensions (d,n_movies), where d=min{n_words,n_movies}. So, the 
        columns correspond to the movies.
    s : np.ndarray
        Monodimensional array, containing the singular values, sorted in descending order.
    dimensions : Tuple[int], optional
        Two LSA dimensions onto which plot the movies vectors, by default (0,1)
    labels : Iterable[str], optional
        List of string labels to assign to each point in the plot, by default None.
        Basically, these labels should be the movies titles.
    subset : np.ndarray, optional
        Array of integers, containing the indicies of the movies in which we want to focus on, by default None
    normalize : bool, optional
        Whether to normalize or not the movies vectors, by default False
    k : int, optional
        Level of approximation for the LSA: k-rank approximation, by default 100. Basically, new number of dimensions.
        This is used only if `normalize` is True.
    """
    if subset is None:
        subset = list(range(vh.shape[1]))
    vh = vh[:,subset]

    vh_s = np.reshape(s, newshape=(s.shape[0],1)) * vh
    if normalize:   
        vh_sk = vh_s[:k,:]
        vh_sk_normalized = vh_sk/np.sqrt(np.sum(np.square(vh_sk), axis=0))
        vh_s = vh_sk_normalized

    dim_x = dimensions[0]
    dim_y = dimensions[1]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(vh_s[dim_x,:], vh_s[dim_y,:])
    if labels is not None:
        labels = labels[subset]
        for i in range(vh.shape[1]):
            txt = labels[i]
            ax.annotate(txt, (vh_s[dim_x,i], vh_s[dim_y,i]))
    ax.scatter(0, 0, c='black', marker='s', label='origin')
    ax.set_xlabel(f'LSA dimension {dim_x}')
    ax.set_ylabel(f'LSA dimension {dim_y}')
    ax.set_title(f'LSA movies vectors along dimensions {dim_x} and {dim_y}')
    plt.legend()
    ax.grid()



########## FUNCTIONS FOR THE CONTENT-BASED PART


def plot_words(u : np.ndarray, s : np.ndarray, dimensions : Tuple[int] = (0,1), labels : Iterable[int] = None, 
               subset : np.ndarray = None, normalize : bool = False, k : int = 100):
    """Plot the words onto the two specified dimensions of the LSA space.

    The given words are encoded as row vectors in the matrix `u`. Optionally, we can consider only some of the words, 
    and not all of them (we can consider only a subset of the rows in `u`).

    Parameters
    ----------
    u : np.ndarray
        Bidimensional array, obtained from the SVD.
        It's the SVD matrix related to the words. It has dimensions (n_words,d), where d=min{n_words,n_movies}. So, the rows
        correspond to the words.
    s : np.ndarray
        Monodimensional array, containing the singular values, sorted in descending order.
    dimensions : Tuple[int], optional
        Two LSA dimensions onto which plot the words vectors, by default (0,1)
    labels : Iterable[str], optional
        List of string labels to assign to each point in the plot, by default None.
        Basically, these labels should be the words themselves.
    subset : np.ndarray, optional
        Array of integers, containing the indicies of the words in which we want to focus on, by default None
    normalize : bool, optional
        Whether to normalize or not the words vectors, by default False
    k : int, optional
        Level of approximation for the LSA: k-rank approximation, by default 100. Basically, new number of dimensions.
        This is used only if `normalize` is True.
    """
    if subset is None:
        subset = list(range(u.shape[1]))
    u = u[subset,:]

    u_s = u * np.reshape(s, newshape=(1,s.shape[0]))
    if normalize:
        u_sk = u_s[:,:k]
        u_sk_normalized = u_sk/np.reshape(np.sqrt(np.sum(np.square(u_sk), axis=1)),newshape=(u_sk.shape[0],1))
        u_s = u_sk_normalized

    dim_x = dimensions[0]
    dim_y = dimensions[1]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(u_s[:,dim_x], u_s[:,dim_y])
    if labels is not None:
        labels = labels[subset]
        for i in range(u.shape[0]):
            txt = labels[i]
            ax.annotate(txt, (u_s[i,dim_x], u_s[i,dim_y]))
    ax.scatter(0, 0, c='black', marker='s', label='origin')
    ax.set_xlabel(f'LSA dimension {dim_x}')
    ax.set_ylabel(f'LSA dimension {dim_y}')
    ax.set_title(f'LSA words vectors along dimensions {dim_x} and {dim_y}')
    plt.legend()
    ax.grid()



def plot_genres_analysis(vh : np.ndarray, s : np.ndarray, df : pd.DataFrame, genres : List[str], 
                         dimensions : Tuple[int] = (0,1), words : List[str] = None, plot_most_relevant_words : bool = False, 
                         n : int = 1, u : np.ndarray = None, voc : np.ndarray = None, normalize : bool = False, 
                         k : int = 100, delete_intersection : bool = True):
    """Make a plot for analyzing the given genres of interest.
    
    Basically, we plot the movies onto the two specified dimensions of the LSA space, coloring the points in different ways 
    according to their genre.
    The given movies are encoded as column vectors in the matrix `vh`. 

    Optionally, a list of words to be plotted can be specified. In this way, we can see the relation of these words with the
    genres of interest.

    Still optionally, we can ask to plot, for each genre, the `n` most relevant words. Namely, the `n` words which are the 
    most similar to the movies with that genre.
    To be more specific, for each word, its average cosine similarity computed w.r.t. all the movies with that genre is 
    calculated, and then the `n` words with biggest score are taken.
    In the plot, also the mean cosine similairity of each of these words is shown.

    Parameters
    ----------
    vh : np.ndarray
        Bidimensional array, obtained from the SVD.
        It's the SVD matrix related to the movies. It has dimensions (d,n_movies), where d=min{n_words,n_movies}. So, the 
        columns correspond to the movies.
    s : np.ndarray
        Monodimensional array, containing the singular values, sorted in descending order.
    df : pd.DataFrame
        Input dataframe
    genres : List[str]
        List of the genres of interest
    dimensions : Tuple[int], optional
        Two LSA dimensions onto which plot the words vectors, by default (0,1), by default (0,1)
    words : List[str], optional
        List of words to plot alongside the genres movies, by default None
    plot_most_relevant_words : bool, optional
        Whether to plot the most relevant words for each genre, by default False
    n : int, optional
        Number of most relevant words to plot for each genre, by default 1.
        This is used only if `plot_most_relevant_words` is True
    u : np.ndarray
        Bidimensional array, obtained from the SVD, by default None.
        It's the SVD matrix related to the words. It has dimensions (n_words,d), where d=min{n_words,n_movies}. So, the rows
        correspond to the words.
        It must be specified only if either `word` or `plot_most_relevant_words` are specified.
    voc : np.ndarray, optional
        Vocabulary, namely mapping from integer ids into words, by default None.
        It must be specified only if either `word` or `plot_most_relevant_words` are specified.
    normalize : bool, optional
        Whether to normalize or not the movies vectors and the words vectors, by default False
    k : int, optional
        Level of approximation for the LSA: k-rank approximation, by default 100. Basically, new number of dimensions.
        This is used only if either `plot_most_relevant_words` or `normalize` are True.
    delete_intersection : bool, optional
        Whether to delete or not the movies which belong to more than one of the specified genres, by default True
    """
    colors_indices = list(mcolors.TABLEAU_COLORS)

    if words is not None and (voc is None or u is None):
        raise ValueError('`words` is not None but either `u` or `voc` or both of them are None')
    if plot_most_relevant_words and (voc is None or u is None):
        raise ValueError('`plot_most_relevant_words` is True but either `u` or `voc` or both of them are None')
    if words is not None or plot_most_relevant_words:
        u_s = u * np.reshape(s, newshape=(1,s.shape[0]))
        if normalize:
            u_sk = u_s[:,:k]
            u_sk_normalized = u_sk/np.reshape(np.sqrt(np.sum(np.square(u_sk), axis=1)),newshape=(u_sk.shape[0],1))
            u_s = u_sk_normalized

    vh_s = np.reshape(s, newshape=(s.shape[0],1)) * vh
    if normalize:   
        vh_sk = vh_s[:k,:]
        vh_sk_normalized = vh_sk/np.sqrt(np.sum(np.square(vh_sk), axis=0))
        vh_s = vh_sk_normalized

    dim_x = dimensions[0]
    dim_y = dimensions[1]
    fig, ax = plt.subplots(figsize=(10,10))
    intersection_mask = df['genres'].map(lambda s: len(set(genres).intersection(s))>=2).to_numpy()
    for i, genre in enumerate(genres):
        color = mcolors.TABLEAU_COLORS[colors_indices[i]]
        genre_mask = df['genres'].map(lambda s: genre in s).to_numpy()
        if delete_intersection:
            genre_mask = np.logical_and(genre_mask, np.logical_not(intersection_mask))
        ax.scatter(vh_s[dim_x,genre_mask], vh_s[dim_y,genre_mask], label=f'{genre} movies', c=color)
        if plot_most_relevant_words:
            subset = np.arange(vh.shape[1])[genre_mask]
            selected_words_ids, mean_cos_similarities = find_most_interesting_words(vh, s, u, subset=subset, n=n, k=k, 
                                                                                    normalize=normalize)
            ax.scatter(u_s[selected_words_ids,dim_x], u_s[selected_words_ids,dim_y], c=color, marker='*', #edgecolors='black',
                       s=100, label=f'{genre} words')
            for i, word_id in enumerate(selected_words_ids):
                txt = voc[word_id]
                mean_cos_similarity = mean_cos_similarities[i]
                ax.annotate(txt + f' {mean_cos_similarity:.2f}', (u_s[word_id,dim_x], u_s[word_id,dim_y])) 

    if words is not None:
        word2id = {word:id for id,word in enumerate(voc)}
        words_array = np.zeros(shape=(len(words),u.shape[1] if not normalize else k))
        for i, word in enumerate(words):  
            word_id = word2id[word]           
            words_array[i,:] = u_s[word_id,:]
        
        plt.scatter(words_array[:,dim_x], words_array[:,dim_y], c='red', marker='*', s=100, label='Specified words')
        for i in range(len(words)):
            txt = words[i]
            ax.annotate(txt, (words_array[i,dim_x], words_array[i,dim_y]))

    ax.scatter(0, 0, c='black', marker='s', label='origin')
    ax.set_xlabel(f'LSA dimension {dim_x}')
    ax.set_ylabel(f'LSA dimension {dim_y}')
    ax.set_title(f'Genre analysis in the LSA space along dimensions {dim_x} and {dim_y}')
    ax.grid()
    ax.legend()



########## FUNCTIONS FOR THE COLLABORATIVE PART

def plot_genres_analysis_coll(vh : np.ndarray, s : np.ndarray, df_movies : pd.DataFrame, genres : List[str],
                             dimensions : Tuple[int] = (0,1),  normalize : bool = False, k : int = 100, 
                             delete_intersection : bool = True):
    """Make a plot for analyzing the given genres of interest.
    
    Basically, we plot the movies onto the two specified dimensions of the LSA space, coloring the points in different ways 
    according to their genre.
    The given movies are encoded as column vectors in the matrix `vh`. 

    Parameters
    ----------
    vh : np.ndarray
        Bidimensional array, obtained from the SVD.
        It's the SVD matrix related to the movies. It has dimensions (d,n_movies), where d=min{n_users,n_movies}. So, the 
        columns correspond to the movies.
    s : np.ndarray
        Monodimensional array, containing the singular values, sorted in descending order.
    df_movies : pd.DataFrame
        Input dataframe
    genres : List[str]
        List of the genres of interest
    dimensions : Tuple[int], optional
        Two LSA dimensions onto which plot the words vectors, by default (0,1), by default (0,1)
    normalize : bool, optional
        Whether to normalize or not the movies vectors, by default False
    k : int, optional
        Level of approximation for the LSA: k-rank approximation, by default 100. Basically, new number of dimensions.
        This is used only if either `plot_most_relevant_words` or `normalize` are True.
    delete_intersection : bool, optional
        Whether to delete or not the movies which belong to more than one of the specified genres, by default True
    """
    colors_indices = list(mcolors.TABLEAU_COLORS)

    vh_s = np.reshape(s, newshape=(s.shape[0],1)) * vh
    if normalize:   
        vh_sk = vh_s[:k,:]
        vh_sk_normalized = vh_sk/np.sqrt(np.sum(np.square(vh_sk), axis=0))
        vh_s = vh_sk_normalized

    dim_x = dimensions[0]
    dim_y = dimensions[1]
    fig, ax = plt.subplots(figsize=(10,10))
    intersection_mask = df_movies['genres'].map(lambda s: len(set(genres).intersection(s))>=2).to_numpy()
    for i, genre in enumerate(genres):
        color = mcolors.TABLEAU_COLORS[colors_indices[i]]
        genre_mask = df_movies['genres'].map(lambda s: genre in s).to_numpy()
        if delete_intersection:
            genre_mask = np.logical_and(genre_mask, np.logical_not(intersection_mask))
        ax.scatter(vh_s[dim_x,genre_mask], vh_s[dim_y,genre_mask], label=f'{genre} movies', c=color)

    ax.scatter(0, 0, c='black', marker='s', label='origin')
    ax.set_xlabel(f'LSA dimension {dim_x}')
    ax.set_ylabel(f'LSA dimension {dim_y}')
    ax.set_title(f'Genre analysis in the LSA space along dimensions {dim_x} and {dim_y}')
    ax.grid()
    ax.legend()

