import random

import numpy as np
import matplotlib.pyplot as plt

from .preprocessing import build_rating_matrix


def predict_ratings(u, s, vh, k):
    """Predict the ratings by using the approximated LSA space. 

    Basically, we are applying the truncated SVD. We are reconstructing back the rating matrix, but using a smaller rank `k`.

    Parameters
    ----------
    u : np.ndarray
        Bidimensional array, obtained from the SVD, by default None.
        It's the SVD matrix related to the words. It has dimensions (n_users,d), where d=min{n_users,n_movies}. So, the rows
        correspond to the users.
    s : np.ndarray
        Monodimensional array, containing the singular values, sorted in descending order.
    vh : np.ndarray
        Bidimensional array, obtained from the SVD.
        It's the SVD matrix related to the movies. It has dimensions (d,n_movies), where d=min{n_users,n_movies}. So, the 
        columns correspond to the movies.
    k : int
        Level of approximation for the LSA: k-rank approximation. Basically, new number of dimensions.

    Returns
    -------
    rating_matrix_pred : np.array
        Predicted (i.e. reconstructed rating matrix), using the truncated SVD.
    """
    uk = u[:,:k]
    sk = s[:k]
    vhk = vh[:k,:]

    usk = np.sqrt(sk) * uk 
    vhsk = np.reshape(np.sqrt(sk),newshape=(k,1)) * vhk

    rating_matrix_pred = np.matmul(usk,vhsk)

    return rating_matrix_pred


def evaluate(target, pred, mask_no_evaluate=None, metric='rmse'):
    """Evaluate the given predictions w.r.t. the given target.

    Basically, we want to measure the goodness of our predicted ratings w.r.t. the true ones.
    The predicted/true ratings can either be the actual ones or the ones obtained after having subtracted certain means.

    `target` and `pred` can have whathever number of dimensions, the only important thing is that they have the exact same
    shape.

    Parameters
    ----------
    target : np.array
        True ratings (either actual or after subtraction by certain means).
    pred : np.array
        Predicted ratings (either actual or after subtraction by certain means).
    mask_no_evaluate : np.array, optional
        Boolean mask (same shape of `target` and `pred`) tellings us which are the positions of `target` and `pred` to 
        consider, by default None.
        If None, all positions are taken into account.
    metric : str, optional
        Evaluation metric, by default 'rmse'.
        It must be a value in ['rmse','mae']

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    float
        Error measure
    """
    if metric=='rmse':
        if mask_no_evaluate is None:
            return np.sqrt(np.mean(np.square(target-pred)))
        else:
            return np.sqrt(np.mean(np.square(target[~mask_no_evaluate]-pred[~mask_no_evaluate])))
    elif metric=='mae':
        if mask_no_evaluate is None:
            return np.mean(np.abs(target-pred))
        else:
            return np.mean(np.abs(target[~mask_no_evaluate]-pred[~mask_no_evaluate]))
    else:
        raise ValueError('`metric` must be either "rmse" or "mae"')



def rebuild_rating_matrix(rating_matrix, means, nan_mask=None):
    """Rebuild the actual rating matrix.

    The current rating matrix can contain the ratings after subtraction by certain means. We want to rebuild back the actual 
    ratings in [10..5].

    Parameters
    ----------
    rating_matrix : np.array
        Rating matrix, containing the ratings subtracted with certain means.
    means : np.array
        matrix of the same shape of the rating matrix, containing the means values subtracted from the true ratings.
    nan_mask : np.array, optional
        Boolean mask tellings us which are the original missing values cells, by default None.
        This is used in order to set these cells to nan, as in the original rating matrix.

    Returns
    -------
    rating_matrix_rebuilt
        Rebuilt original rating matrix
    """
    rating_matrix_rebuilt = rating_matrix + means
    if nan_mask is not None:
        rating_matrix_rebuilt[nan_mask] = np.nan
    return rating_matrix_rebuilt



def _find_actual_means(means_dict, subtract_mean):
    """Return the `means` matrix which has been used in the beginning for subtracting the original ratings in the rating 
       matrix.
    """
    items_means, users_means, items_users_means = means_dict['items_means'], means_dict['users_means'], means_dict['items_users_means']
    if subtract_mean=='zero':
        return np.zeros(shape=(users_means.shape[0],users_means.shape[1]))
    elif subtract_mean=='items_means':
        return items_means
    elif subtract_mean=='users_means':
        return users_means
    elif subtract_mean=='items_users_means':
        return items_users_means
    else:
        raise ValueError('`subtract_mean`: the value must be among ["zero","items_means","users_means","items_users_means"]')



def compute_train_test_errors(df_movies, df_ratings, n_users, k_range=(0,200), fill_na='zero', subtract_mean='zero', 
                              test_prop=0.2, random_seed=44, metric='rmse', cv_splits=1, plot_train=True, plot_test=True,
                              verbose=True):
    """Compute the list of training and test errors for the specified range of different k values. 

    Basically, we obtain the training and test error for each possible 'k' value.
    Optionally, a plot of these scores is made.

    A particular rating strategy is specified, which is the combination of a `fill_na` (i.e. a strategy for silling the missing 
    values) and a `subtract_mean` (i.e. a strategy for subtracting the values in the rating matrix).

    A cross-validation approach is used. `cv_splits` different training-test couples are generated and, for each 'k', their 
    training-test scores are aggregated, in order to compute the single couple train_err-test_err for that 'k'.
    Basically, not a single splitting training-test is c0nsidered, but `cv_splits` different ones.

    Parameters
    ----------
    df_movies : pd.DataFrame
        Movies dataframe
    df_ratings : pd.DataFrame
        Ratings dataframe
    n_users : int
        Number of users
    k_range : tuple, optional
        Tuple containing the lower and upper bounds for the range of the k values, by default (0,200)
    fill_na : str, optional
        Strategy for filling the missing values, by default 'zero'.
        It must be a value among ['zero','items_means','users_means','items_users_means']
    subtract_mean : str, optional
        Strategy for subtracting the values in the rating matrix, by default 'zero'.
        It must be a value among ['zero','items_means','users_means','items_users_means']
    test_prop : float, optional
        Test set proportion, by default 0.2
    random_seed : int, optional
        Random seed for the test set random samples, by default 44
    metric : str, optional
        Metric for computing the error scores, by default 'rmse'
    cv_splits : int, optional
        Number of cross validation splits, by default 1
    plot_train : bool, optional
        Whether to plot the train errors or not, by default True
    plot_test : bool, optional
        Whether to plot the test errors or not, by default True
    verbose : bool, optional
        Whether to print the best score and the best 'k', by default True

    Returns
    -------
    train_err_list : list
    test_err_list : list
    """
    train_err_list = []
    test_err_list = []

    random.seed(random_seed)
    random_seeds_splits = np.random.randint(low=0, high=10000, size=(cv_splits,))

    # List of dictionaries, one for each cross-validation split, containing the training-test split and all the other useful
    # information
    splits_dictionaries = []
    for random_seed in random_seeds_splits:
        rating_matrix, no_rating_mask, test_dict, means_dict = build_rating_matrix(df_movies, df_ratings, n_users, 
                                                                                fill_na=fill_na, subtract_mean=subtract_mean, 
                                                                                test_prop=test_prop, random_seed=random_seed)
        test_mask, test_target = test_dict['test_mask'], test_dict['test_target']
        means = _find_actual_means(means_dict, subtract_mean)
        nan_mask = np.logical_or(no_rating_mask, test_mask)

        u, s, vh = np.linalg.svd(rating_matrix, full_matrices=False)

        splits_dictionaries.append({
            'rating_matrix':rating_matrix,
            'no_rating_mask':no_rating_mask,
            'test_mask':test_mask,
            'test_target':test_target,
            'nan_mask':nan_mask,
            'means':means,
            'u':u,
            's':s,
            'vh':vh
        })

    # Iterate over all k
    for k in range(k_range[0],k_range[1]):
        train_err_splits = [] 
        test_err_splits = [] 
        # Iterate over all cv splits
        for split_dictionary in splits_dictionaries:   
            rating_matrix, test_mask, test_target, nan_mask, means, u, s, vh = (split_dictionary['rating_matrix'], 
                                                                                split_dictionary['test_mask'], 
                                                                                split_dictionary['test_target'], 
                                                                                split_dictionary['nan_mask'],
                                                                                split_dictionary['means'],
                                                                                split_dictionary['u'],
                                                                                split_dictionary['s'],
                                                                                split_dictionary['vh'])

            rating_matrix_pred = predict_ratings(u, s, vh, k=k)

            train_err_split = evaluate(rating_matrix, rating_matrix_pred, mask_no_evaluate=nan_mask, metric=metric)
            train_err_splits.append(train_err_split)                
    
            test_pred = (rating_matrix_pred+means)[test_mask]
            test_err_split = evaluate(test_target, test_pred, metric=metric)
            test_err_splits.append(test_err_split)

        # Aggregate the errors obtained from the different splits
        train_err = np.mean(train_err_splits)
        test_err = np.mean(test_err_splits)

        # Append these errors
        train_err_list.append(train_err)
        test_err_list.append(test_err)

    if plot_train or plot_test:
        plt.figure(figsize=(9,7))
    if plot_train:
        plt.plot(train_err_list, label='train error')
    if plot_test:
        plt.plot(test_err_list, label='test error')
    if plot_train or plot_test:
        plt.grid()
        plt.title('Errors')
        plt.xlabel('k')
        plt.ylabel(f'{metric}')
        plt.legend()
        plt.show()

    if verbose:
        print(f'min err: {min(test_err_list)}; best k: {list(range(k_range[0],k_range[1]))[np.argmin(test_err_list)]}')

    return train_err_list, test_err_list
        


def compare_rating_strategies(df_movies, df_ratings, n_users, rating_strategies, k_range=(0,200), test_prop=0.2, 
                              random_seed=44, metric='rmse', cv_splits=1):
    """Compare different rating strategies.

    A rating strategy is the combination of a `fill_na` (i.e. a strategy for silling the missing values) and a `subtract_mean`
    (i.e. a strategy for subtracting the values in the rating matrix).

    The different strategies are compared by measuring the test errors among all the different 'k' values specified with `k_range`.
    For each strategy and for each 'k', an error score is computed.
    The test error plot of each rating strategy is made, for having a visualization of the comparison.
    In the end, the best strategy with the best score and the best 'k' are printed in the console.

    This function is implemented using the `compute_train_test_errors` function. 
    As exaplained in the docstring of that function, a cross-validation approach is followed for computing the test score 
    for each strategy for each 'k'. Basically, not a single splitting training-test is c0nsidered, but `cv_splits` different
    ones.

    Parameters
    ----------
    df_movies : pd.DataFrame
        Movies dataframe
    df_ratings : pd.DataFrame
        Ratings dataframe
    n_users : int
        Number of users
    rating_strategies : list of str
        List of different rating strategies to take into account.
        A rating strategy is represented as a string formatted as '{fill_na}/{subtract_mean}', containing both the 'fill_na'
        and the 'subtract_mean' strategies.
    k_range : tuple, optional
        Tuple containing the lower and upper bounds for the range of the k values, by default (0,200)
    test_prop : float, optional
        Test set proportion, by default 0.2
    random_seed : int, optional
        Random seed for the test set random samples, by default 44
    metric : str, optional
        Metric for computing the error scores, by default 'rmse'
    cv_splits : int, optional
        Number of cross validation splits, by default 1
    """
    plt.figure(figsize=(9,7))
    best_strategy = None 
    best_k = None 
    best_err = None
    for rating_strategy_name in rating_strategies:
        fill_na, subtract_mean = [(s+'_means' if s!='zero' else s) for s in rating_strategy_name.split('/')]
        train_err_list, test_err_list = compute_train_test_errors(df_movies, df_ratings, n_users, k_range=k_range, 
                                                                  fill_na=fill_na, subtract_mean=subtract_mean, 
                                                                  test_prop=test_prop, random_seed=random_seed, 
                                                                  metric=metric, cv_splits=cv_splits, plot_train=False, 
                                                                  plot_test=False, verbose=False)

        if (best_strategy is None) or min(test_err_list)<best_err:
            best_strategy = rating_strategy_name
            best_k = list(range(k_range[0],k_range[1]))[np.argmin(test_err_list)]
            best_err = min(test_err_list)
        
        plt.plot(test_err_list, label=rating_strategy_name)

    plt.xlabel('k')
    plt.ylabel(f'{metric}')
    plt.title('Comparison of different rating strategies')
    plt.grid()
    plt.legend()

    print(f'Best strategy: {best_strategy}')
    print(f'Best k: {best_k}')
    print(f'Best error: {best_err}')


