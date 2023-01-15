# Latent Semantic Analysis applied on movies

Project and Project Work in Text Mining.

In this work, we explore the *LSA (Latent Semantic Analysis)* applied on movies. In particular, we follow two different approaches.
1. First of all, we follow a *content-based* approach. Basically, we want to capture the similarities and relations between movies by exploiting their textual overviews. We compare the results achieved by a syntactic movel (i.e. the tf-idf model) with the semantic LSA model. We also make visualizations of the movies and words vectors onto two dimensions of the LSA, showing the ability of this powerful method to capture some semantic features of documents and words. In this first part, a subset of the [The Movie Database](https://www.themoviedb.org/) (TMDB) dataset has been used.
2. Then, we follow the *collaborative* approach. We don't have anymore any information about the content of the movies. We have only a database of rates given to the movies by a community of users. By exploiting this information, still in the LSA space, we can capture the similarities and relations between movies in a very poweful way. We are also able to build a recommender system, to predict the rates given by an user to a certain movie. In this second part, three different [MovieLens](https://grouplens.org/datasets/movielens/) datasets are used: small, 100k, 1M. On each of these three datasets, the recommender system is evaluated and compared with SOTA results and with the results achieved by a paper.

The implementation of LSA in both approaches is done from scratch, without using any external library other than the following basic Python libraries for matrix processing.

## Dependencies
- [NumPy](https://pypi.org/project/numpy/)
- [Matplotlib](https://pypi.org/project/matplotlib/)
- [Pandas](https://pypi.org/project/pandas/)

## Repository structure

    .
    ├── dataset    # It contains the dataset files                        
    ├── utils    # It contains the python files with useful functions
    ├── 1) content-based.ipynb   
    ├── 2) collaborative.ipynb 
    ├── .gitignore
    ├── LICENSE
    └── README.md

## Group members

|  Name           |  Surname  |     Email                           |    Username                                             |
| :-------------: | :-------: | :---------------------------------: | :-----------------------------------------------------: |
| Enrico          | Pittini   | `enrico.pittini@studio.unibo.it`    | [_EnricoPittini_](https://github.com/EnricoPittini)     |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
