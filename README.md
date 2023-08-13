# Recommender-System-FML

### Akshay Srinivasan

## Introduction

This is a project completed for NYU'S Fundamentals of Machine Learning. The goal of this project is to address the challenging task of predicting movie ratings based on an extensive dataset encompassing more than 400,000 users and 5,000 movies. The prediction of movie ratings holds immense significance in the realm of recommendation systems, facilitating diverse applications such as movie recommendation engines, personalized content suggestions, and user profiling.

## Data Preparation

Data was extracted from the data.txt file containing movie ratings into a DataFrame with columns like movie_id, user_id, rating, and date. Data types were optimized for efficiency. A separate file, movie_titles.csv, had movie details but was not merged with the main dataset as movies were already identifiable by movie_id. The data.txt file contained 400k users' ratings for about 5k movies. Python functions were developed for data processing, including checks for missing user and movie IDs.

## Data Pre-processing

The user-item rating data was transformed into a sparse matrix with user_ids as rows, movie_ids as columns, and ratings as values. This transformation aids memory efficiency and simplifies feature engineering. The matrix was derived from 2,649,430 unique users and 2,115 unique movies. Zero ratings were removed before constructing the matrix using the SciPy library's csr_matrix function. A function to view initial matrix entries was also included.

## Data Exploration

A thorough analysis of movie ratings distribution was conducted. A horizontal bar chart displayed the distribution of rating scores, revealing that most ratings were centered around the 2 to 3 range. Statistical analysis of the mean and mode ratings for each user was performed. The mean rating showed a normal distribution, suggesting balanced rating preferences. The mode rating, representing the most common score given by a user, had noticeable peaks at different rating levels, emphasizing the varied user rating tendencies and the importance of a diverse recommendation system.

## Baseline Models
Baseline models predict movie ratings using aggregate statistics to set the foundation for advanced models:

#### Global Mean: 
Calculates the average of all non-zero ratings and uses it as the prediction. RMSE: 1.289.
#### User Mean and Median: 
Predicts future ratings using the mean and median ratings from each user. RMSEs are 1.119 (mean) and 1.195 (median) respectively.
#### Movie Mean and Median: 
Predicts future ratings based on each movie's mean and median ratings. RMSEs are 1.092 (mean) and 1.127 (median) respectively.

These baselines establish biases for each user and item. Further models add deviations from these predictions using different criteria.

## Matrix Factorization Model
This collaborative filtering method fills missing entries in user-item matrices by breaking them down into two lower-rank matrices, often using Singular Value Decomposition (SVD). It identifies latent user preferences and item characteristics. The dot product of latent vectors approximates the user's rating for an item.
To train, the alternating least squares (ALS) approach is employed. It transforms the problem into a Quadratic Convex Problem (QCP) and optimizes the user and item matrices iteratively. The goal is to minimize the difference between estimated and actual ratings, using L2 norms of user and item vectors as regularization against overfitting.

Enhancements include:
- Bias terms: Considering user and item-specific biases and a global average rating.
- Confidence levels: Introducing a confidence weight for each rating.

The final prediction formula is: r_ui hat = c_ui * (µ + b_u + b_i + p_u.T * q_i). Performance evaluation showed an RMSE of 1.266, which is slightly inferior to the baseline models.

## Singular Value Decomposition (SVD) Model:
The SVD model, a matrix factorization method, decomposes the user-item rating matrix into three lower-dimension matrices that represent latent features. Using the Surprise library, data is preprocessed and divided for training and testing. A parameter grid identifies optimal hyperparameters through cross-validation with the GridSearchCV class, focusing on RMSE. Once optimal hyperparameters are chosen, the model is trained using stochastic gradient descent (SGD) and evaluated with a test set. The SVD model's RMSE is 0.981, outperforming baseline models and the matrix factorization model.

## Model Evaluation and Performance Analysis:
Models are evaluated with RMSE:

1. Global Mean Rating: RMSE of 1.277.
2. User Mean Rating: RMSE of 1.123.
3. User Median Rating: RMSE of 1.206.
4. Movie Mean Rating: RMSE of 1.116.
5. Movie Median Rating: RMSE of 1.158.
6. Matrix Factorization: RMSE of 1.272.
7. SVD: RMSE of 1.058.
8. SVD w/ Grid Search: RMSE of 0.985, the best performing model.

In essence, the SVD with Grid Search offers the most accurate predictions. Combining different models might enhance predictive accuracy.

## Further Data Analysis:
Users are segmented into:
- Cinema Connoisseurs: Top-tier users who've rated over 213 movies, totaling 23,447 users.
- Selective Appraisers: Lower-tier users who've rated up to 2 movies, totaling 12,516 users.
- Avid Users: Those rating between 2 and 213 movies, representing the largest group at 421,635 users.

These segments offer insights into diverse user rating behavior, aiding in understanding user engagement with movies.

## Conclusion:

The research was focused on predicting movie ratings using a vast dataset to enhance recommendation system efficiency. Initial steps included processing and exploring the dataset to grasp the distribution and trends of movie ratings. While baseline models set a foundation, they lacked personalization, evident from the Global Mean model's consistent yet generalized predictions. The User and Movie Mean and Median models aimed at personalization by focusing on individual user and movie data, which aided in decreasing the error rate.

The Matrix Factorization model's complexity did not necessarily lead to superior performance, but the Singular Value Decomposition (SVD) model stood out, particularly when integrated with Grid Search, showcasing its ability to predict movie ratings with high accuracy.

A segmentation of the user base provided deeper insights into distinct user behaviors. "Cinema Connoisseurs", "Selective Appraisers", and "Avid Users" were categories that highlighted diverse movie rating habits among users.

To conclude, the SVD model with Grid Search was the standout performer in terms of prediction accuracy. Nevertheless, the research underscores the value of every model in providing unique insights, suggesting that combining different models could enhance prediction precision.
