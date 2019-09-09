# Import relevant libraries
import os
import implicit
import pandas as pd
import numpy as np
import random
import scipy.sparse as sparse
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import warnings

# Implicit recommends disabling internal multi-threading by OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'


# This make_train function creates training and test sets for use with ALS models, like recommendation systems.
# ratings argument is the original user-item matrix and the pct_test argument is the percent of user-item
# interactions to mask in the training set for later comparison with the test set.
def make_train(ratings, pct_test = 0.2):

    # Make a copy of the original set to be the test set.
    test_set = ratings.copy()

    # Store the test set as a binary preference matrix.
    test_set[test_set != 0] = 1

    # Make a copy of the original data we can alter as our training set.
    training_set = ratings.copy()

    # Find the indices in the ratings data where an interaction exists.
    nonzero_inds = training_set.nonzero()

    # Zip these pairs together of user,item index into list.
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))

    # Set the random seed to zero for reproducibility.
    random.seed(0)

    # Round the number of samples needed to the nearest integer.
    num_samples = int(np.ceil(pct_test * len(nonzero_pairs)))

    # Sample a random number of user-item pairs without replacement.
    samples = random.sample(nonzero_pairs, num_samples)

    # Get the user row indices.
    user_inds = [index[0] for index in samples]

    # Get the item column indices.
    item_inds = [index[1] for index in samples]

    # Assign all of the randomly chosen user-item pairs to zero.
    training_set[user_inds, item_inds] = 0

    # Remove zeros in sparse array storage after update to save space.
    training_set.eliminate_zeros()

    # Output the unique list of user rows that were masked.
    return training_set, test_set, list(set(user_inds))

# Function for calculating AUC to evaluate model performance
def auc_score(predictions, test):
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)

# Function for calculating AUC for each user in training set who had at least one item masked
# It will also calculate AUC for the most popular items for all users as a baseline.
def calc_mean_auc(training_set, masked_users, predictions, test_set):

    # Initialize list to store AUC for each user who had an item masked in training set.
    store_auc = []

    # Initialize list to store AUC for popular items.
    popularity_auc = []

    # Sum user-item interactions to find most popular items.
    pop_items = np.array(test_set.sum(axis=0)).reshape(-1)
    item_vecs = predictions[1]

    # Iterate through all masked users.
    for user in masked_users:
        training_row = training_set[user, :].toarray().reshape(-1)

        # Find games where there is no interaction.
        zero_inds = np.where(training_row == 0)

        # Get predicted values based on user-item vectors.
        user_vec = predictions[0][user, :]
        pred = user_vec.dot(item_vecs).toarray()[0, zero_inds].reshape(-1)

        # Get only the items that were originally zero.
        # Select all predicted game plays for users who originally had no interactions.
        actual = test_set[user, :].toarray()[0, zero_inds].reshape(-1)

        # Select the binary yes/no interaction pairs from the original full data
        # that align with the same pairs in training.
        pop = pop_items[zero_inds]  # Get the item popularity for our chosen items.
        store_auc.append(auc_score(pred, actual))  # Calculate AUC for the given user and store.
        popularity_auc.append(auc_score(pop, actual))  # Calculate AUC using most popular and score.

    # Return the mean AUC rounded to three decimal places for both test and popularity benchmark.
    return float('%.3f' % np.mean(store_auc)), float('%.3f' % np.mean(popularity_auc))

# Retrieve games played by a specific user in the training set.
def get_games_played(user_id, train_set, users_list, games_list, game_id_map):

    user_ind = np.where(users_list == user_id)[0][0]  # Returns the index row of the user_id.
    played_ind = train_set[user_ind, :].nonzero()[1]  # Get column indices of played games.
    game_codes = games_list[played_ind]  # Get the game_ids for the user's played games.
    return game_id_map.loc[game_id_map.game_id.isin(game_codes)]


def recommend_games(user_id, train_set, user_vecs, item_vecs, user_list, game_list, game_id_map, num_items=10):

    user_ind = np.where(user_list == user_id)[0][0]  # Returns the index row of the user_id.
    pref_vec = train_set[user_ind, :].toarray()  # Get the ratings from the training set matrix.
    pref_vec = pref_vec.reshape(-1) + 1  # Add 1 to everything, so that unplayed games become equal to 1.
    pref_vec[pref_vec > 1] = 0  # Make played games equal to 0.
    rec_vector = user_vecs[user_id, :].dot(item_vecs.T)  # Get dot product of user vector and all item vectors.
    # Scale this recommendation vector between 0 and 1
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1, 1))[:, 0]
    recommend_vector = pref_vec * rec_vector_scaled
    # Games already played have their recommendation multiplied by zero.
    game_idx = np.argsort(recommend_vector)[::-1][:num_items]  # Sort the indices of the items into order
    # of best recommendations
    rec_list = []  # start empty list to store items
    for index in game_idx:
        game_id = game_list[index]
        rec_list.append([game_id, game_id_map.game_title.loc[game_id_map.game_id == game_id].iloc[0]])
        # Append our descriptions to the list
    game_ids = [game[0] for game in rec_list]
    game_titles = [game[1] for game in rec_list]
    final_frame = pd.DataFrame({'game_id': game_ids, 'game_title': game_titles})  # Create a dataframe
    return final_frame[['game_id', 'game_title']]  # Switch order of columns around

# Import data.
raw = pd.read_csv("steam-200k.csv", header=None, usecols=[0,1,2,3], \
    names=['user_id', 'game_title', 'behavior_name', 'value'])

# Filter dataframe for only play (behavior_code = 2) behavior.
df = raw[raw.behavior_name == 'purchase'].loc[:, ['user_id', 'game_title', 'value']]
df = df.groupby(['user_id', 'game_title'], as_index=False)['value'].agg('sum')

# Assign integers to user_id.
new_user_id = dict([(y, x+1) for x, y in enumerate(sorted(set(df.user_id)))])

# Assign integers to game_title.
new_game_id = dict([(y, x+1) for x, y in enumerate(sorted(set(df.game_title)))])

# Create new dataframe with coded user_id and game_id.
df_play = pd.DataFrame()
df_play['user_id'] = df['user_id'].map(new_user_id)
df_play['game_id'] = df['game_title'].map(new_game_id)
df_play['value'] = raw['value']

# Build sparse matrix usable for modeling.
df_play_wide = df_play.pivot(index='user_id', columns='game_id', values='value')
df_play_wide = df_play_wide.fillna(0)
df_play_wide_array = np.array(df_play_wide)
df_play_sparse = sparse.csr_matrix(df_play_wide_array)

# Create training and test sets; data must be in sparse matrix format.
play_train, play_test, masked_users = make_train(df_play_sparse, pct_test=0.2)

# Initialize implicit ALS model.
alpha = 15
play_train_alpha = (play_train * alpha).astype('double')
model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)
model.fit(play_train_alpha.T)
user_vecs = model.user_factors
item_vecs = model.item_factors

# Train the model on a sparse matrix of item/user/confidence weights.
#model.fit(df_play_sparse)

# Calculate AUC of model.
calc_mean_auc(play_train, masked_users,
              [sparse.csr_matrix(user_vecs), sparse.csr_matrix(item_vecs.T)], play_test)

# Convert sparse matrix to Compressed Sparse Row format.
#user_games = play_sparse_matrix.tocsr()

# Recommend games for a user.
#recommendations = model.recommend(1, user_games)

# Create lists of unique users and games.
users = list(np.sort(df_play.user_id.unique()))
games = list(df_play.game_id.unique())
#plays = list(df_play.value)

#rows = df_play.user_id.astype('category', categories=users).cat.codes

# Convert lists to arrays for processing.
users_array = np.array(users)
games_array = np.array(games)

# Convert new_game_id map from dict to dataframe for using with recommend_games function.
id_game_dict = {str(v): k for k, v in new_game_id.items()}
id_game_series = pd.Series(id_game_dict, name='game_title')
id_game_series.name = 'game_id'
id_game_df = pd.Series.to_frame(id_game_series, name='game_title')
id_game_df = id_game_df.reset_index()
id_game_df.index = id_game_df.index.astype(int)
id_game_df = id_game_df.rename(columns={'index': 'game_id'})

# Create list of users who've played MapleStory
lookup_game = 'MapleStory'
lookup_game_id = id_game_df[id_game_df.game_title==lookup_game].game_id.astype(int).values[0]
lookup_game_users = df_play.user_id[df_play.game_id==lookup_game_id].unique()

# Create lists of users who have or haven't played MapleStory
ms_users = lookup_game_users
non_ms_users = np.setdiff1d(users, ms_users)

# Get recommended games for
#test = recommend_games(1, play_train, user_vecs, item_vecs, users_array, games_array, id_game_df, num_items=10)

pot_ms_users = []

for user in non_ms_users:
    temp_recs_list = model.recommend(user-1, df_play_sparse, 10, filter_already_liked_items=True)
    temp_recs_df = pd.DataFrame(temp_recs_list, columns=['game_id', 'score'])
    if sum(temp_recs_df.game_id == 2646) > 0:
        pot_ms_users.append(user+1)