import sys
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import random
import implicit
from scipy.stats import rankdata


def main():
    cold_start_input = sys.argv[1]
    purchase_input = sys.argv[2]

    # read guest start, item start
    cold_start = pd.read_csv(cold_start_input)

    # read purchase data
    df = pd.read_csv(purchase_input)
    df.columns = ['qty', 'item_id', 'guest_id', 'purchase_date']
    df = df[['guest_id', 'item_id', 'qty']]

    # drop missing value and negative qty
    df = df.dropna()
    df = df[df.qty > 0]
    df = df[df.guest_id.map(lambda x: x.isdigit()) & df.item_id.map(lambda x: x.isdigit())]

    # merge two df
    df['guest_id'] = df['guest_id'].astype(int)
    df['item_id'] = df['item_id'].astype(int)
    df = df.append(cold_start)

    # drop new users and new items
    item_count = df.item_id.value_counts()
    old_item = item_count.index[item_count > 1]
    guest_count = df.guest_id.value_counts()
    old_guest = guest_count.index[guest_count >= 5]
    df = df[df.guest_id.isin(old_guest) & df.item_id.isin(old_item)]

    # construct utility matrix
    guests = list(np.sort(df.guest_id.unique()))
    items = list(np.sort(df.item_id.unique()))
    quantity = list(df.qty)
    rows = df.guest_id.astype('category', categories=guests).cat.codes
    cols = df.item_id.astype('category', categories=items).cat.codes
    utility_mat = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(guests), len(items)))

    # check sparsity
    sparsity = 100 * (1 - 1.0 * len(df) / (utility_mat.shape[0] * utility_mat.shape[1]))

    # split training and testing data
    train_set, test_index = train_test_split(utility_mat, 0.1)

    # run ALS for implicit feedback to generate hidden features
    alpha = 100
    guest_feature, item_feature = implicit.alternating_least_squares((train_set * alpha).astype('double'),
                                                                     factors=10, regularization=0.1, iterations=50)

    # collect predicted values
    predict_matrix = guest_feature.dot(item_feature.T)

    # evaluate performance using average rank
    hidden_rank, all_rank = average_rank(predict_matrix, test_index, rows, cols)
    print(hidden_rank, all_rank)


# hidden part of data for testing
def train_test_split(utility_mat, test_pencent=0.2):
    training_set = utility_mat.copy()
    nonzero_inds = training_set.nonzero()
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))
    num_samples = int(np.ceil(test_pencent * len(nonzero_pairs)))
    samples = random.sample(nonzero_pairs, num_samples)
    user_inds = [index[0] for index in samples]
    item_inds = [index[1] for index in samples]
    training_set[user_inds, item_inds] = 0
    training_set.eliminate_zeros()
    return training_set, samples


def average_rank(predict_mat, test_index, rows_index, cols_index):
    # generate item rank percents for each guest
    rank_array = np.apply_along_axis(lambda row: 1 - (rankdata(row) - 1) / (len(row) - 1),
                                     axis=1, arr=predict_mat)
    # extract guest and item index
    guest_inds = [index[0] for index in test_index]
    item_inds = [index[1] for index in test_index]

    # calculate average rank for testing set and all set
    hidden_rank = np.sum(rank_array[guest_inds, item_inds]) / len(guest_inds)
    all_rank = np.sum(rank_array[rows_index, cols_index]) / len(rows_index)
    return hidden_rank, all_rank

if __name__ == "__main__":
    main()

