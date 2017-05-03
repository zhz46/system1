import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    item_input = sys.argv[1]
    purchase_input = sys.argv[2]

    # read item profiles
    cb = pd.read_csv(item_input)
    cb.columns = ['item_id', 'feature_list']

    # remove obs that has non-digit item_id
    cb = cb[cb.item_id.map(lambda x: x.isdigit())]

    # use tf-idf to extract features
    item_df = tfidf_extract(cb)
    items_unique = item_df['item_id'].unique()

    # read purchase data
    df = pd.read_csv(purchase_input)
    df.columns = ['qty', 'item_id', 'guest_id', 'purchase_date']
    df = df[['guest_id', 'item_id', 'qty']]

    # drop missing value, negative qty and non-digit obs
    df = df.dropna()
    df = df[df.qty > 0]
    df = df[df.guest_id.map(lambda x: x.isdigit()) & df.item_id.map(lambda x: x.isdigit())]

    # drop items that are not in item profiles
    df['guest_id'] = df['guest_id'].astype(int)
    df['item_id'] = df['item_id'].astype(int)
    df = df[df['item_id'].isin(items_unique)].reset_index(drop=True)
    items = list(np.sort(df.item_id.unique()))

    # inner join item profiles and purchase history
    all = pd.merge(df, item_df, on='item_id', how='inner')

    # construct user profiles
    guest_df = all.drop(['qty', 'item_id'], axis=1).groupby('guest_id').mean()

    # construct item profiles
    item_df = item_df[item_df['item_id'].isin(items)]
    item_df.index = item_df['item_id']
    item_df = item_df.drop('item_id', axis=1)
    item_df = item_df.sort_index()

    # calculate guest, item similarity matrix, using 0.6 as threshold for initialization
    cold_start = cold_start_extract(guest_df, item_df,threshold=0.6)
    cold_start.to_csv("cold_start.csv", index=False)


# use tf-idf to extract features
def tfidf_extract(data):
    tf = TfidfVectorizer(analyzer='word', min_df=0)
    tfidf_matrix = tf.fit_transform(data['feature_list'])
    feature_names = tf.get_feature_names()
    item_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    item_df = item_df.drop('null', axis=1)
    item_df['item_id'] = data['item_id']
    item_df = item_df.dropna()
    item_df['item_id'] = item_df['item_id'].astype(int)
    return item_df


def cold_start_extract(guest_df, item_df, threshold=0.7):
    # calculate similarity matrix for all guest, item pairs
    cross_similarity = cosine_similarity(guest_df, item_df)
    row_ind, col_ind = np.where(cross_similarity > threshold)
    guest_start = guest_df.index[row_ind]
    item_start = item_df.index[col_ind]
    cold_start = pd.DataFrame({'guest_id': guest_start, 'item_id': item_start, 'qty': 1})
    return cold_start


if __name__ == "__main__":
    main()





