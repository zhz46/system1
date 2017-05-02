import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# read item profiles
cb = pd.read_csv("items.csv")
cb.columns = ['item_id','feature_list']

# use tf-idf to extract features
tf = TfidfVectorizer(analyzer='word', min_df=0)
tfidf_matrix = tf.fit_transform(cb['feature_list'])
feature_names = tf.get_feature_names()
feature_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

feature_df['item_id'] = cb['item_id']
feature_df.drop('null',axis=1, inplace=True)
items_unique = feature_df['item_id'].unique()

# read purchase data
df = pd.read_csv("purchases.csv")
df.columns = ['qty', 'item_id', 'guest_id', 'purchase_date']
df = df.dropna().reset_index(drop=True)
df = df[df.qty > 0]
df = df[['guest_id', 'item_id', 'qty']]
df = df[df.guest_id.map(lambda x: x.isdigit()) & df.item_id.map(lambda x: x.isdigit())]

# drop items that are not in item profiles
df_short = df[df['item_id'].isin(items_unique)]

# join feature profiles and purchase history
all = pd.merge(df_short, feature_df, on='item_id', how='inner')

# construct user profiles
user_df = all.drop(['qty', 'item_id'], axis=1).groupby('guest_id').mean()

# construct item profiles
feature_df.index = feature_df['item_id']
item_df = feature_df.drop('item_id', axis=1)

# calculate similarity matrix for all user, item pairs
cross_similarity = cosine_similarity(user_df, item_df)