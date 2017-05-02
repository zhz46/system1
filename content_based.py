import pandas as pd


# read item profiles
cb = pd.read_csv("items.csv")
cb.columns = ['item_id','feature_list']

# generate feature dict
all = []
for i in xrange(len(cb)):
    a = {}
    features = cb.feature_list.ix[i].split()
    for j in features:
        a[j] = a.get(j, 0) + 1
    all.append(a)

# create item profiles dataframe
from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse=False)
X = v.fit_transform(all)
feature_names = v.get_feature_names()
feature_df = pd.DataFrame(X, columns=feature_names)
