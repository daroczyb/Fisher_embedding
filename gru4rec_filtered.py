import sys
import os
sys.path.insert(0, "./gru4rec/")
import pandas as pd
import numpy as np
import theano
import theano.gpuarray
theano.gpuarray.use('cuda0')
from gru4rec import GRU4Rec
from pathlib import Path


train_file = ""
test_file = ""
content_file = ""
log_file = "logs/gru4rec.csv"

epochs = 4

implicit_threshold = 0

data_train = pd.read_csv(
    train_file,
    sep=";",
    quotechar="|",
    header=None,
    names=['user', 'item', 'rating', 'time'],
    index_col=False
)
content = pd.read_csv(content_file, sep=' ', names=['item', 'category', 'category_name'])
data_train = data_train[data_train.item.isin(content.item.unique())]
data_train = data_train[data_train.rating >= implicit_threshold].sort_values('time')
data_gru_train = data_train.rename(columns={'user': 'SessionId', 'item': 'ItemId'})[['SessionId', 'ItemId']]
data_gru_train['Time'] = range(data_gru_train.shape[0])

data_train_items = data_train.item.unique()

data_test = pd.read_csv(
    test_file,
    sep=";",
    quotechar="|",
    header=None,
    names=['user', 'item', 'rating', 'time'],
    index_col=False
)
data_test = data_test[(data_test.rating >= implicit_threshold) & (data_test.item.isin(data_train_items))].sort_values('time')
data_gru_test = data_test.rename(columns={'user': 'SessionId', 'item': 'ItemId'})[['SessionId', 'ItemId']]
data_gru_test['Time'] = range(data_gru_test.shape[0])

items = data_train.item.unique()

params = dict(
    loss='bpr-max',
    final_act='elu-0.5',
    hidden_act='tanh',
    layers=[100],
    adapt='adagrad',
    n_epochs=epochs,
    batch_size=32,
    dropout_p_embed=0,
    dropout_p_hidden=0,
    learning_rate=0.05,
    momentum=0.3,
    n_sample=2048,
    sample_alpha=0,
    bpreg=1,
    constrained_embedding=False,
    embedding=100,
)
gru = GRU4Rec(**params)
gru.fit(data_gru_train)

item_ranks = []
eval_log = []
for u, i_s in data_test.groupby('user').item.apply(list).reset_index().values:
    r199 = np.random.choice(items, (len(i_s) - 1, 199))
    pred_items = np.hstack([np.array(i_s[1:]).reshape(-1, 1), r199])
    ranks = []
    for i1, rs in zip(i_s, pred_items):
        preds = gru.predict_next_batch([u], [i1], rs, batch=1).values
        ranks.append(np.sum(preds[0] > preds[1:]) + np.sum(preds[0] == preds[1:]) / 2)
        eval_log.append((u, i1, rs[0], ranks[-1]))
    item_ranks.append(ranks)

all_ranks = [i for j in item_ranks for i in j]
pd.DataFrame.from_records(eval_log, columns=['user', 'item1', 'item2', 'rank']).to_csv(log_file, index=False)

print(params)
print(dict(
    implicit_threshold=implicit_threshold,
    log_file=log_file
))
print(np.array(all_ranks).mean())
out_embed = gru.Wy.eval()
np.save(log_file, np.asarray(out_embed))
gru.itemidmap.to_frame().to_csv(log_file + ".itemidmap.csv")
