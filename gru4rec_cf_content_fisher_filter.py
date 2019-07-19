import sys
import os
sys.path.insert(0, "./gru4rec/")
import pandas as pd
import numpy as np
import theano
import theano.gpuarray
theano.gpuarray.use('cuda0')
from gru4rec import GRU4Rec
import scipy.sparse as sp
from pathlib import Path


train_file = ""
test_file = ""
content_file = ""
epochs_cf, epochs_content, fisher_dim = 3, 4, 50

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

# selecting fisher_dim diverse pop items
pop_items = np.array(data_train.item.value_counts().index[:fisher_dim * 10])
uimat = sp.coo_matrix((np.ones(data_train.shape[0]), (data_train['item'].values, data_train['user'].values)))
uimat_csr = uimat.tocsr()
items_left = list(pop_items)
items_selected = list(np.random.choice(list(items_left), 1))
items_left.remove(items_selected[-1])
for i in range(fisher_dim):
    intersections = uimat_csr[items_left].dot(uimat_csr[items_selected].T.todense())
    unions = (-intersections) + \
        uimat_csr[items_selected].getnnz(axis=1).reshape(1, -1) +\
        uimat_csr[items_left].getnnz(axis=1).reshape(-1, 1)

    J = intersections / unions
    mean_dists = J.mean(axis=1)
    next_item = items_left[mean_dists.argmin()]
    items_selected.append(next_item)
    items_left.remove(next_item)

# calculating fisher embeddings
pop_items = np.array(items_selected)
uimat = sp.coo_matrix((np.ones(data_train.shape[0]), (data_train['item'].values, data_train['user'].values)))
uimat_csr = uimat.tocsr()
intersections = uimat_csr.dot(uimat_csr[pop_items].T.todense())
unions = (-intersections) + \
    uimat_csr[pop_items].getnnz(axis=1).reshape(1, -1) +\
    uimat_csr.getnnz(axis=1).reshape(-1, 1)
J = intersections / unions
fd = (J - J.mean(axis=0)) / J.std(axis=0)
preset_embeddings = fd

# selecting fisher_dim diverse pop items
fisher_dim = 200
pop_items = np.array(data_train.item.value_counts().index[:fisher_dim * 10])
uimat = sp.coo_matrix((np.ones(content.shape[0]), (content['item'].values, content['category'].values)))
uimat_csr = uimat.tocsr()
pop_items = pop_items[pop_items < uimat_csr.shape[0]]
items_left = list(pop_items)
items_selected = list(np.random.choice(list(items_left), 1))
items_left.remove(items_selected[-1])
for i in range(fisher_dim):
    intersections = uimat_csr[items_left].dot(uimat_csr[items_selected].T.todense())
    unions = (-intersections) + \
        uimat_csr[items_selected].getnnz(axis=1).reshape(1, -1) +\
        uimat_csr[items_left].getnnz(axis=1).reshape(-1, 1)

    J = intersections / unions
    mean_dists = J.mean(axis=1)
    next_item = items_left[mean_dists.argmin()]
    items_selected.append(next_item)
    items_left.remove(next_item)

# calculating fisher embeddings
pop_items = np.array(items_selected)
uimat = sp.coo_matrix((np.ones(content.shape[0]), (content['item'].values, content['category'].values)))
uimat_csr = uimat.tocsr()
intersections = uimat_csr.dot(uimat_csr[pop_items].T.todense())
unions = (-intersections) + \
    uimat_csr[pop_items].getnnz(axis=1).reshape(1, -1) +\
    uimat_csr.getnnz(axis=1).reshape(-1, 1)
J = intersections / unions
fd = (J - J.mean(axis=0)) / J.std(axis=0)
preset_content_embeddings = fd
preset_content_embeddings_ = np.where(~np.isnan(preset_content_embeddings), preset_content_embeddings, np.zeros_like(preset_content_embeddings))
preset_content_embeddings = np.zeros((preset_embeddings.shape[0], preset_content_embeddings_.shape[1]))
preset_content_embeddings[:preset_content_embeddings_.shape[0], :preset_content_embeddings_.shape[1]] = preset_content_embeddings_

params = dict(
    loss='bpr-max',
    final_act='elu-0.5',
    hidden_act='tanh',
    layers=[100],
    adapt='adagrad',
    n_epochs=epochs_cf,
    batch_size=32,
    dropout_p_embed=0,
    dropout_p_hidden=0.0,
    learning_rate=0.01,
    momentum=0.3,
    n_sample=2048,
    sample_alpha=0,
    bpreg=1,
    constrained_embedding=False,
    preset_embeddings=preset_embeddings,
    embedding_transform_matrix=True,
)
gru1 = GRU4Rec(**params)
gru1.fit(data_gru_train)

params = dict(
    loss='bpr-max',
    final_act='elu-0.5',
    hidden_act='tanh',
    layers=[100],
    adapt='adagrad',
    n_epochs=epochs_content,
    batch_size=32,
    dropout_p_embed=0,
    dropout_p_hidden=0.0,
    learning_rate=0.01,
    momentum=0.3,
    n_sample=2048,
    sample_alpha=0,
    bpreg=1,
    constrained_embedding=False,
    preset_embeddings=preset_content_embeddings,
    embedding_transform_matrix=True,
)
gru2 = GRU4Rec(**params)
gru2.fit(data_gru_train)

eval_log = []
score_log = []
for u, i_s in data_test.groupby('user').item.apply(list).reset_index().values:
    r199 = np.random.choice(items, (len(i_s) - 1, 199))
    pred_items = np.hstack([np.array(i_s[1:]).reshape(-1, 1), r199])
    ranks = []
    for i1, rs in zip(i_s, pred_items):
        preds1 = gru1.predict_next_batch([u], [i1], rs, batch=1).values
        preds2 = gru2.predict_next_batch([u], [i1], rs, batch=1).values
        score_log.append((u, i1, rs[0], preds1, preds2))
        for a in np.linspace(0, 1, 11):
            preds = a * preds1 + (1 - a) * preds2
            rank = np.sum(preds[0] > preds[1:]) + np.sum(preds[0] == preds[1:]) / 2
            eval_log.append((u, i1, rs[0], rank, a))
res = pd.DataFrame.from_records(eval_log, columns=['user', 'item1', 'item2', 'rank', 'combi'])

jaccard_scores = np.vstack([i[3].squeeze() for i in score_log])
content_scores = np.vstack([i[4].squeeze() for i in score_log])
np.save("logs/jaccard_scores", jaccard_scores)
np.save("logs/content_scores", content_scores)
pd.DataFrame.from_records([i[:3] for i in score_log], columns=["user", "item1", "item2"]).to_csv('logs/combined_order_filtered.csv')
