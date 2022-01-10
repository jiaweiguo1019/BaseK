import os

import pandas as pd
import networkx as nx

from basek.utils.imports import numpy as np


def process_raw_dataset(dataset, raw_dataset_df, savepath, drop_dups=False, k_core=None):
    processed_raw_dataset_path = os.path.join(savepath, 'processed_raw_dataset.pkl')

    if os.path.exists(processed_raw_dataset_path):
        processed_raw_dataset_df = pd.read_pickle(processed_raw_dataset_path)
        user_count = processed_raw_dataset_df.groupby('user')['user'].count().to_dict()
        item_count = processed_raw_dataset_df.groupby('item')['item'].count().to_dict()
        cate_count = processed_raw_dataset_df.groupby('cate')['cate'].count().to_dict()
        behavior_count = processed_raw_dataset_df.groupby('behavior')['behavior'].count().to_dict()
        print('process_raw_dataset finished!')
        return processed_raw_dataset_df, user_count, item_count, cate_count, behavior_count

    processed_raw_dataset_df = raw_dataset_df
    if drop_dups is True:
        processed_raw_dataset_df.drop_duplicates(['user', 'item'], inplace=True)

    if k_core is not None:
        if not isinstance(k_core, int) or k_core <= 1:
            raise ValueError(f'k_core should be an integer greater than 1, got {k_core}')
        processed_raw_dataset_df = reduce_to_k_core(dataset, processed_raw_dataset_df, savepath, k_core)

    processed_raw_dataset_df.sort_values('timestamp', inplace=True)
    processed_raw_dataset_df.reset_index(drop=True, inplace=True)
    user_count = processed_raw_dataset_df.groupby('user')['user'].count().to_dict()
    item_count = processed_raw_dataset_df.groupby('item')['item'].count().to_dict()
    cate_count = processed_raw_dataset_df.groupby('cate')['cate'].count().to_dict()
    behavior_count = processed_raw_dataset_df.groupby('behavior')['behavior'].count().to_dict()

    print('process_raw_dataset finished!')
    return processed_raw_dataset_df, user_count, item_count, cate_count, behavior_count


def reduce_to_k_core(dataset, dataset_df, savepath, k_core):

    reduced_dataset_path = os.path.join(savepath, f'reduced_k_core_{k_core}_dataset.pkl')
    if os.path.exists(reduced_dataset_path):
        reduced_dataset_df = pd.read_pickle(reduced_dataset_path)
        print(f'reduce_to_{k_core}_core finished!')
        return reduced_dataset_df

    if dataset in ('amazon_books','amazon_electronics', 'yelp'):
        sep = '$^$'
        user_list = dataset_df['user'].tolist()
        user_list = list(map(lambda x: 'user' + sep + x, user_list))
        item_list = dataset_df['item'].tolist()
        item_list = list(map(lambda x: 'item' + sep + x, item_list))
        edges = list(zip(user_list, item_list))

        g = nx.Graph()
        g.add_edges_from(edges)
        user_and_item = list(nx.k_core(g, k_core).edges)
        if not user_and_item:
            raise ValueError(f'k_core numbser: {k_core} is too much, no more interactons left')
        reduced_user_list, reduced_item_list = list(zip(*user_and_item))
        reduced_user_list = list(map(lambda x: x.split(sep)[-1], reduced_user_list))
        reduced_item_list = list(map(lambda x: x.split(sep)[-1], reduced_item_list))
        reduced_user_set = set(reduced_user_list)
        reduced_item_set = set(reduced_item_list)

        reduced_dataset_df = dataset_df[
            dataset_df['user'].isin(reduced_user_set) & dataset_df['item'].isin(reduced_item_set)
        ]
    elif dataset in ('kwai', 'movielens', 'taobao', 'kwai'):
        user_list = dataset_df['user'].tolist()
        item_list = dataset_df['item'].tolist()
        user_offset = np.max(user_list) + 1
        item_offset = np.max(item_list) + 1
        edges = list(zip(np.array(user_list) - user_offset, np.array(item_list) + item_offset))

        g = nx.Graph()
        g.add_edges_from(edges)
        user_and_item = list(nx.k_core(g, k_core).nodes)
        if not user_and_item:
            raise ValueError(f'k_core numbser: {k_core} is too much, no more interactons left')

        reduced_user_list = np.array(list(filter(lambda x: x < 0, user_and_item))) + user_offset
        reduced_item_list = np.array(list(filter(lambda x: x > 0, user_and_item))) - item_offset
        reduced_dataset_df = dataset_df[
            dataset_df['user'].isin(set(reduced_user_list)) & dataset_df['item'].isin(set(reduced_item_list))
        ]

    reduced_dataset_df = reduced_dataset_df.copy()
    reduced_dataset_df.to_pickle(reduced_dataset_path)
    print(f'reduce_to_{k_core}_core finished!')
    return reduced_dataset_df


def build_entity_id_map(
    entity_count, entity_prefix='default',
    id_prefix='default', savepath='./', id_ordered_by_count=True
):
    if id_ordered_by_count:
        entity_count = dict(sorted(entity_count.items(), key=lambda x: x[-1], reverse=True))
        entity_to_id_path = os.path.join(savepath, f'id_ordered_by_count-{entity_prefix}_to_{id_prefix}.pkl')
        id_to_entity_path = os.path.join(savepath, f'id_ordered_by_count-{id_prefix}_to_{entity_prefix}.pkl')
    else:
        entity_count = dict(entity_count.items())
        entity_to_id_path = os.path.join(savepath, f'no_id_ordered_by_count-{entity_prefix}_to_{id_prefix}.pkl')
        id_to_entity_path = os.path.join(savepath, f'no_id_ordered_by_count-{id_prefix}_to_{entity_prefix}.pkl')

    default_count = entity_count.get('default', 0)
    if 'default' in entity_count:
        del entity_count['default']
    entity_to_id = {'null': 0, 'default': 1}
    id_to_entity = {0: 'null', 1: 'default'}
    for idx, entity in enumerate(entity_count):
        entity_to_id[entity] = idx + 2
        id_to_entity[idx + 2] = entity
    entity_count['default'] = default_count
    entity_to_id = dict(sorted(entity_to_id.items(), key=lambda x: x[1]))
    id_to_entity = dict(sorted(id_to_entity.items(), key=lambda x: x[0]))

    entity_to_id = pd.Series(entity_to_id)
    id_to_entity = pd.Series(id_to_entity)
    entity_to_id.to_pickle(entity_to_id_path)
    id_to_entity.to_pickle(id_to_entity_path)

    return entity_to_id, id_to_entity


def build_ordinal_entity_id_map(
    dataset, entity_count, entity_prefix='default',
    id_prefix='default', savepath='./'
):
    entity_to_id_path = os.path.join(savepath, f'{entity_prefix}_to_{id_prefix}.pkl')
    id_to_entity_path = os.path.join(savepath, f'{id_prefix}_to_{entity_prefix}.pkl')

    if dataset == 'taobao':
        entities = ['pv', 'fav', 'cart', 'buy']
    else:
        entities = sorted(list(entity_count.keys()))

    entity_to_id = {'null': 0, 'default': 1}
    id_to_entity = {0: 'null', 1: 'default'}
    for idx, entity in enumerate(entities):
        entity_to_id[entity] = idx + 2
        id_to_entity[idx + 2] = entity

    entity_to_id = pd.Series(entity_to_id)
    id_to_entity = pd.Series(id_to_entity)
    entity_to_id.to_pickle(entity_to_id_path)
    id_to_entity.to_pickle(id_to_entity_path)

    return entity_to_id, id_to_entity
