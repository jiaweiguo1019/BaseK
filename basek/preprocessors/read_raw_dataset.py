import json
import os
import pandas as pd
import time

from tqdm import tqdm
from basek.utils.imports import numpy as np


def read_raw_dataset(dataset, dirpath, savepath, pp=None):
    item_to_cate_path = os.path.join(savepath, 'item_to_cate.pkl')
    raw_dataset_path = os.path.join(savepath, 'raw_dataset.pkl')
    if os.path.exists(item_to_cate_path) and os.path.exists(raw_dataset_path):
        item_to_cate = pd.read_pickle(item_to_cate_path)
        raw_dataset_df = pd.read_pickle(raw_dataset_path)
    else:
        if dataset in ('movielens-1m', 'movielens-25m'):
            item_to_cate, raw_dataset_df = read_raw_dataset_movielens(dirpath, savepath, dataset, pp)
        elif dataset in ('amazon_electronics', 'amazon_books'):
            item_to_cate, raw_dataset_df =  read_raw_dataset_amazon(dataset, dirpath, savepath, pp)
        elif dataset == 'taobao':
            item_to_cate, raw_dataset_df = read_raw_dataset_taobao(dirpath, savepath, pp)
        elif dataset == 'yelp':
            item_to_cate, raw_dataset_df = read_raw_dataset_yelp(dirpath, savepath, pp)
        elif dataset == 'kwai':
            item_to_cate, raw_dataset_df = read_raw_dataset_kwai(dirpath, savepath, pp)
        else:
            raise ValueError(f'dataset: {dataset} not supported')
    print('read_review finished!')
    return item_to_cate, raw_dataset_df


def read_raw_dataset_movielens(dirpath, savepath, dataset, pp):
    item_to_cate_path = os.path.join(savepath, 'item_to_cate.pkl')
    raw_dataset_path = os.path.join(savepath, 'raw_dataset.pkl')
    if dataset == 'movielens-1m':
        review_path = os.path.join(dirpath, 'ratings.dat')
        meta_path = os.path.join(dirpath, 'movies.dat')
        
        review_dataset_df = pd.read_csv(review_path, sep='::', names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')
        movies_dataset_df = pd.read_csv(meta_path, sep='::', names=['movieId','title','genres'], encoding='latin-1', engine='python')
    elif dataset == 'movielens-25m':
        review_path = os.path.join(dirpath, 'ratings.csv')
        meta_path = os.path.join(dirpath, 'movies.csv')

        review_dataset_df = pd.read_csv(review_path)
        movies_dataset_df = pd.read_csv(meta_path)
    else:
        raise ValueError(f'dataset: {dataset} not supported')

    movies_dataset_df['genres'] = movies_dataset_df['genres'].map(
        lambda x: 'default' if x == '(no genres listed)' else x.split('|')[-1]
    )
    merged_dataset_df = pd.merge(review_dataset_df, movies_dataset_df, on='movieId')
    raw_dataset_df = pd.DataFrame()
    raw_dataset_df['user'] = merged_dataset_df['userId']
    raw_dataset_df['item'] = merged_dataset_df['movieId']
    raw_dataset_df['cate'] = merged_dataset_df['genres']
    raw_dataset_df['behavior'] = merged_dataset_df['rating']
    raw_dataset_df['timestamp'] = merged_dataset_df['timestamp']
    raw_dataset_df.dropna(inplace=True)
    if pp:
        user_list = raw_dataset_df['user'].unique()
        sampled_user_list = []
        for user in user_list:
            if np.random.random() < pp / 100:
                sampled_user_list.append(user)
        sampled_user_set = set(sampled_user_list)
        raw_dataset_df = raw_dataset_df[raw_dataset_df['user'].isin(sampled_user_set)]
    raw_dataset_df.drop_duplicates(['user', 'item', 'behavior', 'timestamp'], inplace=True)
    raw_dataset_df.sort_values('timestamp', inplace=True)
    raw_dataset_df.reset_index(drop=True, inplace=True)
    item_to_cate = dict(zip(raw_dataset_df['item'], raw_dataset_df['cate']))

    item_to_cate = pd.Series(item_to_cate)
    item_to_cate['null'] = 'null'
    item_to_cate['default'] = 'default'
    raw_dataset_df['cate'] = raw_dataset_df['item'].map(item_to_cate)
    item_to_cate.to_pickle(item_to_cate_path)
    raw_dataset_df.to_pickle(raw_dataset_path)
    return item_to_cate, raw_dataset_df


def read_raw_dataset_amazon(dataset, dirpath, savepath, pp=None):
    item_to_cate_path = os.path.join(savepath, 'item_to_cate.pkl')
    raw_dataset_path = os.path.join(savepath, 'raw_dataset.pkl')
    _, sub_dataset = dataset.split('_')
    if sub_dataset == 'books':
        review_path = os.path.join(dirpath, 'reviews_Books.json')
        meta_path = os.path.join(dirpath, 'meta_Books.json')
    else:
        review_path = os.path.join(dirpath, 'reviews_Electronics.json')
        meta_path = os.path.join(dirpath, 'meta_Electronics.json')

    item_to_cate = {}
    with open(meta_path, 'r') as f:
        for line in tqdm(f):
            line = eval(line)
            item, cate = line['asin'], line['categories'][-1][-1]
            item_to_cate[item] = cate

    raw_dataset = {}
    with open(review_path, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            line = json.loads(line)
            user, item, behavior, timestamp = (
                line['reviewerID'], line['asin'], line['overall'], line['unixReviewTime']
            )
            raw_dataset[i] = {'user': user, 'item': item, 'behavior': behavior, 'timestamp': timestamp}
    raw_dataset_df = pd.DataFrame.from_dict(raw_dataset, orient='index')
    if pp:
        user_list = raw_dataset_df['user'].unique()
        sampled_user_list = []
        for user in user_list:
            if np.random.random() < pp / 100:
                sampled_user_list.append(user)
        sampled_user_set = set(sampled_user_list)
        raw_dataset_df = raw_dataset_df[raw_dataset_df['user'].isin(sampled_user_set)]
    raw_dataset_df.drop_duplicates(['user', 'item', 'behavior', 'timestamp'], inplace=True)
    raw_dataset_df.sort_values('timestamp', inplace=True)
    raw_dataset_df.reset_index(drop=True, inplace=True)

    item_to_cate = pd.Series(item_to_cate)
    item_to_cate['null'] = 'null'
    item_to_cate['default'] = 'default'
    raw_dataset_df['cate'] = raw_dataset_df['item'].map(item_to_cate)
    item_to_cate.to_pickle(item_to_cate_path)
    raw_dataset_df.to_pickle(raw_dataset_path)
    return item_to_cate, raw_dataset_df


def read_raw_dataset_taobao(dirpath, savepath, pp):
    item_to_cate_path = os.path.join(savepath, 'item_to_cate.pkl')
    raw_dataset_path = os.path.join(savepath, 'raw_dataset.pkl')
    review_path = os.path.join(dirpath, 'UserBehavior.csv')

    raw_dataset_df = pd.read_csv(review_path, names=['user', 'item', 'cate', 'behavior', 'timestamp'])
    start_ts = int(time.mktime(time.strptime('2017-11-25 0:0:0', '%Y-%m-%d %H:%M:%S')))
    end_ts = int(time.mktime(time.strptime('2017-12-4 0:0:0', '%Y-%m-%d %H:%M:%S')))
    raw_dataset_df = raw_dataset_df[
        (raw_dataset_df['timestamp'] >= start_ts) & (raw_dataset_df['timestamp'] < end_ts)
    ]
    raw_dataset_df.dropna(inplace=True)
    if pp:
        user_list = raw_dataset_df['user'].unique()
        sampled_user_list = []
        for user in user_list:
            if np.random.random() < pp / 100:
                sampled_user_list.append(user)
        sampled_user_set = set(sampled_user_list)
        raw_dataset_df = raw_dataset_df[raw_dataset_df['user'].isin(sampled_user_set)]
    raw_dataset_df.drop_duplicates(['user', 'item', 'behavior', 'timestamp'], inplace=True)
    raw_dataset_df.sort_values('timestamp', inplace=True)
    raw_dataset_df.reset_index(drop=True, inplace=True)
    item_to_cate = dict(zip(raw_dataset_df['item'], raw_dataset_df['cate']))

    item_to_cate = pd.Series(item_to_cate)
    item_to_cate['null'] = 'null'
    item_to_cate['default'] = 'default'
    # raw_dataset_df['cate'] = item_to_cate[raw_dataset_df['item']].values
    raw_dataset_df['cate'] = raw_dataset_df['item'].map(item_to_cate)
    item_to_cate.to_pickle(item_to_cate_path)
    raw_dataset_df.to_pickle(raw_dataset_path)
    return item_to_cate, raw_dataset_df


def read_raw_dataset_kwai(dirpath, savepath, pp):
    item_to_cate_path = os.path.join(savepath, 'item_to_cate.pkl')
    raw_dataset_path = os.path.join(savepath, 'raw_dataset.pkl')

    review_path = os.path.join(dirpath, 'all_review.pkl')
    raw_dataset_df = pd.read_pickle(review_path)
    start_ts = int(time.mktime(time.strptime('2021-08-20 0:0:0', '%Y-%m-%d %H:%M:%S')))
    end_ts = int(time.mktime(time.strptime('2021-08-29 0:0:0', '%Y-%m-%d %H:%M:%S')))
    raw_dataset_df = raw_dataset_df[
        (raw_dataset_df['timestamp'] >= start_ts) & (raw_dataset_df['timestamp'] < end_ts)
    ]
    raw_dataset_df.dropna(inplace=True)
    if pp:
        user_list = raw_dataset_df['user'].unique()
        sampled_user_list = []
        for user in user_list:
            if np.random.random() < pp / 100:
                sampled_user_list.append(user)
        sampled_user_set = set(sampled_user_list)
        raw_dataset_df = raw_dataset_df[raw_dataset_df['user'].isin(sampled_user_set)]
    raw_dataset_df.drop_duplicates(['user', 'item', 'behavior', 'timestamp'], inplace=True)
    raw_dataset_df.sort_values('timestamp', inplace=True)
    raw_dataset_df.reset_index(drop=True, inplace=True)
    item_to_cate = dict(zip(raw_dataset_df['item'], raw_dataset_df['cate']))

    item_to_cate = pd.Series(item_to_cate)
    item_to_cate['null'] = 'null'
    item_to_cate['default'] = 'default'
    raw_dataset_df['cate'] = raw_dataset_df['item'].map(item_to_cate)
    item_to_cate.to_pickle(item_to_cate_path)
    raw_dataset_df.to_pickle(raw_dataset_path)
    return item_to_cate, raw_dataset_df


def read_raw_dataset_yelp(dirpath, savepath, pp=None):
    item_to_cate_path = os.path.join(savepath, 'item_to_cate.pkl')
    raw_dataset_path = os.path.join(savepath, 'raw_dataset.pkl')
    review_path = os.path.join(dirpath, 'yelp_academic_dataset_review.json')
    meta_path = os.path.join(dirpath, 'yelp_academic_dataset_business.json')

    item_to_cate = {}
    with open(meta_path, 'r') as f:
        for line in tqdm(f):
            line = json.loads(line)
            item, cate = line['business_id'], line['categories']
            cate = 'default' if cate is None else cate.split(',')[-1].strip()
            item_to_cate[item] = cate

    raw_dataset = {}
    with open(review_path, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            line = json.loads(line)
            user, item, behavior, timestamp = (
                line['user_id'], line['business_id'], line['stars'],
                int(time.mktime(time.strptime(line['date'], '%Y-%m-%d %H:%M:%S')))
            )
            raw_dataset[i] = {'user': user, 'item': item, 'behavior': behavior, 'timestamp': timestamp}
    raw_dataset_df = pd.DataFrame.from_dict(raw_dataset, orient='index')
    if pp:
        user_list = raw_dataset_df['user'].unique()
        sampled_user_list = []
        for user in user_list:
            if np.random.random() < pp / 100:
                sampled_user_list.append(user)
        sampled_user_set = set(sampled_user_list)
        raw_dataset_df = raw_dataset_df[raw_dataset_df['user'].isin(sampled_user_set)]
    raw_dataset_df.drop_duplicates(['user', 'item', 'behavior', 'timestamp'], inplace=True)
    raw_dataset_df.sort_values('timestamp', inplace=True)
    raw_dataset_df.reset_index(drop=True, inplace=True)

    item_to_cate = pd.Series(item_to_cate)
    item_to_cate['null'] = 'null'
    item_to_cate['default'] = 'default'
    raw_dataset_df['cate'] = raw_dataset_df['item'].map(item_to_cate)
    item_to_cate.to_pickle(item_to_cate_path)
    raw_dataset_df.to_pickle(raw_dataset_path)
    return item_to_cate, raw_dataset_df
