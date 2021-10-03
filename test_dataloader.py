import pickle as pkl
from basek.preprocessors.split_time_amazon import read_reviews, DataLoader


def main():
    sparse_features = ['uid', 'iid', 'cid']
    meta_path = './datasets/Amazon/meta_Books.json'
    review_path = './datasets/Amazon/reviews_Books_5.json'
    data_files, sparse_features_max_idx_path, all_indices_path = read_reviews(meta_path, review_path)

    data_files = './datasets/Amazon/train.pkl', './datasets/Amazon/test.pkl'
    sparse_features_max_idx_path = './datasets/Amazon/sparse_features_max_idx.pkl'
    all_indices_path = './datasets/Amazon/all_indices.pkl'

    train_file, test_file = data_files
    test_input = DataLoader(test_file, 5, 3, kind='test')


if __name__ == '__main__':
    main()
