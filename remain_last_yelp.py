from basek.preprocessors.remain_last_yelp import read_reviews, records_writer
from multiprocessing import Process
import argparse
import os

parser = argparse.ArgumentParser(description='parser')

parser.add_argument('--review_path', type=str, default='./datasets/Taobao/UserBehavior.csv')
parser.add_argument('--seq_len', type=int, default=100)
parser.add_argument('--from_raw', type=bool, default=True)
parser.add_argument('--drop_dups', type=bool, default=False)
parser.add_argument('--only_click', type=bool, default=False)
parser.add_argument('--k_core', type=int, default=5)
parser.add_argument('--id_ordered_by_count', type=bool, default=True)


args, _ = parser.parse_known_args()

if __name__ == '__main__':

    dirpath = '/data/project/datasets/yelp2018'

    for drop_dups in (False, True):
        for pp in (50, None):
            write_ps = []
            for id_ordered_by_count in (True, False):
                read_ps = []
                for k_core in (20, 10, 5):
                    subdir = ''
                    if pp:
                        subdir = f'{subdir}-pp_{pp}'
                    if drop_dups:
                        subdir = f'{subdir}-drop_dups'
                    if k_core:
                        subdir = f'{subdir}-k_core_{k_core}'
                    else:
                        subdir = f'{subdir}-k_core_1'
                    if not subdir:
                        subdir = 'raw'
                    else:
                        subdir = subdir[1:]
                    savepath = os.path.join(dirpath, subdir)

                    read_p = Process(
                        target=read_reviews,
                        kwargs={
                            'business_file': 'yelp_academic_dataset_business.json',
                            'review_file': 'yelp_academic_dataset_review.json',
                            'dirpath': dirpath,
                            'savepath': savepath,
                            'pp': pp,
                            'drop_dups': drop_dups,
                            'k_core': k_core,
                            'id_ordered_by_count': id_ordered_by_count
                        }
                    )
                    read_p.daemon = True
                    read_ps.append(read_p)
                    read_p.start()
                for read_p in read_ps:
                    read_p.join()

                for k_core in (20, 10, 5):
                    for max_seq_len in (50, 100):
                        for neg_samples in (10,):
                            subdir = ''
                            if pp:
                                subdir = f'{subdir}-pp_{pp}'
                            if drop_dups:
                                subdir = f'{subdir}-drop_dups'
                            if k_core:
                                subdir = f'{subdir}-k_core_{k_core}'
                            else:
                                subdir = f'{subdir}-k_core_1'
                            if not subdir:
                                subdir = 'raw'
                            else:
                                subdir = subdir[1:]
                            savepath = os.path.join(dirpath, subdir)
                            write_p = Process(
                                target=records_writer,
                                kwargs={
                                    'savepath': savepath,
                                    'max_seq_len': max_seq_len,
                                    'neg_samples': neg_samples,
                                    'id_ordered_by_count': id_ordered_by_count,
                                    'write_train': True,
                                    'write_test': True
                                }
                            )
                            write_p.daemon = True
                            write_ps.append(write_p)
                            write_p.start()
            for write_p in write_ps:
                write_p.join()
