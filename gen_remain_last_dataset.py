from basek.preprocessors import read_reviews, records_writer
from multiprocessing import Process
import argparse
import os

ABS_PATH = os.path.abspath('.')
parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--dataset', type=str, default='movielens-1m')
parser.add_argument('--dirpath', type=str, default=os.path.join(ABS_PATH, 'datasets', 'movielens-1m'))

# parser.add_argument('--dataset', type=str, default='movielens')
# parser.add_argument('--review_path', type=str, default='./datasets/Taobao/UserBehavior.csv')
# parser.add_argument('--seq_len', type=int, default=100)
# parser.add_argument('--from_raw', type=bool, default=True)
# parser.add_argument('--drop_dups', type=bool, default=False)
# parser.add_argument('--only_click', type=bool, default=False)
# parser.add_argument('--k_core', type=int, default=5)
# parser.add_argument('--id_ordered_by_count', type=bool, default=True)

args, _ = parser.parse_known_args()


all_pp = (None,)
all_k_core = (10,)
all_max_seq_len = (50, 100)
all_neg_samples = (10,)


if __name__ == '__main__':
    dataset = args.dataset
    dirpath = args.dirpath

    for drop_dups in (False,):
        for pp in all_pp:
            write_ps = []
            for id_ordered_by_count in (True,):
                read_ps = []
                for k_core in all_k_core:
                    subdir = ''
                    if pp:
                        subdir = f'{subdir}-pp_{pp}'
                    if drop_dups:
                        subdir = f'{subdir}-drop_dups'
                    if k_core:
                        subdir = f'{subdir}-k_core_{k_core}'
                    if not subdir:
                        subdir = 'raw'
                    else:
                        subdir = subdir[1:]
                    savepath = os.path.join(dirpath, subdir)

                    read_p = Process(
                        target=read_reviews,
                        kwargs={
                            'dataset': dataset,
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

                for k_core in all_k_core:
                    for max_seq_len in all_max_seq_len:
                        for neg_samples in all_neg_samples:
                            subdir = ''
                            if pp:
                                subdir = f'{subdir}-pp_{pp}'
                            if drop_dups:
                                subdir = f'{subdir}-drop_dups'
                            if k_core:
                                subdir = f'{subdir}-k_core_{k_core}'
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
