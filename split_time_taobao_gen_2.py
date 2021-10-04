from basek.preprocessors.split_time_taobao_gen import read_reviews
from multiprocessing import Process
import argparse

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
    p_s = []
    for drop_dups in (True,):
        for only_click in (True,):
            for id_ordered_by_count in (True,):
                for k_core in (5, 10, 15, 20):
                    p = Process(
                        target=read_reviews,
                        kwargs={
                            'review_path': './datasets/Taobao/UserBehavior.csv',
                            'seq_len': 100,
                            'from_raw': True,
                            'drop_dups': drop_dups,
                            'only_click': only_click,
                            'k_core': k_core,
                            'id_ordered_by_count': id_ordered_by_count
                        }
                    )
                    p.daemon = True
                    p_s.append(p)
                    p.start()
    for p in p_s:
        p.join()

    p_s = []
    for drop_dups in (True,):
        for only_click in (True,):
            for id_ordered_by_count in (False,):
                for k_core in (5, 10, 15, 20):
                    p = Process(
                        target=read_reviews,
                        kwargs={
                            'review_path': './datasets/Taobao/UserBehavior.csv',
                            'seq_len': 100,
                            'from_raw': True,
                            'drop_dups': drop_dups,
                            'only_click': only_click,
                            'k_core': k_core,
                            'id_ordered_by_count': id_ordered_by_count
                        }
                    )
                    p.daemon = True
                    p_s.append(p)
                    p.start()
    for p in p_s:
        p.join()

    p_s = []
    for drop_dups in (True,):
        for only_click in (False,):
            for id_ordered_by_count in (True,):
                for k_core in (5, 10, 15, 20):
                    p = Process(
                        target=read_reviews,
                        kwargs={
                            'review_path': './datasets/Taobao/UserBehavior.csv',
                            'seq_len': 100,
                            'from_raw': True,
                            'drop_dups': drop_dups,
                            'only_click': only_click,
                            'k_core': k_core,
                            'id_ordered_by_count': id_ordered_by_count
                        }
                    )
                    p.daemon = True
                    p_s.append(p)
                    p.start()
    for p in p_s:
        p.join()

    p_s = []
    for drop_dups in (True,):
        for only_click in (False,):
            for id_ordered_by_count in (False,):
                for k_core in (5, 10, 15, 20):
                    p = Process(
                        target=read_reviews,
                        kwargs={
                            'review_path': './datasets/Taobao/UserBehavior.csv',
                            'seq_len': 100,
                            'from_raw': True,
                            'drop_dups': drop_dups,
                            'only_click': only_click,
                            'k_core': k_core,
                            'id_ordered_by_count': id_ordered_by_count
                        }
                    )
                    p.daemon = True
                    p_s.append(p)
                    p.start()
    for p in p_s:
        p.join()

    p_s = []
    for drop_dups in (False,):
        for only_click in (True,):
            for id_ordered_by_count in (True,):
                for k_core in (5, 10, 15, 20):
                    p = Process(
                        target=read_reviews,
                        kwargs={
                            'review_path': './datasets/Taobao/UserBehavior.csv',
                            'seq_len': 100,
                            'from_raw': True,
                            'drop_dups': drop_dups,
                            'only_click': only_click,
                            'k_core': k_core,
                            'id_ordered_by_count': id_ordered_by_count
                        }
                    )
                    p.daemon = True
                    p_s.append(p)
                    p.start()
    for p in p_s:
        p.join()

    p_s = []
    for drop_dups in (False,):
        for only_click in (True,):
            for id_ordered_by_count in (False,):
                for k_core in (5, 10, 15, 20):
                    p = Process(
                        target=read_reviews,
                        kwargs={
                            'review_path': './datasets/Taobao/UserBehavior.csv',
                            'seq_len': 100,
                            'from_raw': True,
                            'drop_dups': drop_dups,
                            'only_click': only_click,
                            'k_core': k_core,
                            'id_ordered_by_count': id_ordered_by_count
                        }
                    )
                    p.daemon = True
                    p_s.append(p)
                    p.start()
    for p in p_s:
        p.join()

    p_s = []
    for drop_dups in (False,):
        for only_click in (False,):
            for id_ordered_by_count in (True,):
                for k_core in (5, 10, 15, 20):
                    p = Process(
                        target=read_reviews,
                        kwargs={
                            'review_path': './datasets/Taobao/UserBehavior.csv',
                            'seq_len': 100,
                            'from_raw': True,
                            'drop_dups': drop_dups,
                            'only_click': only_click,
                            'k_core': k_core,
                            'id_ordered_by_count': id_ordered_by_count
                        }
                    )
                    p.daemon = True
                    p_s.append(p)
                    p.start()
    for p in p_s:
        p.join()

    p_s = []
    for drop_dups in (False,):
        for only_click in (False,):
            for id_ordered_by_count in (False,):
                for k_core in (5, 10, 15, 20):
                    p = Process(
                        target=read_reviews,
                        kwargs={
                            'review_path': './datasets/Taobao/UserBehavior.csv',
                            'seq_len': 100,
                            'from_raw': True,
                            'drop_dups': drop_dups,
                            'only_click': only_click,
                            'k_core': k_core,
                            'id_ordered_by_count': id_ordered_by_count
                        }
                    )
                    p.daemon = True
                    p_s.append(p)
                    p.start()
    for p in p_s:
        p.join()
