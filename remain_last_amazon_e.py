from basek.preprocessors.remain_last_amazon import read_reviews, records_writer
from multiprocessing import Process
import argparse
import os

parser = argparse.ArgumentParser(description='parser')


args, _ = parser.parse_known_args()

if __name__ == '__main__':

    dirpath = '/data/project/dataset/Electronics'

    for drop_dups in (False,):
        for pp in (30, 50, None):
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
                            'meta_file': 'meta_Electronics.json',
                            'review_file': 'reviews_Electronics.json',
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
