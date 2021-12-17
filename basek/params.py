import argparse


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser('Parser for BaseK')

# -- Basic --
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
parser.add_argument('--from_raw', type=bool, default=False, help='from raw data')
parser.add_argument('--k_core', type=int, default=None, help='reduce to k_core')
parser.add_argument('--only_click', type=bool, default=True, help='only_click')
parser.add_argument('--first_lines', type=int, default=None, help='first_lines')

# -- Training -- #
# parser.add_argument('-f', '--config_file', required=True, help='config file')
parser.add_argument('-t', '--task', type=str, default='train', help='train, test or eval')
parser.add_argument('-e', '--epochs', type=int, default=8, help='num of epochs')
parser.add_argument('--gpus', type=str, default='0', help='gups')
parser.add_argument('--filter', type=int, default=5, help='filter')
parser.add_argument('--dirpath', type=str, default=None, help='dirpath')
parser.add_argument('--lr', type=float, default=1e-3, help='lr')
parser.add_argument('--seq_len', type=int, default=50, help='seq_len')
parser.add_argument('--short_seq_len', type=int, default=10, help='short_seq_len')
parser.add_argument('--emb_dropout', type=boolean_string, default=False, help='emb_dropout')
parser.add_argument('--id_ordered_by_count', type=boolean_string, default=True, help='id_ordered_by_count')
parser.add_argument('--shuffle', type=boolean_string, default=True, help='shuffle')
parser.add_argument('--ffn_hidden_unit', type=int, default=128, help='shuffle')
parser.add_argument('--emb_dim', type=int, default=64, help='shuffle')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')


args, _ = parser.parse_known_args()

print('#' * 132)
print(args)
print('#' * 132)
