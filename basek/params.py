import argparse


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
parser.add_argument('-e', '--epochs', type=int, default=50, help='num of epochs')


args, _ = parser.parse_known_args()

print('#' * 132)
print(args)
print('#' * 132)
