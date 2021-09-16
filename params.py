import argparse


parser = argparse.ArgumentParser(description='SpeedScheduler')


# -- Basic --
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')


# -- Training -- #
# parser.add_argument('-f', '--config_file', required=True, help='config file')
parser.add_argument('-t', '--task', type=str, default='train', help='train, test or eval')
parser.add_argument('-e', '--epochs', type=int, default=10, help='num of epochs')


args, _ = parser.parse_known_args()

print(args)
