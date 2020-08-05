import argparse


def arg_parser(argv=None):
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # common args
    parser.add_argument('--seed', type=int, default=543, help='random seed (default: 543)')
    parser.add_argument(
        '--log_interval', type=int, default=10, required=False,
        help='the interval for log average episode reward',
    )

    parser.add_argument('--gamma', type=float, default=0.99, required=False, help='Discount factor for rewards')
    parser.add_argument('--lr', type=float, default=0.01, required=False, help='Learning rate for optimizer')

    args = parser.parse_args()
    return args
