import argparse

def parse_opt():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--in_channels',
        default=1,
        type=int,
        help="Input channels of the net"
    )
    parse.add_argument(
        '--out_channels',
        default=4,
        type=int,
        help="Output channels of the net"
    )
    parse.add_argument(
        '--final_sigmoid',
        default=False,
        type=bool,
        help="Select multi-classification or binary-classification"
    )
    parse.add_argument(
        '--learning_rate',
        default=0.0002,
        type=float,
        help="Initial learning rate"
    )
    parse.add_argument(
        '--epoch',
        default=1,
        type=int,
        help="Number of total epochs to run"
    )
    parse.add_argument(
        '--batch_size',
        default=5,
        type=int,
        help="Batch size"
    )
    parse.add_argument(
        '--train_size',
        default=1703,
        type=int,
        help="The number of trainset"
    )
    parse.add_argument(
        '--valid_size',
        default=169,
        type=int,
        help="The number of validset"
    )
    parse.add_argument(
        '--test_size',
        default=281,
        type=int,
        help="The number of testset"
    )
    parse.add_argument(
        '--trianing',
        default=False,
        type=bool,
        help="Select training or testing"
    )
    parse.add_argument(
        '--num_agents',
        default=10,
        type=int,
        help="The number of agents"
    )
    parse.add_argument(
        '--rounds',
        default=200,
        type=int,
        help="The rounds of the federated learning"
    )
    parse.add_argument(
        '--aggr',
        default='avg',
        type=str,
        help="Aggregation mode"
    )
    args = parse.parse_args()
    return args