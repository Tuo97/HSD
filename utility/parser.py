import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Running Model.")
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--verbose', type=int, default=1, help='Interval of evaluation.')
    parser.add_argument('--save_model', type=bool, default=False, help='Whether to save')

    parser.add_argument('--epoch', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')

    parser.add_argument('--embed_size', type=int, default=64, help='Embedding size.')
    parser.add_argument('--n_layers', type=int, default=3, help='Layer numbers.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Reg weight for ssl loss')
    parser.add_argument('--curvature', type=float, default=1.0, help='Reg weight for ssl loss')

    parser.add_argument('--channel_weight', type=float, default=0.6, help='Weight for modal-wise bpr')
    parser.add_argument('--kl_weight', type=float, default=0.6, help='Weight for self-distillation ')
    parser.add_argument('--tau', type=float, default=1.5, help='Temperature for self-distillation')
    parser.add_argument('--l2_weight', type=float, default=0.0, help='Reg weight for l2 regulation')

    parser.add_argument('--show_step', type=int, default=3, help='Test every show_step epochs.')
    parser.add_argument('--Ks', nargs='?', default='[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]', help='Metrics scale')

    # Useless
    parser.add_argument('--data_path', nargs='?', default='data/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='gowalla', help='Choose a dataset from {gowalla, amazon, tmall}')
    parser.add_argument('--n_batch', type=int, default=40, help='Number of mini-batches')
    parser.add_argument('--train_num', type=int, default=10000, help='Number of training instances per epoch')
    parser.add_argument('--sample_num', type=int, default=40, help='Number of pos/neg samples for each instance')

    return parser.parse_args()


