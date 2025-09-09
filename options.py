import argparse


def set_opts():
    parser = argparse.ArgumentParser()
    # trainning settings
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batchsize of training, (default:64)")
    parser.add_argument('--epochs', type=int, default=2,
                        help="Training epohcs,  (default:60)")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Initialized learning rate, (default: 2e-4)")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="Decaying rate for the learning rate, (default: 0.5)")
    parser.add_argument('-p', '--print_freq', type=int, default=100,
                        help="Print frequence (default: 100)")
    parser.add_argument('-s', '--save_model_freq', type=int, default=10,
                        help="Save moel frequence (default: 1)")

    # dataset settings
    parser.add_argument('--Train_dir', default='./data/data_50-air', type=str, metavar='PATH',
                        help="Path to save the aircraft train dataset")
    parser.add_argument('--Test_dir', default='./data/data_55-air', type=str,
                        metavar='PATH', help="Path to save the aircraft test dataset")

    # model and log saving
    parser.add_argument('--log_dir', default='./log', type=str, metavar='PATH',
                        help="Path to save the log file, (default: ./log)")
    parser.add_argument('--model_dir', default='./model', type=str, metavar='PATH',
                        help="Path to save the model file, (default: ./model)")
    parser.add_argument('--num_workers', default=0, type=int,
                        help="Number of workers to load data, (default: 8)")

    # SNR
    parser.add_argument('--SNR_min', type=float, default=-5, help="SNR_min")
    parser.add_argument('--SNR_max', type=float, default=5, help="SNR_max")
    parser.add_argument('--test_SNR', type=float, default=0, help="test_SNR")
    args = parser.parse_args()

    return args
