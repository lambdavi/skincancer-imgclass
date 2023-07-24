import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['resnet', 'xception', 'cnn'], default='cnn', help='model name')
    parser.add_argument('--save', action='store_true', default=False, help='Model saved at the end (training performed)')
    parser.add_argument('--load', action='store_true', default=False, help='Load saved model')
    parser.add_argument('--pred_path', type=str, default = None, help='Path of image to predict')

    return parser
