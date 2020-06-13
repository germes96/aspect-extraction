import argparse
from pretrained import Parser
# import tensorflow as tf
# tf.get_logger().setLevel('INFO')
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def run(args):
    size = Parser(args).run(args.limit_len)
    print(size)
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aspect Term Extraction With GRAM-BIGRU-CRF")
    parser.add_argument("-ds_name", type=str, default="Laptop", help="dataset name")
    parser.add_argument("-limit_len", type=int, default=30, help="nombre de mot max par phrase")
    parser.add_argument("-train", type=int, default=1, help="signaler si c'est l'entrainement 0 ou 1")
    args = parser.parse_args()
    run(args)
