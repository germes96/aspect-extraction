import argparse
from model import GRAM_BIGRU_CRF
from test import Test
from pretrained import Parser

import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

print(tf.contrib.util.constant_value(tf.ones([1])))


def run(args):
    if args.train == 1:
        model = GRAM_BIGRU_CRF(params=args)
        model.run(336)
    else:
        Test(args).run()
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aspect Term Extraction With GRAM-BIGRU-CRF")
    parser.add_argument("-ds_name", type=str, default="Laptop", help="dataset name")
    parser.add_argument("-limit_len", type=int, default=30, help="nombre de mot max par phrase")
    parser.add_argument("-n_epoch", type=int, default=10, help="number of training epoch")
    parser.add_argument("-model_name", type=str, default="biLSM", help="model name: BiLSMT or other")
    parser.add_argument("-train", type=int, default=1, help="signaler si c'est l'entrainement 0 ou 1")
    args = parser.parse_args()
    run(args)
