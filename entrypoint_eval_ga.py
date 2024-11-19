import argparse
import sys
import os 
# from functions.trainer import Trainer
from functions.evaluateMod import EvaluateGA
from options import update_options, options, reset_options


def parse_args():
    parser = argparse.ArgumentParser(description='Pixel2Mesh Training Entrypoint')
    parser.add_argument('--options', help='experiment options file name', required=False, type=str)

    

    args, rest = parser.parse_known_args()
    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        # load the configuration from the file 'options' inserted
        update_options(args.options)

    # training
    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--shuffle', help='shuffle samples', default=False, action='store_true')
    parser.add_argument('--checkpoint_ga', help='checkpoint ga refinement file', type=str)
    parser.add_argument('--checkpoint', help='checkpoint file', type=str)
    parser.add_argument('--version', help='version of task (timestamp by default)', type=str)
    parser.add_argument('--name', required=True, type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger, writer = reset_options(options, args,phase='eval')
    #logger -> used to see the messages
    #writer -> used to save data of training
    evaluate = EvaluateGA(options, logger, writer)
    evaluate.evaluate()


if __name__ == "__main__":
    main()