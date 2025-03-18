import os 
import sys
# sys.path.append('geometric-algebra-transformer')

from models.layers.new_ga_refinement import New_ga_refinement


import argparse
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
    parser.add_argument('--checkpoint', help='checkpoint file', type=str)
    parser.add_argument('--checkpoint_ga', help='checkpoint file of refinement', type=str)
    parser.add_argument('--num-epochs', help='number of epochs', type=int)
    parser.add_argument('--my_epoch_count', help='number of epochs from which resume', type=int)
    parser.add_argument('--my_step_count', help='number of step_count from which resume', type=int)
    parser.add_argument('--version', help='version of task (timestamp by default)', type=str)
    parser.add_argument('--name', required=True, type=str) 
    parser.add_argument('--from_checkpoint', type=bool) 
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if hasattr(args, "from_checkpoint") and args.from_checkpoint:
        from_checkpoint = True
    else:
        from_checkpoint = False


    logger, writer = reset_options(options, args)
    prova = New_ga_refinement(options)



if __name__ == "__main__":
    main()