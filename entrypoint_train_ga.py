
# import argparse
# import sys

# from functions.saver import CheckpointSaver
# from functions.evaluator import Evaluator
# from options import update_options, options, reset_options


# epoch_count = step_count = 0

# def parse_args():
#     parser = argparse.ArgumentParser(description='Pixel2Mesh Evaluation Entrypoint')
#     parser.add_argument('--options', help='experiment options file name', required=False, type=str)

#     args, rest = parser.parse_known_args()
#     if args.options is None:
#         print("Running without options file...", file=sys.stderr)
#     else:
#         update_options(args.options)

#     parser.add_argument('--batch-size', help='batch size', type=int)
#     parser.add_argument('--shuffle', help='shuffle samples', default=False, action='store_true')
#     parser.add_argument('--checkpoint', help='trained checkpoint file', type=str, required=True)
#     parser.add_argument('--version', help='version of task (timestamp by default)', type=str)
#     parser.add_argument('--name', help='subfolder name of this experiment', required=True, type=str)
#     parser.add_argument('--gpus', help='number of GPUs to use', type=int)

#     args = parser.parse_args()

#     return args


# def main():
#     args = parse_args()
#     logger, writer = reset_options(options, args, phase='eval')


#     saver = CheckpointSaver(logger, checkpoint_dir=str(options.checkpoint_dir),
#                                 checkpoint_file=options.checkpoint)
    


#     def init_with_checkpoint(saver,logger,models_dict):
#         checkpoint = saver.load_checkpoint()
#         if checkpoint is None:
#             logger.info("Checkpoint not loaded")
#             return
#         for model_name, model in models_dict().items():
#             if model_name in checkpoint:
#                 if isinstance(model, torch.nn.DataParallel):
#                     model.module.load_state_dict(checkpoint[model_name], strict=False)
#                 else:
#                     model.load_state_dict(checkpoint[model_name], strict=False)
#         if optimizers_dict() is not None:
#             for optimizer_name, optimizer in optimizers_dict().items():
#                 if optimizer_name in checkpoint:
#                     optimizer.load_state_dict(checkpoint[optimizer_name])
#         else:
#             logger.warning("Optimizers not found in the runner, skipping...")
#         if "epoch" in checkpoint:
#             epoch_count = checkpoint["epoch"]
#         if "total_step_count" in checkpoint:
#             step_count = checkpoint["total_step_count"]


#     def models_dict(self):
#         #TODO MODIFY
#         return {'model': self.model}

#     def optimizers_dict(self):
#         #TODO MODIFY
#         return {'optimizer': self.optimizer,
#                 'lr_scheduler': self.lr_scheduler}

    


# if __name__ == "__main__":
#     main()




'''
prova 2
'''


import argparse
import sys
import time
from datetime import timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.backbones import get_backbone
from functions.evaluator import Evaluator


from functions.evaluator import Evaluator
from options import update_options, options, reset_options
from utils.mesh import Ellipsoid


from models.backbones import get_backbone
from models.layers.gbottleneck import GBottleneck
from models.layers.gconv import GConv
from models.layers.gpooling import GUnpooling
from models.layers.gprojection import GProjection
from models.layers.selfattention_ga import SelfAttentionGA
from algebra.cliffordalgebra import CliffordAlgebra



def parse_args():
    parser = argparse.ArgumentParser(description='Pixel2Mesh Evaluation Entrypoint')
    parser.add_argument('--options', help='experiment options file name', required=False, type=str)

    args, rest = parser.parse_known_args()
    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--shuffle', help='shuffle samples', default=False, action='store_true')
    parser.add_argument('--checkpoint', help='trained checkpoint file', type=str, required=True)
    parser.add_argument('--version', help='version of task (timestamp by default)', type=str)
    parser.add_argument('--name', help='subfolder name of this experiment', required=True, type=str)
    parser.add_argument('--gpus', help='number of GPUs to use', type=int)

    args = parser.parse_args()

    return args


def main():

    epoch_count = step_count = 0
    args = parse_args()
    logger, writer = reset_options(options, args, phase='eval')

    evaluator = Evaluator(options, logger, writer)

    # 1. get the input batch data

    
    # 2. analyze output 
    # 3. apply geomtric algebra on output and train

    train_data_loader = evaluator.train_data_ga()

    print(f"prova -> {options['model']}")
    # print(options)


    if hasattr(options, 'backbone'):
        print(options.backbone)
    else:
        print("Backbone attribute is not set")



    nn_encoder, nn_decoder = get_backbone(options)
    coord_dim = options.coord_dim
    hidden_dim = options.hidden_dim
    last_hidden_dim = options.last_hidden_dim
    features_dim = nn_encoder.features_dim + coord_dim
    ellipsoid = Ellipsoid(options.dataset.mesh_pos)


    algebra_dim = 3
    metric = [1 for _ in range(algebra_dim)]
    algebra = CliffordAlgebra(metric)
    embed_dim = 2**algebra_dim

    self_attention_ga = SelfAttentionGA(algebra,embed_dim)
    






    my_gconv = GBottleneck(6, features_dim + hidden_dim + 8 , hidden_dim, last_hidden_dim,
                ellipsoid.adj_mat[2], activation=options.gconv_activation)



    for epoch in range(epoch_count, options.train.num_epochs):
        epoch_count+=1

        #TODO Reset loss

        for step,batch in enumerate(train_data_loader):
            batch = {k: v.cuda() if isinstance(v,torch.Tensor) else v for k, v in batch.items()}

            out_pretrained = evaluator.evaluate_step_mod(batch)
            x3 = out_pretrained['pred_coord'][2]
            print(f"x3 shape -> {x3.shape}")

            







     
            for key in out_pretrained.keys():
                print(key)

            exit()


        




    # evaluator.evaluate()


if __name__ == "__main__":
    main()
