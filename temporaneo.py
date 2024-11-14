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
from utils.average_meter import AverageMeter
from models.losses.p2m import P2MLoss


from models.backbones import get_backbone
from models.layers.gbottleneck import GBottleneck
from models.layers.gconv import GConv
from models.layers.gpooling import GUnpooling
from models.layers.gprojection import GProjection
from models.layers.selfattention_ga import SelfAttentionGA
from algebra.cliffordalgebra import CliffordAlgebra

from models.layers.ga_refinement import ga_refinement


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


class trainer_ga():
    def __init__(self):
        self.epoch_count = self.step_count = 0

    def 

def train_step():
    x2 = out_pretrained['pred_coord'][1]
    x = out_pretrained['my_var'][0]
    x_hidden = out_pretrained['my_var'][1]

def main():


    epoch_count = step_count = 0
    args = parse_args()
    logger, writer = reset_options(options, args, phase='eval')
    evaluator = Evaluator(options, logger, writer)


    #TODO 
    # 1. get the input batch data OK 
    # 2. analyze output OK 
    # 3. apply geomtric algebra on output and train OK 
    # 4. Set up loss 

    train_data_loader = evaluator.train_data_ga()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device is :",device)


    nn_encoder, nn_decoder = get_backbone(options.model)
    coord_dim = options.model.coord_dim
    hidden_dim = options.model.hidden_dim
    last_hidden_dim = options.model.last_hidden_dim
    features_dim = nn_encoder.features_dim + coord_dim
    ellipsoid = Ellipsoid(options.dataset.mesh_pos)


    algebra_dim = 3
    metric = [1 for _ in range(algebra_dim)]
    algebra = CliffordAlgebra(metric)
    embed_dim = 2**algebra_dim

    self_attention_ga = SelfAttentionGA(algebra,embed_dim)
    model = ga_refinement(hidden_dim,features_dim,coord_dim,last_hidden_dim,ellipsoid,options.model.gconv_activation).to(device)



    #LOSS SETUP
            
    
    # Setup a joint optimizer for the 2 models
    if options.optim.name == "adam":
        optimizer = torch.optim.Adam(
            params=list(model.parameters()),
            lr=options.optim.lr,
            betas=(options.optim.adam_beta1, 0.999),
            weight_decay=options.optim.wd
        )
    elif options.model.optim.name == "sgd":
        optimizer = torch.optim.SGD(
            params=list(model.parameters()),
            lr = options.optim.lr,
            momentum=options.optim.sgd_momentum,
            weight_decay=options.optim.wd
        )
    else:
        raise NotImplementedError("Your optimizer is not found")
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,options.optim.lr_step, options.optim.lr_factor
    )


    # create loss function
    critetrion = P2MLoss(options.loss,ellipsoid)
    losses = AverageMeter()


    
    
    
    
    




    for epoch in range(epoch_count, options.train.num_epochs):
        epoch_count+=1

        #TODO Reset loss
        losses.reset()

        print("OK")
        exit()


        for step,batch in enumerate(train_data_loader):
            batch = {k: v.cuda() if isinstance(v,torch.Tensor) else v for k, v in batch.items()}

            out_pretrained = evaluator.evaluate_step_mod(batch)
        

            train_step(batch,out_pretrained)

            x4 = model(x,x2,x_hidden).to(device)





            


        




    # evaluator.evaluate()

if __name__ == "__main__":
    main()