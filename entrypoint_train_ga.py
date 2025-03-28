
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
import os
import time
from datetime import timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from functions.evaluator import Evaluator
from functions.base import CheckpointRunner
from functions.saver import CheckpointSaver



from functions.evaluator import Evaluator
from options import update_options, options, reset_options
from utils.mesh import Ellipsoid
from utils.average_meter import AverageMeter
from utils.tensor import recursive_detach
from utils.vis.renderer import MeshRenderer



from models.p2m import P2MModel
from models.backbones import get_backbone
from models.layers.gbottleneck import GBottleneck
from models.layers.gconv import GConv
from models.layers.gpooling import GUnpooling
from models.layers.gprojection import GProjection
from models.layers.selfattention_ga import SelfAttentionGA
from models.losses.p2m import P2MLoss
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


#TODO trtansform in class paradigm
class trainer_ga(CheckpointRunner):
    
    def init_fn(self,shared_model=None,ckp_file=None):

        # self.logger = logger
        # self.writer = writter
        # self.options = options
        # self.evaluator = evaluator


        self.saver = CheckpointSaver(self.logger, checkpoint_dir=str(self.options.checkpoint_dir),
                                         checkpoint_file=self.options.checkpoint)
        

        self.epoch_count = self.step_count = 0
        # self.train_data_loader = evaluator.train_data_ga() # old way to inizialite dataset
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        
        self.renderer = MeshRenderer(self.options.dataset.camera_f, self.options.dataset.camera_c,
                                    self.options.dataset.mesh_pos)
        self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)




        if (ckp_file != None):
            self.ckpt_file = os.path.abspath(ckp_file)
        
        
        self.nn_encoder, self.nn_decoder = get_backbone(self.options.model)
        self.coord_dim = self.options.model.coord_dim
        self.hidden_dim = self.options.model.hidden_dim
        self.last_hidden_dim = self.options.model.last_hidden_dim
        self.features_dim = self.nn_encoder.features_dim + self.coord_dim

        algebra_dim = 3
        metric = [1 for _ in range(algebra_dim)]
        self.algebra = CliffordAlgebra(metric)
        self.embed_dim = 2**algebra_dim

        self.self_attention_ga = SelfAttentionGA(self.algebra,self.embed_dim)
        self.model = ga_refinement(self.hidden_dim,
                                   self.features_dim,
                                   self.coord_dim,
                                   self.last_hidden_dim,
                                   self.ellipsoid,
                                   self.options.model.gconv_activation).to(self.device)
        

        print("model n parameters:")
        print("GA REFINEMENT -> ", self.model)
        # parametr estimation
        total_params = sum(p.numel() for p in self.model.parameters())
        param_size_bytes = total_params * 4  # Assuming float32
        model_size_mb = param_size_bytes / (1024 ** 2)
        print(f"Total Parameters: {total_params}")
        print(f"Model Size: {model_size_mb:.2f} MB")
        

        
        if self.options.optim.name == "adam":
            self.optimizer = torch.optim.Adam(
                params=list(self.model.parameters()),
                lr=self.options.optim.lr,
                betas=(self.options.optim.adam_beta1, 0.999),
                weight_decay=self.options.optim.wd
            )
        elif self.options.optim.name == "sgd":
            self.optimizer = torch.optim.SGD(
                params=list(self.model.parameters()),
                lr=self.options.optim.lr,
                momentum=self.options.optim.sgd_momentum,
                weight_decay=self.options.optim.wd
            )
        else:
            raise NotImplementedError("Your optimizer is not found")
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.options.optim.lr_step, self.options.optim.lr_factor
        )

        self.criterion = P2MLoss(self.options.loss,self.ellipsoid).to(self.device)
        self.losses = AverageMeter()



        #pretrained model

        self.gpus = list(range(self.options.num_gpus))
        
        self.p2m_model = P2MModel(self.options.model, self.ellipsoid,
                                self.options.dataset.camera_f, self.options.dataset.camera_c,
                                self.options.dataset.mesh_pos)
        

        self.p2m_model = torch.nn.DataParallel(self.p2m_model, device_ids=self.gpus).cuda()

        ckpt = self.load_checkpoint_2()
        missing_keys = self.p2m_model.module.load_state_dict(ckpt, strict=False)
        print("MISSING KEYS : ", missing_keys)


        print("model n parameters:")
        # parametr estimation
        total_params = sum(p.numel() for p in self.model.parameters())
        param_size_bytes = total_params * 4  # Assuming float32
        model_size_mb = param_size_bytes / (1024 ** 2)
        print(f"Total Parameters: {total_params}")
        print(f"Model Size: {model_size_mb:.2f} MB")
        

        #TODO Evaluators (think about if needed)



    def load_checkpoint_2(self):
        if self.ckpt_file is None:
            self.logger.info("Checkpoint file not found, skipping...")
            return None
        self.logger.info("Loading checkpoint file (TRAINING_GA): %s" % self.ckpt_file)
        try:
            return torch.load(self.ckpt_file)
        except UnicodeDecodeError:
            # to be compatible with old encoding methods
            return torch.load(self.ckpt_file, encoding="bytes")


    def models_dict(self):
        return {'model': self.model}

    def optimizers_dict(self):
        return {'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler}

    def train(self):

        # print(f"self.options.train.num_epochs -> {self.options.train.num_epochs}")
        # exit()

        for epoch in range(self.epoch_count, self.options.train.num_epochs):
            
            self.epoch_count+=1

            self.losses.reset()


            # Create a new data loader for every epoch
            train_data_loader = DataLoader(self.dataset,
                                        batch_size=self.options.train.batch_size * self.options.num_gpus,
                                        num_workers=self.options.num_workers,
                                        pin_memory=self.options.pin_memory,
                                        shuffle=self.options.train.shuffle,
                                        collate_fn=self.dataset_collate_fn)


            for step,batch in enumerate(train_data_loader):
                batch = {k: v.cuda() if isinstance(v,torch.Tensor) else v for k, v in batch.items()}
                # print(f"batch -> {batch.keys()}")
                # print(f"batch -> {len(batch)}")
                # exit()

                
             

                # print("step :",step)
                # out_pretrained = self.evaluator.evaluate_step_mod(batch)
                out_pretrained = self.pretrained_step(batch)
            

                out = self.train_step(batch,out_pretrained)

                self.step_count+=1

                # Tensorboard logging every summary_steps steps
                if self.step_count % self.options.train.summary_steps == 0:
                    print("entro?")
                    self.train_summaries(batch, *out)

                # Save checkpoint every checkpoint_steps steps
                if self.step_count % self.options.train.checkpoint_steps == 0:
                    print("entro dump?")
                    self.dump_checkpoint() #da cambiare non penso funzioni
            

            # save checkpoint after each epoch
            self.dump_checkpoint()
            
            # TODO Run validation every test_epochs (DA VALUTARE COME IMPLEMENTARE)
            # if self.epoch_count % self.options.train.test_epochs == 0:
            #     self.test()

            # lr scheduler step
            self.lr_scheduler.step()



    def pretrained_step(self, input_batch):
        self.p2m_model.eval()

        # Run inference
        with torch.no_grad():
            images = input_batch['images']
            out = self.p2m_model(images)
        return out

    def train_step(self,batch,out_pretrained):
        x2 = out_pretrained['pred_coord'][1]
        x = out_pretrained['my_var'][0]
        x_hidden = out_pretrained['my_var'][1]

        x4 = self.model(x,x2,x_hidden)



        out = {
            "pred_coord": [out_pretrained['pred_coord'][0], out_pretrained['pred_coord'][1], x4],
            "pred_coord_before_deform": [out_pretrained['pred_coord_before_deform'][0], out_pretrained['pred_coord_before_deform'][1], out_pretrained['pred_coord_before_deform'][2]],
            "reconst": None,}


        #compute loss
        loss,loss_summary = self.criterion(out,batch)
        self.losses.update(loss.detach().cpu().item())

        #backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        return recursive_detach(out), recursive_detach(loss_summary)
    

    def train_summaries(self, input_batch, out_summary, loss_summary):
        if self.renderer is not None:
            render_mesh = self.renderer.p2m_batch_visualize(input_batch, out_summary, self.ellipsoid.faces)
            self.summary_writer.add_image("render_mesh", render_mesh, self.step_count)
            self.summary_writer.add_histogram("length_distribution", input_batch["length"].cpu().numpy(),
                                              self.step_count)
            

        # Debug info for filenames
        self.logger.debug(input_batch["filename"])

        # Save results in Tensorboard
        for k, v in loss_summary.items():
            self.summary_writer.add_scalar(k, v, self.step_count)


        # Save results to log
        self.logger.info("Epoch %03d, Step %06d/%06d, Time elapsed %s, Loss %.9f (%.9f)" % (
            self.epoch_count, self.step_count,
            self.options.train.num_epochs * len(self.dataset) // (
                        self.options.train.batch_size * self.options.num_gpus),
            self.time_elapsed, self.losses.val, self.losses.avg))
 
    










def main():


    epoch_count = step_count = 0
    args = parse_args()
    logger, writer = reset_options(options, args)
    # logger, writer = reset_options(options, args, phase='eval')
    # evaluator = Evaluator(options, logger, writer)
    print(f"CHECKPOINTS -> {options.checkpoint}")
    print(f"DATASET -> {options.dataset}")
    print(f"DATASET name -> {options.dataset.name}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    my_trainer = trainer_ga(options,logger,writer,ckp_file=options.checkpoint,old_style=False)

    my_trainer.train()

    print("train OK")





    #TODO
    # 1. get the input batch data OK 
    # 2. analyze output OK 
    # 3. apply geomtric algebra on output and train OK 
    # 4. Set up loss 
    # 5. eliminate the superfluos
 


    # my_trainer = trainer_ga(evaluator,options,logger_train,writer_train)

    # extra_parameters = {
    #     "evaluator": evaluator,
    #                     }


    # my_trainer = trainer_ga(options,logger_train,writer_train,evaluator=evaluator,checkpoint_file=options.checkpoint)
    # my_trainer = trainer_ga(options,logger_train,writer_train,checkpoint_file=options.checkpoint)





            


        




    # evaluator.evaluate()

if __name__ == "__main__":
    main()
