import time
from datetime import timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from functions.base import CheckpointRunner
from functions.evaluator import Evaluator
from models.classifier import Classifier
from models.losses.classifier import CrossEntropyLoss
from models.losses.p2m import P2MLoss
from models.p2m import P2MModel
from utils.average_meter import AverageMeter
from utils.mesh import Ellipsoid
from utils.tensor import recursive_detach
from utils.vis.renderer import MeshRenderer


#backup of trainer_ga
class trainer_ga(CheckpointRunner):
    
    def init_fn(self,shared_model=None,evaluator=None,checkpoint_file=None):

        # self.logger = logger
        # self.writer = writter
        # self.options = options
        self.evaluator = evaluator
        self.epoch_count = self.step_count = 0
        # self.train_data_loader = evaluator.train_data_ga() # old way to inizialite dataset
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        
        self.renderer = MeshRenderer(self.options.dataset.camera_f, self.options.dataset.camera_c,
                                    self.options.dataset.mesh_pos)
        self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)




        


        if (checkpoint_file != None):
            self.ckpt_file = os.path.abspath(checkpoint_file)
        

        
        
        
        
        
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

        ckpt = self.load_checkpoint()
        self.p2m_model.module.load_state_dict(ckpt, strict=False)
        

        #TODO Evaluators (think about if needed)



    def load_checkpoint(self):
        if self.ckpt_file is None:
            self.logger.info("Checkpoint file not found, skipping...")
            return None
        self.logger.info("Loading checkpoint file: %s" % self.ckpt_file)
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


            for step,batch in enumerate(self.train_data_loader):
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
                    self.train_summaries(batch, *out)

                # Save checkpoint every checkpoint_steps steps
                if self.step_count % self.options.train.checkpoint_steps == 0:
                    self.dump_checkpoint()
            

            # save checkpoint after each epoch
            self.dump_checkpoint()
            
            # TODO Run validation every test_epochs (DA VALUTARE COME IMPLEMENTARE)
            # if self.epoch_count % self.options.train.test_epochs == 0:
            #     self.test()

            # lr scheduler step
            self.lr_scheduler.step()
            print("after all OK")



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

        # Save results to log (to complete)