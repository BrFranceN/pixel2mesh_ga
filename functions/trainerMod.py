import os
import time
from datetime import timedelta

import gc

import torch
from torch.profiler import profile, record_function, ProfilerActivity #ANALISI PERFORMANCE
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


from models.p2m import P2MModel
from models.losses.p2m import P2MLoss
from models.backbones import get_backbone
from models.layers.ga_refinement import ga_refinement, refinement
from models.layers.mlp_deformer import VertexDeformer
from algebra.cliffordalgebra import CliffordAlgebra



from utils.average_meter import AverageMeter
from utils.mesh import Ellipsoid
from utils.tensor import recursive_detach
from utils.vis.renderer import MeshRenderer



from datasets.imagenet import ImageNet
from datasets.shapenet import ShapeNet, get_shapenet_collate, ShapeNetImageFolder

import config
from tqdm import tqdm

import wandb #track the training





class TrainerGA():
    def __init__(self,options,logger,summary_writer,training=True,from_checkpoint=False,mlp=False,refinement=False):
        self.options = options
        self.logger = logger
        self.summary_writer = summary_writer
        self.training = training
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.min_loss = torch.tensor(float('inf'))
        self.mlp = mlp
        self.refinement = refinement



        if self.mlp:
            self.mlp_deformer = VertexDeformer()



        if from_checkpoint:
            self.my_epoch_count = self.options.my_epoch_count
            self.my_step_count = self.options.my_step_count
        else:
            self.my_epoch_count = 0
            self.my_step_count = 0        

        #prova da cancellare immediatamente:
        # self.training=False


        self.renderer = MeshRenderer(self.options.dataset.camera_f, self.options.dataset.camera_c,
                                    self.options.dataset.mesh_pos)
        self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)
        self.p2m = P2MModel(self.options.model, self.ellipsoid,
                                self.options.dataset.camera_f, self.options.dataset.camera_c,
                                self.options.dataset.mesh_pos).to(self.device)
        

        self.ckpt_file = os.path.abspath(options.checkpoint)

        checkpoint_p2m = self.load_checkpoint()
        self.p2m.load_state_dict(checkpoint_p2m['model'],strict=False)
        gcns_2= self.p2m.gcns[2]
        gconv = self.p2m.gconv




        # print("gcns_2 originale:")
        # for name, param in gcns_2.named_parameters():
        #     print(f"Parameter Name: {name}")
        #     print(f"Shape: {param.size()}")
        #     print(f"Values:\n{param.data}")

        #TODO PENSARE BENE QUANDO CARICARE O NO I PESI
        # if from_checkpoint == False:
        #     gcns_2_sd = gcns_2.state_dict()
        #     gconv_sd = gconv.state_dict()
        # else:
        #     gcns_2_sd = None
        #     gconv_sd = None


        print("load old weight p2m")
        gcns_2_sd = gcns_2.state_dict()
        gconv_sd = gconv.state_dict()





        #model parameters
        self.nn_encoder, self.nn_decoder = get_backbone(self.options.model)
        self.coord_dim = self.options.model.coord_dim
        self.hidden_dim = self.options.model.hidden_dim
        self.last_hidden_dim = self.options.model.last_hidden_dim
        self.features_dim = self.nn_encoder.features_dim + self.coord_dim
        algebra_dim = 3
        metric = [1 for _ in range(algebra_dim)]
        self.algebra = CliffordAlgebra(metric)
        self.embed_dim = 2**algebra_dim
        self.epoch_count = 0
        self.step_count = 0


        if self.mlp:
            self.model = self.mlp_deformer.to(self.device)
        elif self.refinement:
            self.model = refinement(self.hidden_dim,
                                    self.features_dim,
                                    self.coord_dim,
                                    self.last_hidden_dim,
                                    self.ellipsoid,
                                    self.options.model.gconv_activation,
                                    self.p2m,
                                    gcns_2_sd,
                                    gconv_sd
                                    ).to(self.device)

        else:
            self.model = ga_refinement(self.hidden_dim,
                                    self.features_dim,
                                    self.coord_dim,
                                    self.last_hidden_dim,
                                    self.ellipsoid,
                                    self.options.model.gconv_activation,
                                    self.p2m,
                                    gcns_2_sd,
                                    gconv_sd
                                    ).to(self.device)
        
        
        
        self.dataset = self.load_dataset(options.dataset,training)
        self.dataset_collate_fn = self.load_collate_fn(options.dataset,training)


        


       
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

        if from_checkpoint:
            #load checkpoint ga model
            if (options.checkpoint_ga is not None):
                self.checkpoint_file_ga = os.path.abspath(options.checkpoint_ga)
                self.logger.info("FROM CHECKPOINT MODE: %s" % self.checkpoint_file_ga)
                checkpoint_ga = torch.load(self.checkpoint_file_ga,encoding="bytes")
                self.model.load_state_dict(checkpoint_ga['model'],strict=False)
                self.optimizer.load_state_dict(checkpoint_ga['optimizer'])
            else:
                print("no check point ga found")
            # print(self.my_epoch_count)
            # print("Optimizer state_dict keys:", self.optimizer.state_dict().keys())
            # print("Optimizer param_groups:", self.optimizer.state_dict()["param_groups"])
            # print("Optimizer state:", self.optimizer.state_dict()["state"])

        self.criterion = P2MLoss(self.options.loss,self.ellipsoid).to(self.device)
        self.losses = AverageMeter()


        #training parameters
        self.time_start = time.time()

        # try to track training
        # Initialize W&B
        wandb.init(
            project="MV_FORMER",
            config={
                "learning_rate": options.optim.lr,
                "batch_size": options.train.batch_size,
                "num_epochs": options.train.num_epochs,
                "optimizer": options.optim.name,
                # Add more as needed
            },
            name="New_loss_2"
        )


        


    def train(self):

        print("epoche train : ", self.options.train.num_epochs)

        flush_step = 50
       

        # Create a new data loader for every epoch
        for epoch in tqdm(range(self.epoch_count,self.options.train.num_epochs)):
            self.epoch_count+=1
            # loss_values = []
            self.losses.reset()
            train_data_loader = DataLoader(self.dataset,
                                        batch_size=self.options.train.batch_size * self.options.num_gpus,
                                        num_workers=self.options.num_workers,
                                        pin_memory=self.options.pin_memory,
                                        shuffle=self.options.train.shuffle,
                                        collate_fn=self.dataset_collate_fn)
            
            for step,batch in enumerate(tqdm(train_data_loader)):
                batch = {k: v.cuda() if isinstance(v,torch.Tensor) else v for k, v in batch.items()}
                out_pretrained = self.pretrained_step(batch)
                out = self.train_step(batch,out_pretrained)
                self.step_count+=1

                # Tensorboard logging every summary_steps steps
                if self.step_count % self.options.train.summary_steps == 0:
                    self.train_summaries(batch, *out)

                # Save checkpoint every checkpoint_steps steps old
                if self.step_count % self.options.train.checkpoint_steps == 0:
                    if (self.losses.val < self.min_loss):
                        self.my_save_checkpoint()
                    else:
                        print("Loss bigger than before, saving skip ")


                if self.step_count % self.options.train.checkpoint_steps == 0:
                    if (self.losses.val < self.min_loss):
                        self.my_save_checkpoint()
                    else:
                        print("Loss bigger than before, saving skip ")
                
                #cleaning part
                del batch
                del out
                if self.step_count > 0 and self.step_count % flush_step == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    gc.collect()
                

            #end of an epoch
            # loss_values.append(self.losses.val / self.options.train.batch_size)
            if (self.losses.val < self.min_loss):
                self.my_save_checkpoint()
                # self.min_loss = self.losses.val # TODO THIS IS THE FIX, DECOMMENT
            else:
                print("Loss bigger than before, saving skip ")
            self.lr_scheduler.step()





    def my_save_checkpoint(self):
        obj = ({
            'epoch':self.epoch_count + self.my_epoch_count,
            "total_step_count": self.step_count + self.my_step_count, 
            'model_name':"ga_refinement",
            'model':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict()})
        name = "%06d_%06d" % (self.step_count + self.my_step_count, self.epoch_count + self.my_epoch_count)  
        complete_path = os.path.join(self.options.checkpoint_dir, "%s.pt" % name)
        self.logger.info("Dumping to checkpoint file: %s" % complete_path)
        torch.save(obj, complete_path)


        # Save checkpoint to W&B
        wandb.save(complete_path)
        wandb.log({"best_loss": self.min_loss.item()})




    #TODO FIX TO CONVERT IN YOUR CODE
    def get_latest_checkpoint(self):
        # this will automatically find the checkpoint with latest modified time
        checkpoint_list = []
        for dirpath, dirnames, filenames in os.walk(self.save_dir):
            for filename in filenames:
                if filename.endswith('.pt'):
                    file_path = os.path.abspath(os.path.join(dirpath, filename))
                    modified_time = os.path.getmtime(file_path)
                    checkpoint_list.append((file_path, modified_time))
        checkpoint_list = sorted(checkpoint_list, key=lambda x: x[1])
        return None if not checkpoint_list else checkpoint_list[-1][0]







    def train_step(self,batch,out_pretrained):
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        x2 = out_pretrained['pred_coord'][1]
        x_hidden = out_pretrained['my_var'][0]
        mesh3 = out_pretrained['pred_coord'][2]
        # ultimate_x = out_pretrained['my_var'][1]


        # here select the model 
        # if self.mlp:
        #     x4 = self.model(mesh3)
        # elif self.refinement:
        #     pass
        #     # x4 = self.model(ultimate_x)
        # else:
        #     x4 = self.model(x2,x_hidden,batch['images']) # old forward
        #     # x4 = self.model(out_pretrained,batch['images']) # new forward


        x4 = self.model(out_pretrained,batch['images'])



        out = {
            "pred_coord": [out_pretrained['pred_coord'][0], out_pretrained['pred_coord'][1], x4],
            "pred_coord_before_deform": [out_pretrained['pred_coord_before_deform'][0], out_pretrained['pred_coord_before_deform'][1], out_pretrained['pred_coord_before_deform'][2]],
            "reconst": out_pretrained['reconst']}


        #compute loss
        loss,loss_summary = self.criterion(out,batch)
        self.losses.update(loss.detach().cpu().item())

        # print("loss ", loss)
        # print("loss_summary ", loss_summary)


        #backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #RESULT ON PERFORMANCE 
        # if self.step_count % 50 == 0:
        #     print("risultati performance di un train step: ", end='')
        #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


        return recursive_detach(out), recursive_detach(loss_summary)


    def pretrained_step(self, input_batch):
        self.p2m.eval()

        #run inference
        with torch.no_grad():
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as pre_prof:
            images = input_batch['images']
            out = self.p2m(images)

        # if self.step_count % 50 == 0:
        #     print("Risultati sul pretrained step: ", end='')
        #     print(pre_prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        return out
    


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
        print(f"epoch: {self.epoch_count}, step: {self.step_count}, loss: {self.losses.val}, loss_avg: {self.losses.avg }")



        wandb.log({
                "epoch": self.epoch_count,
                "step": self.step_count,
                "loss": self.losses.val,
                "loss_avg": self.losses.avg
            })


    def load_checkpoint(self):
        self.logger.info("Loading checkpoint file (TRAINING_GA_mod): %s" % self.ckpt_file)
        try:
            return torch.load(self.ckpt_file)
        except UnicodeDecodeError:
            # to be compatible with old encoding methods
            return torch.load(self.ckpt_file, encoding="bytes")
        


    def load_collate_fn(self, dataset, training):
        if dataset.name == "shapenet":
            print("dataset shapenet collate load:")
            return get_shapenet_collate(dataset.shapenet.num_points)
        else:
            return default_collate



    @property
    def time_elapsed(self):
        return timedelta(seconds=time.time() - self.time_start)


    def load_dataset(self,dataset,training):
        # self.logger.info("Loading datasets: %s" % dataset.name)
        print("dataset name ->",dataset.name)
        self.logger.info("Loading datasets: %s" % dataset.name)
        if dataset.name == "shapenet":
            return ShapeNet(config.SHAPENET_ROOT, dataset.subset_train if training else dataset.subset_eval,
                            dataset.mesh_pos, dataset.normalization, dataset.shapenet)
        elif dataset.name == "shapenet_demo":
            return ShapeNetImageFolder(dataset.predict.folder, dataset.normalization, dataset.shapenet)
        elif dataset.name == "imagenet":
            return ImageNet(config.IMAGENET_ROOT, "train" if training else "val")
        raise NotImplementedError("Unsupported dataset")
    

