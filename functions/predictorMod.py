import os
import config
import random
from tqdm import tqdm

import imageio
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


from models.layers.ga_refinement import ga_refinement
from models.backbones import get_backbone
from models.p2m import P2MModel
from utils.mesh import Ellipsoid
from utils.vis.renderer import MeshRenderer



from datasets.imagenet import ImageNet
from datasets.shapenet import ShapeNet, get_shapenet_collate, ShapeNetImageFolder



from algebra.cliffordalgebra import CliffordAlgebra



class PredictorGA():
    def __init__(self,options,logger,writer,training=False):
        self.options = options
        self.logger = logger
        self.writer = writer
        self.training = training
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  


        dataset = options.dataset
        self.dataset = self.load_dataset(dataset, training)
        self.dataset_collate_fn = self.load_collate_fn(dataset, training)


        self.gpu_inference = self.options.num_gpus > 0

        self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)
        self.p2m = P2MModel(self.options.model, self.ellipsoid,
                                  self.options.dataset.camera_f, self.options.dataset.camera_c,
                                  self.options.dataset.mesh_pos).to(self.device)
        
        if self.gpu_inference:
            print("Inference mod")
            self.renderer = MeshRenderer(self.options.dataset.camera_f, self.options.dataset.camera_c,
                                             self.options.dataset.mesh_pos)
        
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


        self.model = ga_refinement(self.hidden_dim,
                            self.features_dim,
                            self.coord_dim,
                            self.last_hidden_dim,
                            self.ellipsoid,
                            self.options.model.gconv_activation).to(self.device)
        

        self.chechkpoint_file_p2m = os.path.abspath(options.checkpoint)
        self.checkpoint_file_ga = os.path.abspath(options.checkpoint_ga)
        



    def predict_step(self,input_batch):
        self.model.eval()
        self.p2m.eval()

        with torch.no_grad():
            #follow the pipeline
            images = input_batch['images']
            out_tmp = self.p2m(images)
            x2 = out_tmp['pred_coord'][1]
            x = out_tmp['my_var'][0]
            x_hidden = out_tmp['my_var'][1]
            x4 =  self.model(x,x2,x_hidden)

            out = {
                "pred_coord": [out_tmp['pred_coord'][0], out_tmp['pred_coord'][1], x4],
                "pred_coord_before_deform": [out_tmp['pred_coord_before_deform'][0], out_tmp['pred_coord_before_deform'][1], out_tmp['pred_coord_before_deform'][2]],
                "reconst": out_tmp['reconst']}
            
            self.save_inference_results(input_batch,out)



    def predict(self):
        self.logger.info("Running predictions GA ... ")

        #load checkpoint p2m
        self.logger.info("Loading checkpoint file (pixel 2 mesh): %s" % self.chechkpoint_file_p2m)
        checkpoint_p2m = torch.load(self.chechkpoint_file_p2m,encoding="bytes")
        self.p2m.load_state_dict(checkpoint_p2m['model'],strict=False)
        #load checkpoint ga model
        self.logger.info("Loading checkpoint file (ga_refinement): %s" % self.checkpoint_file_ga)
        checkpoint_ga = torch.load(self.checkpoint_file_ga,encoding="bytes")
        self.model.load_state_dict(checkpoint_ga['model'],strict=False)

        predict_data_loader = DataLoader(self.dataset,
                                         batch_size=self.options.test.batch_size,
                                         pin_memory=self.options.pin_memory,
                                         collate_fn=self.dataset_collate_fn)
        
        for step, batch in enumerate(predict_data_loader):
            # print("qui arrivo?")
            self.logger.info("Predicting [%05d/%05d]" % (step * self.options.test.batch_size, len(self.dataset)))

            if self.gpu_inference:
                # Send input to GPU
                # print("e qua?")
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            self.predict_step(batch)
    
    def save_inference_results(self,inputs,outputs):
        batch_size = inputs["images"].size(0)
        for i in range(batch_size):
            basename, ext = os.path.splitext(inputs["filepath"][i])
            mesh_center = np.mean(outputs["pred_coord_before_deform"][0][i].cpu().numpy(), 0)
            verts = [outputs["pred_coord"][k][i].cpu().numpy() for k in range(3)]
            for k, vert in enumerate(verts):
                meshname = basename + ".%d.obj" % (k + 1)
                vert_v = np.hstack((np.full([vert.shape[0], 1], "v"), vert))
                mesh = np.vstack((vert_v, self.ellipsoid.obj_fmt_faces[k]))
                np.savetxt(meshname, mesh, fmt='%s', delimiter=" ")

            if self.gpu_inference:
                # generate gif here

                color_repo = ['light_blue', 'purple', 'orange', 'light_yellow']

                rot_degree = 10
                rot_radius = rot_degree / 180 * np.pi
                rot_matrix = np.array([
                    [np.cos(rot_radius), 0, -np.sin(rot_radius)],
                    [0., 1., 0.],
                    [np.sin(rot_radius), 0, np.cos(rot_radius)]
                ])
                writer = imageio.get_writer(basename + ".gif", mode='I')
                color = random.choice(color_repo)
                for _ in tqdm(range(360 // rot_degree), desc="Rendering sample %d" % i):
                    image = inputs["images_orig"][i].cpu().numpy()
                    ret = image
                    for k, vert in enumerate(verts):
                        vert = rot_matrix.dot((vert - mesh_center).T).T + mesh_center
                        rend_result = self.renderer.visualize_reconstruction(None,
                                                                                vert + \
                                                                                np.array(
                                                                                    self.options.dataset.mesh_pos),
                                                                                self.ellipsoid.faces[k],
                                                                                image,
                                                                                mesh_only=True,
                                                                                color=color)
                        ret = np.concatenate((ret, rend_result), axis=2)
                        verts[k] = vert
                    ret = np.transpose(ret, (1, 2, 0))
                    writer.append_data((255 * ret).astype(np.uint8))
                writer.close()


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
    

    def load_collate_fn(self, dataset, training):
        if dataset.name == "shapenet":
            print("dataset shapenet collate load:")
            return get_shapenet_collate(dataset.shapenet.num_points)
        else:
            return default_collate
            
            
        