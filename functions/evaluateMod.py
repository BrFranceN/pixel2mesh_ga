
import os
import config



import torch
from torch.utils.data import DataLoader






import numpy as np



from utils.mesh import Ellipsoid
from utils.vis.renderer import MeshRenderer
from utils.average_meter import AverageMeter




from models.backbones import get_backbone
from models.layers.chamfer_wrapper import ChamferDist
from models.layers.ga_refinement import ga_refinement
from models.p2m import P2MModel



from datasets.imagenet import ImageNet
from datasets.shapenet import ShapeNet, get_shapenet_collate, ShapeNetImageFolder


from algebra.cliffordalgebra import CliffordAlgebra

from torch.utils.data.dataloader import default_collate







class EvaluateGA():
    def __init__(self,options,logger,summary_writer,training=False):
        self.options = options
        self.logger = logger
        self.summary_writer = summary_writer 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'    


        self.renderer = MeshRenderer(self.options.dataset.camera_f, self.options.dataset.camera_c,
                                    self.options.dataset.mesh_pos)
        self.chamfer = ChamferDist()
        self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)
        self.weighted_mean = self.options.test.weighted_mean

        self.num_classes = self.options.dataset.num_classes



        


        self.p2m = P2MModel(self.options.model, self.ellipsoid,
                                self.options.dataset.camera_f, self.options.dataset.camera_c,
                                self.options.dataset.mesh_pos).to(self.device)
        

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



        self.dataset = self.load_dataset(options.dataset,training)
        self.dataset_collate_fn = self.load_collate_fn(options.dataset,training)

        self.chechkpoint_file_p2m = os.path.abspath(options.checkpoint)
        self.checkpoint_file_ga = os.path.abspath(options.checkpoint_ga)




        # Evaluate step count, useful in summary
        self.evaluate_step_count = 0
        self.total_step_count = 0




    def evaluate(self):
        self.logger.info("Running GA evaluations...")

        #load checkpoints p2m
        self.logger.info("Loading checkpoint file (pixel 2 mesh): %s" % self.chechkpoint_file_p2m)
        checkpoint_p2m = torch.load(self.chechkpoint_file_p2m,encoding="bytes")
        self.p2m.load_state_dict(checkpoint_p2m['model'],strict=False)

        #load checkpoint ga model
        self.logger.info("Loading checkpoint file (ga_refinement): %s" % self.checkpoint_file_ga)
        checkpoint_ga = torch.load(self.checkpoint_file_ga,encoding="bytes")
        self.model.load_state_dict(checkpoint_ga['model'],strict=False)



        # clear evaluate_step_count, but keep total count uncleared
        self.evaluate_step_count = 0


        test_data_loader = DataLoader(self.dataset,
                                        batch_size=self.options.test.batch_size * self.options.num_gpus,
                                        num_workers=self.options.num_workers,
                                        pin_memory=self.options.pin_memory,
                                        shuffle=self.options.test.shuffle,
                                        collate_fn=self.dataset_collate_fn)
        
        self.chamfer_distance = [AverageMeter() for _ in range(self.num_classes)]
        self.f1_tau = [AverageMeter() for _ in range(self.num_classes)]
        self.f1_2tau = [AverageMeter() for _ in range(self.num_classes)]
        
        



        
        for step,batch in enumerate(test_data_loader):
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            #run evaluation step
            out = self.evaluate_step(batch)

            # Tensorboard logging every summary_steps steps
            if self.evaluate_step_count % self.options.test.summary_steps == 0:
                self.evaluate_summaries(batch, out)


            # add later to log at step 0
            self.evaluate_step_count += 1
            self.total_step_count += 1

        for key, val in self.get_result_summary().items():
            scalar = val
            if isinstance(val, AverageMeter):
                scalar = val.avg
            self.logger.info("Test [%06d] %s: %.6f" % (self.total_step_count, key, scalar))
            self.summary_writer.add_scalar("eval_" + key, scalar, self.total_step_count + 1)


    
    
    
    
    def average_of_average_meters(self, average_meters):
        s = sum([meter.sum for meter in average_meters])
        c = sum([meter.count for meter in average_meters])
        weighted_avg = s / c if c > 0 else 0.
        avg = sum([meter.avg for meter in average_meters]) / len(average_meters)
        ret = AverageMeter()
        if self.weighted_mean:
            ret.val, ret.avg = avg, weighted_avg
        else:
            ret.val, ret.avg = weighted_avg, avg
        return ret
    
    
    
    
    def get_result_summary(self):
        return {
            "cd": self.average_of_average_meters(self.chamfer_distance),
            "f1_tau": self.average_of_average_meters(self.f1_tau),
            "f1_2tau": self.average_of_average_meters(self.f1_2tau),
        }
    






    def evaluate_summaries(self,input_batch,out_summary):
        self.logger.info("Test Step %06d/%06d (%06d) " % (self.evaluate_step_count,
                                                          len(self.dataset) // (
                                                                  self.options.num_gpus * self.options.test.batch_size),
                                                          self.total_step_count,) \
                         + ", ".join([key + " " + (str(val) if isinstance(val, AverageMeter) else "%.6f" % val)
                                      for key, val in self.get_result_summary().items()]))
        
        self.summary_writer.add_histogram("eval_labels", input_batch["labels"].cpu().numpy(),
                                          self.total_step_count)     
        
        if self.renderer is not None:
            # Do visualization for the first 2 images of the batch
            render_mesh = self.renderer.p2m_batch_visualize(input_batch, out_summary, self.ellipsoid.faces)
            self.summary_writer.add_image("eval_render_mesh", render_mesh, self.total_step_count)       

    



    def evaluate_step(self,input_batch):

        self.model.eval()
        self.p2m.eval()

        #run inference
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
            

            pred_vertices = out["pred_coord"][-1]
            gt_points = input_batch["points_orig"]
            if isinstance(gt_points, list):
                gt_points = [pts.cuda() for pts in gt_points]
            self.evaluate_chamfer_and_f1(pred_vertices, gt_points, input_batch["labels"])
            
        return out




    def models_dict(self):
        return {'model':self.model}





    def evaluate_chamfer_and_f1(self, pred_vertices, gt_points, labels):
        # calculate accurate chamfer distance; ground truth points with different lengths;
        # therefore cannot be batched
        batch_size = pred_vertices.size(0)
        pred_length = pred_vertices.size(1)
        for i in range(batch_size):
            gt_length = gt_points[i].size(0)
            label = labels[i].cpu().item()
            d1, d2, i1, i2 = self.chamfer(pred_vertices[i].unsqueeze(0), gt_points[i].unsqueeze(0))
            d1, d2 = d1.cpu().numpy(), d2.cpu().numpy()  # convert to millimeter
            self.chamfer_distance[label].update(np.mean(d1) + np.mean(d2))
            self.f1_tau[label].update(self.evaluate_f1(d1, d2, pred_length, gt_length, 1E-4))
            self.f1_2tau[label].update(self.evaluate_f1(d1, d2, pred_length, gt_length, 2E-4))

    def evaluate_f1(self, dis_to_pred, dis_to_gt, pred_length, gt_length, thresh):
        recall = np.sum(dis_to_gt < thresh) / gt_length
        prec = np.sum(dis_to_pred < thresh) / pred_length
        return 2 * prec * recall / (prec + recall + 1e-8)
    
    
    
    
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
        

