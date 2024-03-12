import os
import sys
import argparse
import tomli
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from accelerate import Accelerator
from datetime import datetime

from utils import TotalLoss, GraspDataset, set_ground_truth
from models import ContactTransGrasp2


def make_parser():
    parser = argparse.ArgumentParser(
        description="Train the model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--object_dir",
        default="data/grasps/scenes",
        help="The directory used for loading objects."
    )
    parser.add_argument(
        "--mesh_dir",
        default="data/meshes/scenes",
        help="The directory used for loading objects' meshes."
    )
    # parser.add_argument(
    #     "--gpus",
    #     type=int,
    #     default=4,
    #     help="The number of GPUs to use.")
    
    return parser


def train(args):
    accelerator = Accelerator(split_batches=True)
    # prepare the dataloader
    train_dataset = GraspDataset(
        args.object_dir,
        args.mesh_dir,
        args.point_num,
        'train'
    )
    val_dataset = GraspDataset(
        args.object_dir,
        args.mesh_dir,
        args.point_num,
        'val'
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )
    # instantiate the model
    model = ContactTransGrasp(
        fps_sample_num=args.fps_sample_num,
        ball_radius=args.ball_radius,
        ball_neighbor_num=args.ball_neighbor_num,
        feature_dim=args.feature_dim,
        head_num=args.head_num,
        layer_num=args.layer_num,
        contact_sample_num=args.contact_sample_num,
        gather_neighbor_num=args.gather_neighbor_num,
        soft_proj=True, 
        proj_neighbor_num=10,
        init_temp=1.0,
        is_temp_trainable=True,
        dropout=0.1,
        bn=True
    )
    model.weights_init('kaiming')
    # model.load_state_dict(torch.load("data/train_results/models/50_14_0.000040.pth"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        args.lr,
        weight_decay=0.05
    )
    lr_scheduler = StepLR(
        optimizer,
        step_size=50,
        gamma=0.1
    )
    criterion = TotalLoss(
        args.gamma,
        args.alpha,
        args.beta,
        args.theta
    )
    train_dataloader, val_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(
        train_dataloader,
        val_dataloader,
        model,
        optimizer,
        lr_scheduler
    )
    
    loss_items = []
    val_accuracy = []
    start_time = datetime.now()
    for epoch in range(0, args.epochs):      
        # training
        model.train()
        for step, (point_cloud, T, success) in enumerate(train_dataloader):
            optimizer.zero_grad()
            p1, temp, grasp, score = model(point_cloud)
            grasp_gt, class_gt = set_ground_truth(grasp.detach(), T, success)
            loss = criterion(
                xyz = point_cloud,
                sample_xyz = p1,
                temp = temp,
                grasp_pred = grasp,
                grasp_gt = grasp_gt,
                class_pred = score,
                class_gt = class_gt
            )
            accelerator.backward(loss)
            optimizer.step()
            if step == (len(train_dataloader) - 1):
                accelerator.print("Epoch [%d/%d], Loss: %.5f" % (epoch + 1, args.epochs, loss.item()))
                loss_items.append(loss.item())
                
        lr_scheduler.step()
        
        # validation   
        model.eval()   
        sum_accuracy = 0.0 
        with torch.no_grad(): 
            for step, (point_cloud, T, success) in enumerate(val_dataloader):
                p1, temp, grasp, score = model(point_cloud)
                grasp_gt, class_gt = set_ground_truth(grasp.detach(), T, success)
                cls_accuracy = torch.mean(((score > 0.5) == class_gt).float()).item()
                sum_accuracy += cls_accuracy
        
        accelerator.print("Accuracy: ", sum_accuracy / len(val_dataloader))
        val_accuracy.append(sum_accuracy / len(val_dataloader))
        
    accelerator.print("Training time: " + str(datetime.now() - start_time))
    # saving format: epochs + batch size + learning rate
    np.save("data/train_results/loss/%d_%d_%f.npy" % (args.epochs, args.batch_size, args.lr), np.array(loss_items)) 
    np.save("data/train_results/accuracy/%d_%d_%f.npy" % (args.epochs, args.batch_size, args.lr), np.array(val_accuracy)) 
    model = accelerator.unwrap_model(model) 
    accelerator.wait_for_everyone()     
    accelerator.save(model.state_dict(), "data/train_results/models/%d_%d_%f.pth" % (args.epochs, args.batch_size, args.lr)) 
    
    
def main(argv=sys.argv[1:]):
    
    parser = make_parser()
    args = parser.parse_args(argv)

    with open('config.toml', 'rb') as f:
        _config = tomli.load(f)
        args.point_num = int(_config['model']['point_num'])
        args.fps_sample_num = int(_config['model']['fps_sample_num'])
        args.ball_neighbor_num = int(_config['model']['ball_neighbor_num'])
        args.ball_radius = float(_config['model']['ball_radius'])
        args.contact_sample_num = int(_config['model']['contact_sample_num'])
        args.gather_neighbor_num = int(_config['model']['gather_neighbor_num'])
        args.feature_dim = int(_config['model']['feature_dim'])
        args.head_num = int(_config['model']['head_num'])
        args.layer_num = int(_config['model']['layer_num'])
        args.batch_size  = int(_config['train']['batch_size'])
        args.epochs = int(_config['train']['epochs'])
        args.lr = float(_config['train']['lr'])
        args.gamma = float(_config['loss']['gamma'])
        args.alpha = float(_config['loss']['alpha'])
        args.beta = float(_config['loss']['beta'])
        args.theta = float(_config['loss']['theta'])
       
    train(args)
     

if __name__ == "__main__":
    main()
