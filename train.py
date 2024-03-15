import os
import sys
import argparse
import tomli
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from accelerate import Accelerator
from datetime import datetime

from utils import TotalLoss, GraspDataset, set_ground_truth
from models import ContactTransGrasp


def make_parser():
    parser = argparse.ArgumentParser(
        description="Train the model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--object_dir",
        default="data/grasps/scenes",
        help="The directory used for loading objects (.h5)."
    )
    parser.add_argument(
        "--mesh_dir",
        default="data/meshes/scenes",
        help="The directory used for loading meshes (.obj)."
    )
    
    return parser


def train(args):
    accelerator = Accelerator(split_batches=True)
    # prepare dataloader
    train_dataset = GraspDataset(
        args.object_dir,
        args.mesh_dir,
        2048,
        "train"
    )
    val_dataset = GraspDataset(
        args.object_dir,
        args.mesh_dir,
        2048,
        "val"
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )
    # instantiate model
    model = ContactTransGrasp(
        sample_num=args.sample_num,
        radius=args.radius,
        neighbor_num=args.neighbor_num,
        out_channels=args.out_channels,
        nhead=args.nhead,
        num_layers=args.num_layers,
        point_num=args.point_num
    )
    model.weights_init("normal")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        args.lr,
        weight_decay=args.weight_decay
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
    for epoch in tqdm(range(args.epochs)):      
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
                
        # lr_scheduler.step()
        
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
    
def main(argv=sys.argv[1: ]):
    
    parser = make_parser()
    args = parser.parse_args(argv)

    _config = tomli.load(open('config.toml', 'rb'))
   
    args.sample_num = int(_config["model"]["sample_num"])
    args.radius = float(_config["model"]["radius"])
    args.neighbor_num = int(_config["model"]["neighbor_num"])
    args.nhead = int(_config["model"]["nhead"])
    args.num_layers = int(_config["model"]["num_layers"])
    args.point_num = int(_config["model"]["point_num"])
     
    args.batch_size  = int(_config["train"]["batch_size"])
    args.epochs = int(_config["train"]["epochs"])
    args.lr = float(_config["train"]["lr"])
    
    args.gamma = float(_config["loss"]["gamma"])
    args.alpha = float(_config["loss"]["alpha"])
    args.beta = float(_config["loss"]["beta"])
    args.theta = float(_config["loss"]["theta"])
       
    train(args)
     
if __name__ == "__main__":
    main()
