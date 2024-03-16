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

from utils import TotalLoss, GraspDataset, set_ground_truth, evaluate
from models import ContactTransGrasp

def make_parser():
    parser = argparse.ArgumentParser(
        description="Train the model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--object_dir",
        default="/home/wangjunlin/project/acronym2/data/grasps/scenes",
        help="The directory used for loading objects (.h5)."
    )
    parser.add_argument(
        "--mesh_dir",
        default="/home/wangjunlin/project/acronym2/data/meshes/scenes",
        help="The directory used for loading meshes (.obj)."
    )
    parser.add_argument(
        "--save_dir",
        default="train_results",
        help="The directory used for saving training results."
    )
    
    return parser

# def train(args):
#     accelerator = Accelerator(split_batches=True)
#     # prepare dataloader
#     train_dataset = GraspDataset(
#         args.object_dir,
#         args.mesh_dir,
#         2048,
#         "train"
#     )
#     val_dataset = GraspDataset(
#         args.object_dir,
#         args.mesh_dir,
#         2048,
#         "val"
#     )
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True,
#         drop_last=False
#     )
#     val_dataloader = DataLoader(
#         val_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True,
#         drop_last=False
#     )
#     # instantiate model
#     model = ContactTransGrasp(
#         sample_num=args.sample_num,
#         radius=args.radius,
#         neighbor_num=args.neighbor_num,
#         out_channels=args.out_channels,
#         nhead=args.nhead,
#         num_layers=args.num_layers,
#         point_num=args.point_num
#     )
#     model.weights_init("normal")
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         args.lr,
#         weight_decay=args.weight_decay
#     )
#     lr_scheduler = StepLR(
#         optimizer,
#         step_size=50,
#         gamma=0.1
#     )
#     criterion = TotalLoss(
#         args.gamma,
#         args.alpha,
#         args.beta,
#         args.theta
#     )
#     train_dataloader, val_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(
#         train_dataloader,
#         val_dataloader,
#         model,
#         optimizer,
#         lr_scheduler
#     )
    
#     accelerator.print(f"Training with batch size {args.batch_size} and learning rate {args.lr}.")
#     loss_items = []
#     start_time = datetime.now()
#     for epoch in tqdm(range(args.epochs)):      
#         # training
#         model.train()
#         for step, (point_cloud, T, success) in enumerate(train_dataloader):
#             optimizer.zero_grad()
#             p1, temp, grasp, score = model(point_cloud)
#             grasp_gt, class_gt = set_ground_truth(grasp.detach(), T, success)
#             loss = criterion(
#                 xyz = point_cloud,
#                 sample_xyz = p1,
#                 temp = temp,
#                 grasp_pred = grasp,
#                 grasp_gt = grasp_gt,
#                 class_pred = score,
#                 class_gt = class_gt
#             )
#             accelerator.backward(loss)
#             optimizer.step()
#             if step == (len(train_dataloader) - 1):
#                 accelerator.print("Epoch [%d/%d], Loss: %.5f" % (epoch + 1, args.epochs, loss.item()))
#                 loss_items.append(loss.item())
                
#         # lr_scheduler.step()
        
#         # validation   
#         model.eval()   
#         sum_trans_error = 0.0
#         sum_rot_error = 0.0
#         sum_accuracy = 0.0 
#         with torch.no_grad(): 
#             for step, (point_cloud, T, success) in enumerate(val_dataloader):
#                 p1, temp, grasp, score = model(point_cloud)
#                 grasp_gt, class_gt = set_ground_truth(grasp, T, success)
#                 trans_error, rot_error, cls_accuracy = evaluate(grasp, grasp_gt, score, class_gt)
#                 sum_trans_error += trans_error
#                 sum_rot_error += rot_error
#                 sum_accuracy += cls_accuracy
        
#         mean_trans_error = sum_trans_error / len(val_dataloader)
#         mean_rot_error = sum_rot_error / len(val_dataloader)
#         mean_accuracy = sum_accuracy / len(val_dataloader)
#         accelerator.print(f"Translation Error: {mean_trans_error}; \
#             Rotation Error: {mean_rot_error}; Accuracy: {mean_accuracy}")
        
#     accelerator.print(f"Training time: {str(datetime.now() - start_time)}")
#     # save loss
#     loss_save_path = os.path.join(args.save_dir, f"{args.epochs}_{args.lr}.npy")
#     np.save(loss_save_path, np.array(loss_items)) 
#     # save model
#     model = accelerator.unwrap_model(model) 
#     accelerator.wait_for_everyone()
#     model_save_path = os.path.join(args.save_dir, f"{args.epochs}_{args.lr}.pth")     
#     accelerator.save(model.state_dict(), model_save_path) 
#     accelerator.print("Finished.")

def train(args):
    # torch.cuda.set_device(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # prepare dataloader
    train_dataset = GraspDataset(
        args.object_dir,
        args.mesh_dir,
        1024,
        "train"
    )
    val_dataset = GraspDataset(
        args.object_dir,
        args.mesh_dir,
        1024,
        "val"
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
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
    ).to(device)
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
    ).to(device)
    
    print(f"Training with batch size {args.batch_size} and learning rate {args.lr}.")
    loss_items = []
    start_time = datetime.now()
    for epoch in tqdm(range(args.epochs)):      
        # training
        model.train()
        for step, (point_cloud, T, success) in enumerate(train_dataloader):
            optimizer.zero_grad()
            p1, temp, grasp, score = model(point_cloud.to(device))
            grasp_gt, class_gt = set_ground_truth(grasp.to(device), T.to(device), success.to(device))
            loss = criterion(
                xyz = point_cloud.to(device),
                sample_xyz = p1.to(device),
                temp = temp.to(device),
                grasp_pred = grasp.to(device),
                grasp_gt = grasp_gt.to(device),
                class_pred = score.to(device),
                class_gt = class_gt.to(device)
            )
            loss.backward()
            optimizer.step()
            if step == (len(train_dataloader) - 1):
                print("Epoch [%d/%d], Loss: %.5f" % (epoch + 1, args.epochs, loss.item()))
                loss_items.append(loss.item())
                
        # lr_scheduler.step()
        
        # validation   
        model.eval()   
        sum_trans_error = 0.0
        sum_rot_error = 0.0
        sum_accuracy = 0.0 
        with torch.no_grad(): 
            for step, (point_cloud, T, success) in enumerate(val_dataloader):
                p1, temp, grasp, score = model(point_cloud.to(device))
                grasp_gt, class_gt = set_ground_truth(grasp.to(device), T.to(device), success.to(device))
                trans_error, rot_error, cls_accuracy = evaluate(grasp, grasp_gt, score, class_gt)
                sum_trans_error += trans_error
                sum_rot_error += rot_error
                sum_accuracy += cls_accuracy
        
        mean_trans_error = sum_trans_error / len(val_dataloader)
        mean_rot_error = sum_rot_error / len(val_dataloader)
        mean_accuracy = sum_accuracy / len(val_dataloader)
        print(f"Translation Error: {mean_trans_error}; \
            Rotation Error: {mean_rot_error}; Accuracy: {mean_accuracy}")
        
    print(f"Training time: {str(datetime.now() - start_time)}")
    # save loss
    loss_save_path = os.path.join(args.save_dir, f"{args.epochs}_{args.lr}.npy")
    np.save(loss_save_path, np.array(loss_items)) 
    # # save model
    # model = accelerator.unwrap_model(model) 
    # accelerator.wait_for_everyone()
    # model_save_path = os.path.join(args.save_dir, f"{args.epochs}_{args.lr}.pth")     
    # accelerator.save(model.state_dict(), model_save_path) 
    # accelerator.print("Finished.")
    
def main(argv=sys.argv[1: ]):
    
    parser = make_parser()
    args = parser.parse_args(argv)
    
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    _config = tomli.load(open('config.toml', 'rb'))
   
    args.sample_num = list(_config["model"]["sample_num"])
    args.radius = list(_config["model"]["radius"])
    args.neighbor_num = list(_config["model"]["neighbor_num"])
    args.out_channels = list(_config["model"]["out_channels"])
    args.nhead = int(_config["model"]["nhead"])
    args.num_layers = int(_config["model"]["num_layers"])
    args.point_num = int(_config["model"]["point_num"])
     
    args.batch_size  = int(_config["train"]["batch_size"])
    args.epochs = int(_config["train"]["epochs"])
    args.lr = float(_config["train"]["lr"])
    args.weight_decay = float(_config["train"]["weight_decay"])
    
    args.gamma = float(_config["loss"]["gamma"])
    args.alpha = float(_config["loss"]["alpha"])
    args.beta = float(_config["loss"]["beta"])
    args.theta = float(_config["loss"]["theta"])
       
    train(args)
     
if __name__ == "__main__":
    main()
