import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import shutil
import numpy as np
from tqdm.rich import tqdm
from timm.utils import CheckpointSaver
from timm.models import resume_checkpoint
from easydict import EasyDict
from sklearn.metrics import auc, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import copy
from glob import glob
import datetime
from load_dataset import *
from abstract_dataset import DeepfakeAbstractBaseDataset
import yaml
from torch.utils.data import DataLoader
# from net.clip import CLIPDetector
from net.clip_effort import CLIPDetector
from utils.logger import Logger
from utils.init import setup
from utils.parameters import get_parameters
from utils.misc import *
import pdb
import random
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from torch.utils.data import ConcatDataset
import torch.utils.data as data
import torch.nn.functional as F
from metric_utils import cal_metrics
from utils.plot_utils import plot_score_distribution
import pickle
from timm.utils.model import unwrap_model, get_state_dict



def main():
    # --- 单卡设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # logger.info(f"Running on device: {device}")

    # Create model
    model = CLIPDetector()
    model = model.to(device)

    train_dataloaders, val_dataloaders, test_dataloaders = load_incremental_datasets(args)
    if args.model.ckpt_path is not None:
        checkpoint = torch.load(args.model.ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint if args.model.ckpt_path.endswith('pth') else checkpoint['state_dict']
        model.load_state_dict(state_dict)
        logger.info(f'Resumed model from {args.model.ckpt_path}')

    criterion = nn.BCEWithLogitsLoss().to(device)
    
    # Traing misc
    saver = None
    index = 0
    start_epoch = 1
    teacher_model = None

    initial_model_save_path = os.path.join(args.exam_dir, 'pretrained_model_initial.pth.tar')
    save_state = {
        'epoch': 0,
        'arch': type(model).__name__.lower(),
        'state_dict': get_state_dict(model, unwrap_model),
        }
    torch.save(save_state, initial_model_save_path)

    for index,train_dataloader in enumerate(train_dataloaders):
        cur_task = args.dataset.task_order[index]
        # evaluation_loaders = val_dataloaders[index]     # [(任务名, loader), ...]
        test_loader_bundle = test_dataloaders[index]    # 同结构

        # # --- 【修改】为 head 和 backbone 设置不同的学习率 ---
        # # 1. 定义 head 的学习率，通常是基础学习率的倍数（例如10倍）
        # base_lr = args.optimizer.params['lr']
        # head_lr_multiplier = 10.0
        # head_lr = base_lr * head_lr_multiplier

        # # 2. 分离 backbone 和 head 的参数
        # backbone_params = [p for n, p in model.named_parameters() if not n.startswith('head.') and p.requires_grad]
        # head_params = model.head.parameters()

        # # 3. 创建参数组
        # param_groups = [
        #     {'params': backbone_params},  # 使用优化器默认的基础学习率
        #     {'params': head_params, 'lr': head_lr} # 为 head 指定独立的学习率
        # ]

        # # 4. 使用参数组初始化优化器
        # optimizer = optim.__dict__[args.optimizer.name](param_groups, **args.optimizer.params)
        
        # logger.info(f"Optimizer setup: Backbone LR = {base_lr}, Head LR = {head_lr}")
        # # --- 修改结束 ---

        optimizer = optim.__dict__[args.optimizer.name](model.parameters(), **args.optimizer.params)
        scheduler = optim.lr_scheduler.__dict__[args.scheduler.name](optimizer, **args.scheduler.params)
        epoch_size = len(train_dataloader.dataset) // args.train.batch_size

        # 创建 CheckpointSaver
        os.makedirs(f"{args.exam_dir}/{cur_task}", exist_ok=True)
        os.makedirs(f"{args.exam_dir}/{cur_task}/vis_results", exist_ok=True)
        vis_dir = f"{args.exam_dir}/{cur_task}/vis_results"
        # 增加评估策略以及排序方式
        test_metric = args.test.metric.upper()
        if test_metric in ["AUC", "ACC"]:
            Decreasing = False
        elif test_metric in ["EER", "LOSS"]:
            Decreasing = True
        else:
            raise ValueError(f"Unsupported test metric: {test_metric}")
        
        saver = CheckpointSaver(
            model, optimizer,
            args=args,
            checkpoint_dir=f'{args.exam_dir}/{cur_task}/',
            recovery_dir=f'{args.exam_dir}/{cur_task}/',
            max_history=3,
            decreasing=Decreasing
        )
        # logger.info(f"Training task {index + 1}, dataset is {cur_task}")

        # val_loader_map = dict(evaluation_loaders)
        # val_loader = val_loader_map[cur_task]

        for epoch in range(start_epoch, args.train.epochs + 1):
            logger.info(f"=============================================================")
            logger.info(f"Training Task: {index + 1}, Dataset: {cur_task}, Epoch: {epoch}")
            logger.info(f"=============================================================")
            train(
                train_dataloader, model, teacher_model, criterion,
                optimizer, device, epoch, epoch_size
            )
            logger.info(f"================== START EVALUATION ===============================")

            # 在每个 epoch 结束后，对所有测试集进行评估，并根据“已见”数据集的平均性能保存模型
            seen_metrics = []
            pre_dict_list = []
            for eval_name, test_dataloader in test_loader_bundle:
                if eval_name in args.dataset.task_order:
                    logger.info(f"############ Seen Dataset Testing: {eval_name} ############")
                    results,pre_dict = test(test_dataloader, model, eval_name=eval_name, ckpt_task_name=cur_task, device=device, best=False)
                    seen_metrics.append(getattr(results, test_metric))
                else:
                    # continue
                    logger.info(f"############ Unseen Dataset Testing: {eval_name} ############")
                    results, pre_dict = test(test_dataloader, model, eval_name=eval_name, ckpt_task_name=cur_task, device=device, best=False)
                pre_dict_list.append(pre_dict)
            # 根据 "已见" 数据集的平均指标保存模型
    
            avg_metric = np.mean(seen_metrics)
            logger.info(f"Epoch {epoch} | Average metric on Seen datasets ({test_metric}): {avg_metric:.4f}")
            best_metric, best_epoch = saver.save_checkpoint(epoch, metric=avg_metric)
            if epoch == best_epoch:
                with open(vis_dir+'/data.pkl', 'wb') as f:
                    pickle.dump(pre_dict_list, f)
                for pd in pre_dict_list:
                    plot_score_distribution(pd, vis_dir)

            logger.info(f"Best metric so far: {best_metric:.4f} at epoch {best_epoch}\n")
            # logger.info("\n")

            scheduler.step()

        # 任务训练结束后，总结最佳表现并加载最佳模型以备下一个任务
        best_metric_so_far, best_epoch_so_far = saver.best_metric, saver.best_epoch
        logger.info("*************************************************************************")
        logger.info(f"Task {cur_task} finished. Best metric ({test_metric}) was {best_metric_so_far:.4f} at epoch {best_epoch_so_far}.")
        logger.info("*************************************************************************\n\n")

        # 加载当前任务的最佳模型，为下一个任务做准备
        best_model_path = os.path.join(saver.checkpoint_dir, 'model_best.pth.tar')

        if os.path.exists(best_model_path):
            logger.info(f"Loading best model from {best_model_path} for next task.")
            checkpoint = torch.load(best_model_path, map_location=device)
            state_dict = checkpoint['state_dict']
            if all(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        else:
            logger.warning(f"Could not find best model at {best_model_path}. Continuing with the current model.")

        # teacher_model = copy.deepcopy(model)
        # teacher_model = teacher_model.to(device)
        # for param in teacher_model.parameters():
        #     param.requires_grad = False
        # logger.info(f"Set teacher model trained from {cur_task} for next task.")

        start_epoch = 1

def train(train_dataloader, model, teacher_model, criterion,
          optimizer, device, epoch, epoch_size):
    criterion_mse = torch.nn.MSELoss().to(device)
    KL_criterion = torch.nn.KLDivLoss(size_average=True).to(device)
    acces = AverageMeter('Acc', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(epoch_size, [acces, losses])
    model.train()
    train_loader_len = len(train_dataloader)

    pbar = tqdm(
        train_dataloader, 
        desc=f"Epoch {epoch} [Training]", 
        total=len(train_dataloader),
        dynamic_ncols=True,
        leave=True
    )

    for i, datas in enumerate(pbar):
        images = datas['image'].to(device)
        targets = datas['label'].to(device)
        
        output = model(images)
        ce_loss = criterion(output['cls'], targets.float())
        loss = ce_loss
        # --- 如果有教师模型，则计算蒸馏损失 ---
        if teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
                teacher_features = teacher_outputs['feat']
                teacher_logits = teacher_outputs['cls']

            # 2. 计算特征MSE损失
            loss_mse = criterion_mse(output['feat'], teacher_features)
            
            # 3. 计算Logit KD损失 (KL散度)
            loss_kd = KL_criterion(torch.nn.LogSoftmax()(output['cls']),torch.nn.Softmax()(teacher_logits))
            # 将蒸馏损失加入总损失
            loss = ce_loss + loss_mse +loss_kd
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = output['prob']
        prediction = (probs >= 0.5).float()
        targets[targets > 0.5] = 1.0
        targets[targets < 0.5] = 0.0
        acc = (prediction == targets).float().mean()


        acces.update(acc.item(), targets.size(0))
        losses.update(loss.item(), targets.size(0))

        if i % args.train.print_interval == 0 or i == train_loader_len - 1:
            pbar.clear()
            logger.info(f'Epoch_{epoch}: {progress.display(i)}')
            pbar.refresh()

        pbar.set_postfix({
            'Loss': f"{losses.avg:.4f}",
            'Acc': f"{acces.avg:.4f}",
            'LR': f"{optimizer.param_groups[0]['lr']:.1e}"
        })

    logger.info(f"Epoch {epoch} Train Summary: Acc={acces.avg:.4f}, Loss={losses.avg:.4f}")

def test(test_dataloaders, model, eval_name, ckpt_task_name,  device, best=False):
    
    if best:
        ckpt_path = f'{args.exam_dir}/{ckpt_task_name}/model_best.pth.tar'
        logger.info(f"Loading best checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # 处理 DataParallel 包装的模型权重
        state_dict = checkpoint['state_dict']
        if all(k.startswith('module.') for k in state_dict.keys()):
            logger.info("Removing 'module.' prefix from state_dict keys.")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)

    model.eval()

    y_trues = []
    y_preds = []
    pre_dict = {}
    for i, datas in enumerate(tqdm(test_dataloaders, desc=f"Testing on {eval_name}")):
        with torch.no_grad():
            images = datas['image'].to(device)
            targets = datas['label'].to(device)
            
            outputs = model(images)
            probs = outputs['prob']
            y_preds.extend(probs.cpu().tolist())
            y_trues.extend(targets.cpu().tolist())

    results = cal_metrics(y_trues, y_preds, threshold= args.test.threshold)
    logger.info(f'#Total# ACC:{results.ACC:.4f}  AUC:{results.AUC:.4f}  EER:{100*results.EER:.2f}(Thresh:{results.Thre:.3f})')
    logger.info(f'--- Accuracy | Real: {results.TN:.4f}, Fake: {results.TP:.4f}')
    pre_dict['dataset'] = eval_name
    pre_dict['label'] = y_trues
    pre_dict['pred'] = y_preds
    return results, pre_dict

if __name__ == '__main__':
    args = get_parameters()
    setup(args)
    # --- 创建独立的实验目录 ---
    # 基于配置和时间戳创建实验目录
    config_name = args.config.split('/')[-1].replace('.yaml', '')
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    if args.exp_name:
        exp_name = args.exp_name + f'_{config_name}_{timestamp}' 
    else:
        exp_name = f'{config_name}_{timestamp}'
    prefix_root = '/home/OTS1/law/saved_models'
    project_name = os.path.basename(os.getcwd())
    args.exam_dir = os.path.join(prefix_root, project_name, exp_name)
    print('Experiment directory:', args.exam_dir)

    # 如果目录已存在，则删除（除非是恢复训练）
    if os.path.exists(args.exam_dir):
        shutil.rmtree(args.exam_dir)
    os.makedirs(args.exam_dir, exist_ok=True)

    # 备份代码和配置
    shutil.copytree("configs", f'{args.exam_dir}/Code/configs')
    # shutil.copytree("datasets", f'{args.exam_dir}/Code/datasets')
    shutil.copytree("net", f'{args.exam_dir}/Code/net')
    shutil.copytree("utils", f'{args.exam_dir}/Code/utils')

    logger = Logger(name='train', log_path=f'{args.exam_dir}/train.log')
    logger.info(args)
    logger.info(f"Experiment directory: {args.exam_dir}")
    # --- 修改结束 ---
    main()