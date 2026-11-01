import argparse
import warnings
import sys
import os
import open_clip
from open_clip import get_tokenizer
from plot.plot_utils import plot_score_distribution , plot_feature_tsne, plot_feature_tsne_multi
import numpy as np
from sklearn.metrics import accuracy_score
import collections
import collections.abc
collections.MutableMapping = collections.abc.MutableMapping
def get_parser():
    """
    创建并配置命令行参数解析器，返回解析后的参数对象。

    Returns:
        argparse.Namespace: 包含解析后命令行参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description="AIGCDetection @cby Training")
    parser.add_argument("--model_name", default='efficientnet_b0', help="Setting the model name", type=str)
    parser.add_argument("--embedding_size", default=1024, help="Setting the embedding_size", type=int)
    parser.add_argument("--pre_layer", default=None, help="Setting the pre_layer: srm or dct", type=str)
    parser.add_argument("--num_classes", default=2, help="Setting the num classes", type=int)
    parser.add_argument('--freeze_extractor', action='store_true', help='Whether to freeze extractor?')
    parser.add_argument("--model_path", default=None, help="Setting the model path", type=str)
    parser.add_argument('--no_strict', action='store_true', help='Whether to load model without strict?')
    parser.add_argument("--root_path", default='/home/law/HDD/hjs/DRCT/dataset/MSCOCO',
                        help="Setting the root path for dataset loader", type=str)
    parser.add_argument("--fake_root_path", default='/home/law/HDD/hjs/DRCT/dataset/AIGC_MSCOCO',
                        help="Setting the fake root path for dataset loader", type=str)
    parser.add_argument('--is_dire', action='store_true', help='Whether to using DIRE?')
    parser.add_argument('--post_aug_mode', default=None, help='Stetting the post aug mode during test phase.')
    parser.add_argument('--save_txt', default=None, help='Stetting the save_txt path.')
    parser.add_argument("--fake_indexes", default='1', 
                        help="Setting the fake indexes, multi class using '1,2,3,...' ", type=str)
    parser.add_argument("--dataset_name", default='MSCOCO', help="Setting the dataset name", type=str)
    parser.add_argument("--device_id", default='0',
                        help="Setting the GPU id, multi gpu split by ',', such as '0,1,2,3'", type=str)
    parser.add_argument("--input_size", default=224, help="Image input size", type=int)
    parser.add_argument('--is_crop', action='store_true', help='Whether to crop image?')
    parser.add_argument('--is_spectrum', action='store_true', help='Whether to using spectrum?')
    parser.add_argument("--batch_size", default=64, help="Setting the batch size", type=int)
    parser.add_argument("--epoch_start", default=0, help="Setting the epoch start", type=int)
    parser.add_argument("--num_epochs", default=50, help="Setting the num epochs", type=int)
    parser.add_argument("--num_workers", default=4, help="Setting the num workers", type=int)
    parser.add_argument('--is_warmup', action='store_true', help='Whether to using lr warmup')
    parser.add_argument("--lr", default=1e-3, help="Setting the learning rate", type=float)
    parser.add_argument("--save_flag", default='', help="Setting the save flag", type=str)
    parser.add_argument("--sampler_mode", default='', help="Setting the sampler mode", type=str)
    parser.add_argument('--is_test', action='store_true', help='Whether to predict the test set?')
    parser.add_argument('--is_amp', action='store_true', help='Whether to using amp autocast(使用混合精度加速)?')
    parser.add_argument("--inpainting_dir", default='full_inpainting', help="rec_image dir", type=str)
    parser.add_argument("--threshold", default=0.5, help="Setting the valid or testing threshold.", type=float)
    parser.add_argument('--alpha', default=0.3, type=float, help="Setting the alpha for contrastive learning")
    parser.add_argument('--pos_margin', default=0.0, type=float)
    parser.add_argument('--neg_margin', default=1.0, type=float)
    parser.add_argument('--tau', default=0.5, type=float)
    parser.add_argument('--loss_name', default='ContrastiveLoss', type=str)
    parser.add_argument('--use_miner', action='store_true', help='Whether to using miner')
    parser.add_argument('--memory_size', default=None, type=int)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args


warnings.filterwarnings("ignore")

args = get_parser()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)

import torch
import torch.nn as nn
import torch.optim as optim
from catalyst.data import BalanceClassSampler
import time
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import gc
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score, f1_score
import pytorch_warmup as warmup

from utils.utils import Logger, AverageMeter, calculate_fnr
from network.models import get_models
from data.dataset import AIGCDetectionDataset, CLASS2LABEL_MAPPING, GenImage_LIST, FF_LIST
from utils.losses import LabelSmoothing, CombinedLoss, FocalLoss, ResidualConsistencyLoss
from data.transform import create_train_transforms, create_val_transforms
from data.transform import create_sdie_transforms, create_sdie_transforms_dup


def merge_tensor(img, label, is_train=True):
    """
    合并图像张量和标签张量，并在训练阶段对合并后的张量进行随机打乱。

    Args:
        img (list or torch.Tensor): 图像数据，可以是张量列表或单个张量。
        label (list or torch.Tensor): 标签数据，可以是张量列表或单个张量。
        is_train (bool, optional): 是否处于训练阶段，默认为 True。

    Returns:
        tuple: 合并后的图像张量和标签张量。
    """
    def shuffle_tensor(img, label):
        """
        对输入的图像张量和标签张量进行随机打乱。

        Args:
            img (torch.Tensor): 图像张量。
            label (torch.Tensor): 标签张量。

        Returns:
            tuple: 打乱后的图像张量和标签张量。
        """
        indices = torch.randperm(img.size(0))
        return img[indices], label[indices]
    if isinstance(img, list) and isinstance(label, list):
        img, label = torch.cat(img, dim=0), torch.cat(label, dim=0)
        if is_train:
            img, label = shuffle_tensor(img, label)
    return img, label

def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc

GenImage_LIST = ['stable_diffusion_v_1_4', 'stable_diffusion_v_1_5',
                 'Midjourney', 'ADM', 'wukong',
                 'glide', 'VQDM', 'BigGAN']

def eval_model(model, epoch, eval_loader, tokenizer, is_save=True, fake_indexes=None, threshold=0.5, alpha=0.5,
               save_txt=None):
    """
    评估模型在验证集或测试集上的性能。

    Args:
        model (torch.nn.Module): 待评估的模型。
        epoch (int): 当前的训练轮数。
        eval_loader (torch.utils.data.DataLoader): 验证集或测试集的数据加载器。
        is_save (bool, optional): 是否保存评估日志，默认为 True。
        threshold (float, optional): 二分类的阈值，默认为 0.5。
        alpha (float, optional): 对比损失的权重，默认为 0.5。
        save_txt (str, optional): 保存测试结果的文本文件路径，默认为 None。

    Returns:
        float or tuple: 如果 save_txt 不为 None，返回一个包含准确率、AUC、召回率、精确率、F1 分数和 FNR 的元组；
                        否则返回平均准确率。
    """
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    eval_process = tqdm(eval_loader)
    labels = []
    outputs = []
    all_features = []
    with torch.no_grad():
        for i, (img, type,label) in enumerate(eval_process):
            if i > 0 and i % 1 == 0:
                eval_process.set_description("Epoch: %s, Loss: %.4f, Acc: %.4f" %
                                             (epoch, losses.avg, accuracies.avg))

            batch_size = img[0].size(0)

            label_float = label.cuda().float()
            label_long = label.cuda().long()

            preds, _, features, input_score, recon_score = model(
                tokenizer, img, type, None, return_feature=True
            )

            loss_cls = criterion(preds, label_float)

            loss_con = contrastive_loss(features, label_long)

            loss = (1 - alpha) * loss_cls + alpha * loss_con
            probs = torch.sigmoid(preds)
            outputs.append(probs.cpu())

            labels.append(label)
            all_features.append(features.cpu())
            acc = accuracy_score(np.array(label.cpu()), np.array(probs.detach().cpu()) > 0.5)
            losses.update(loss.item(), batch_size)
            accuracies.update(acc, batch_size)

    outputs = torch.cat(outputs, dim=0).cpu().numpy()

    labels = torch.cat(labels, dim=0).cpu().numpy()
    all_features = np.concatenate(all_features, axis=0)

    idx_real = (labels == 0)
    idx_fake = (labels != 0)

    real_features = all_features[idx_real]
    fake_features = all_features[idx_fake]

    current_fake_name = GenImage_LIST[fake_indexes-1] if fake_indexes is not None else 'unknown_fake'

    save_feat_dir = "./tsne_features_nonorm_genimage"
    os.makedirs(save_feat_dir, exist_ok=True)

    np.save(os.path.join(save_feat_dir, f"Real_from_{current_fake_name}.npy"), real_features)
    np.save(os.path.join(save_feat_dir, f"Fake_{current_fake_name}.npy"), fake_features)

    print(f"[*] 已保存用于 t-SNE 的特征: Real ({len(real_features)}), {current_fake_name} ({len(fake_features)})")

    labels[labels > 0] = 1
    pre_dict = {
        'label': labels,
        'pred': outputs,
        'dataset': args.dataset_name,
        'features': all_features
    }
    auc = roc_auc_score(labels, outputs)
    recall = recall_score(labels, outputs > threshold)
    precision = precision_score(labels, outputs > threshold)
    binary_acc = accuracy_score(labels, outputs > threshold)
    f1 = f1_score(labels, outputs > threshold)
    fnr = calculate_fnr(labels, outputs > threshold)
    print(f'AUC:{auc}-Recall:{recall}-Precision:{precision}-BinaryAccuracy:{binary_acc}, f1: {f1}, fnr:{fnr}')
    if is_save:
        train_logger.log(phase="val", values={
            'epoch': epoch,
            'loss': format(losses.avg, '.4f'),
            'acc': format(accuracies.avg, '.4f'),
            'lr': optimizer.param_groups[0]['lr']
        })
    print("Val:\t Loss:{0:.4f} \t Acc:{1:.4f}".format(losses.avg, accuracies.avg))
    acc_avg = accuracies.avg
    del outputs, labels, losses, accuracies
    gc.collect()
    if save_txt is not None:
        return (auc,binary_acc, recall, precision, f1, fnr), pre_dict

    return acc_avg, pre_dict


def train_model(model, criterion, optimizer,tokenizer, epoch, scaler=None, alpha=0.5):
    """
    在一个训练轮次中训练模型。

    Args:
        model (torch.nn.Module): 待训练的模型。
        criterion (torch.nn.Module): 分类损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        epoch (int): 当前的训练轮数。
        scaler (torch.cuda.amp.GradScaler, optional): 混合精度训练的梯度缩放器，默认为 None。
        alpha (float, optional): 对比损失的权重，默认为 0.5。
    """
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    training_process = tqdm(train_loader)
    aux_criterion = nn.BCEWithLogitsLoss()
    for i, (input, types, label) in enumerate(training_process):
        model.total_steps += 1
        optimizer.zero_grad()
        current_lr = optimizer.param_groups[0]['lr']
        if i > 0 and i % 1 == 0:
            training_process.set_description(
                "Epoch: %d, LR: %.8f, Loss: %.4f, Acc: %.4f" % (
                    epoch, current_lr, losses.avg, accuracies.avg))

        input = [i.cuda() for i in input]
        label_float = label.cuda().float()
        label_long = label.cuda().long()
        recon_target = torch.ones_like(label_float)
        if scaler is None:
            print("scaler is None")
            batch_size = input[0].size(0)

            preds, _, features, input_score, recon_score = model(
                tokenizer, input, types, None, return_feature=True
            )

            loss_cls = criterion(preds, label_float)

            loss_con = contrastive_loss(features, label_long)

            loss_input = aux_criterion(input_score, label_float)

            loss_recon = aux_criterion(recon_score, recon_target)

            loss = (1 - alpha) * loss_cls + alpha * loss_con

            preds = torch.sigmoid(preds)
            acc = accuracy_score(np.array(label.cpu()), np.array(preds.detach().cpu()) > 0.5)

            losses.update(loss.item(), batch_size)
            accuracies.update(acc, batch_size)

            loss.backward()
            optimizer.step()
        else:
            print("scaler is not None")
            
            batch_size = input[0].size(0)
            with autocast():
                preds, _, features, input_score, recon_score = model(
                    tokenizer, input, types, None, return_feature=True
                )

                loss_cls = criterion(preds, label_float)

                loss_con = contrastive_loss(features, label_long)

                loss_input = aux_criterion(input_score, label_float)

                loss_recon = aux_criterion(recon_score, recon_target)

                loss = (1 - alpha) * loss_cls + alpha * loss_con +  loss_input
                print(f"{1 - alpha} * loss_cls + {alpha} * loss_con +  loss_input")
            preds = torch.sigmoid(preds)
            acc = accuracy_score(np.array(label.cpu()), np.array(preds.detach().cpu()) > 0.5)
            losses.update(loss.item(), batch_size)
            accuracies.update(acc, batch_size)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if model.total_steps in [400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2500,2700,3000,4000,5000,6000,8000,10000,12000,18000,20000,23000,25000]:
                torch.save(model.state_dict(), os.path.join(store_name, f'iter_{model.total_steps}.pth'))
        if args.is_warmup:
            with warmup_scheduler.dampening():
                scheduler.step()
    if not args.is_warmup:
        scheduler.step()
    train_logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg, '.4f'),
        'acc': format(accuracies.avg, '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    print("Train:\t Loss:{0:.4f} \t Acc:{1:.4f}".format(losses.avg, accuracies.avg))

    del losses, accuracies
    gc.collect()


if __name__ == '__main__':
    batch_size = args.batch_size * torch.cuda.device_count()

    writeFile = f"./RREM_gen/{args.dataset_name}/{args.fake_indexes.replace(',', '_')}/" \
                  f"{args.model_name.split('/')[-1]}_{args.input_size}{args.save_flag}/logs"
    print(f'使用create_side_transforms')
    store_name = writeFile.replace('/logs', '/weights')
    
    vis_dir = f"./RREM_gen/{args.dataset_name}/vis_results/{args.fake_indexes.replace(',', '_')}/"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    print(
        f'Using gpus:{args.device_id},batch size:{batch_size},gpu_count:{torch.cuda.device_count()},num_classes:{args.num_classes}')
    model = get_models(model_name=args.model_name, num_classes=args.num_classes,
                       embedding_size=args.embedding_size, freeze_extractor=args.freeze_extractor)

    with open("/home/law/HDD/Project_hjs/clip/ViT-L-14.pt", "rb") as f:
        checkpoint = torch.load(f, map_location="cpu", weights_only=False)
    start_epoch = 0
    model.load_state_dict(checkpoint.state_dict(), strict=False)

    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=not args.no_strict)
        print('Model found in {}'.format(args.model_path))
    else:
        print('No model found, initializing random model.')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
    criterion = nn.BCEWithLogitsLoss()
    contrastive_loss = CombinedLoss(loss_name=args.loss_name, embedding_size=args.embedding_size,
                                    pos_margin=args.pos_margin, neg_margin=args.neg_margin, tau=args.tau,
                                    memory_size=args.memory_size, use_miner=args.use_miner, num_classes=args.num_classes)
    print(f'Using {args.loss_name} as contrastive loss.')
    print(f'Using {args.memory_size} as memory size.')
    print(f'Using {args.use_miner} as use_miner.')
    print(f'Using {args.pos_margin} as pos_margin.')
    print(f'Using {args.neg_margin} as neg_margin.')
    print(f'Using {args.tau} as tau.')
    print(f'Using {args.neg_margin} as neg_margin.')
    print(f'Using {args.embedding_size} as embedding_size.')
    print(f'Using {args.num_classes} as num_classes.')
    print(f'Using {args.loss_name} as contrastive loss.')
    print(f'Using {args.alpha} as alpha.')
    print(f'Using {args.batch_size} as per-GPU batch size.')
    print(f'Using {args.num_workers} as workers per GPU.')
    print(f'Using {args.is_warmup} as is_warmup.')
    print(f'Using {args.is_amp} as is_amp.')
    print(f'Using {args.post_aug_mode} as post_aug_mode.')


    is_train = not args.is_test
    res_con_criterion = ResidualConsistencyLoss(margin=0.5).cuda()
    print(f'alpha: {args.alpha}')
    if is_train:
        print("训练阶段")
        if store_name and not os.path.exists(store_name):
            os.makedirs(store_name)
        train_logger = Logger(model_name=writeFile, header=['epoch', 'loss', 'acc', 'lr'])
        xdl = AIGCDetectionDataset(args.root_path, fake_root_path=args.fake_root_path, fake_indexes=args.fake_indexes, phase='train',
                                   num_classes=args.num_classes, inpainting_dir=args.inpainting_dir, is_dire=args.is_dire,
                                      transform=create_sdie_transforms(size=args.input_size, phase='train')
                                   )
        sampler = BalanceClassSampler(labels=xdl.get_labels(), mode=args.sampler_mode) if args.sampler_mode != '' else None
        train_loader = DataLoader(xdl, batch_size=batch_size, shuffle=sampler is None, num_workers=args.num_workers, sampler=sampler)
        train_dataset_len = len(xdl)

        xdl_eval = AIGCDetectionDataset(args.root_path, fake_root_path=args.fake_root_path, fake_indexes=args.fake_indexes, phase='val',
                                        num_classes=args.num_classes, inpainting_dir=args.inpainting_dir, is_dire=args.is_dire,
                                        transform=create_sdie_transforms(size=args.input_size, phase='val')
                                        )
        eval_loader = DataLoader(xdl_eval, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
        eval_dataset_len = len(xdl_eval)
        print('train_dataset_len:', train_dataset_len, 'eval_dataset_len:', eval_dataset_len)

        tokenizer = open_clip.get_tokenizer('ViT-L-14')

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=4e-5)
        if not args.is_warmup:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
        else:
            num_steps = train_dataset_len * args.num_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
            warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

        best_acc = 0.5 if args.epoch_start == 0 else eval_model(model, args.epoch_start - 1, eval_loader, is_save=False)
        for epoch in range(args.epoch_start, args.num_epochs):
            train_model(model, criterion, optimizer,tokenizer, epoch, scaler=GradScaler() if args.is_amp else None, alpha=args.alpha)
            if epoch % 1 == 0 or epoch == args.num_epochs - 1:
                acc, pre_dict = eval_model(model, epoch, eval_loader, tokenizer, alpha=args.alpha)
                save_path = '{}/{}_acc{:.4f}.pth'.format(store_name, epoch, acc)
                if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(), save_path)
                else:
                        torch.save(model.state_dict(), save_path)
                plot_score_distribution(pre_dict, vis_dir, args.fake_indexes,epoch)
                plot_feature_tsne(
                    features=pre_dict['features'],
                    labels=pre_dict['label'],
                    save_path=vis_dir,
                    dataset_name=f"{args.dataset_name}_fake{args.fake_indexes}",
                    epoch=epoch,
                    max_points=2000
                )
        last_save_path = '{}/last_acc{:.4f}.pth'.format(store_name, acc)
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), last_save_path)
        else:
            torch.save(model.state_dict(), last_save_path)
    else:
        print('测试阶段')
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
        start = time.time()
        epoch_start = 1
        num_epochs = 1

        if args.model_path is not None:
            ckpt_name = os.path.splitext(os.path.basename(args.model_path))[0]
        else:
            ckpt_name = "random_init"

        print(f"当前测试的模型权重为: {ckpt_name}")

        print('开始全模型特征空间可视化测试...')

        gen_model_indexes = [1, 2, 3, 4, 5, 6, 7, 8]
        total_features = []
        total_labels = []
        has_collected_real = False

        all_model_metrics = {
            'auc': [],
            'acc': [],
            'recall': [],
            'precision': [],
            'f1': [],
            'fnr': []
        }

        for model_idx in gen_model_indexes:
            print(f"\n>>> [Step {model_idx}/8] 正在处理模型索引: {model_idx} ...")

            xdl_test = AIGCDetectionDataset(
                args.root_path, 
                fake_root_path=args.fake_root_path, 
                fake_indexes=str(model_idx),
                phase='test', 
                num_classes=args.num_classes, 
                is_dire=args.is_dire,
                post_aug_mode=args.post_aug_mode,
                inpainting_dir=args.inpainting_dir,
                transform=create_sdie_transforms(size=args.input_size, phase='test')
            )

            test_loader = DataLoader(xdl_test, batch_size=batch_size, shuffle=False, num_workers=4)

            out_metrics, pre_dict = eval_model(
                model, 
                ckpt_name,
                test_loader, 
                tokenizer, 
                is_save=False, 
                fake_indexes=int(model_idx),
                threshold=args.threshold, 
                save_txt=args.save_txt,       
            )

            auc, acc, recall, precision, f1, fnr = out_metrics

            all_model_metrics['auc'].append(auc)
            all_model_metrics['acc'].append(acc)
            all_model_metrics['recall'].append(recall)
            all_model_metrics['precision'].append(precision)
            all_model_metrics['f1'].append(f1)
            all_model_metrics['fnr'].append(fnr)

            feats = pre_dict['features']
            lbls = pre_dict['label']
            np.save('./holist_256/test_features.npy', pre_dict['features'])
            np.save('./holist_256/test_labels.npy', pre_dict['label'])
            if not has_collected_real:
                real_indices = (lbls == 0)
                if real_indices.sum() > 0:
                    total_features.append(feats[real_indices])
                    total_labels.append(lbls[real_indices])
                    has_collected_real = True
                    print(f"  √ 已收集真图样本: {real_indices.sum()}")

            fake_indices = (lbls == 1)
            if fake_indices.sum() > 0:
                fake_feats = feats[fake_indices]
                fake_new_labels = np.full(fake_feats.shape[0], model_idx)

                total_features.append(fake_feats)
                total_labels.append(fake_new_labels)
                print(f"  √ 已收集假图样本 (Label={model_idx}): {len(fake_feats)}")
            plot_score_distribution(pre_dict, vis_dir, int(model_idx), ckpt_name)
            if args.save_txt is not None:
                os.makedirs(os.path.dirname(args.save_txt), exist_ok=True)
                auc ,acc, recall, precision, f1, fnr = out_metrics
                with open(args.save_txt, 'a') as file:
                    if args.dataset_name == 'GenImage':
                        class_name = GenImage_LIST[int(model_idx) - 1]
                    elif args.dataset_name == 'FF++':
                        class_name = FF_LIST[int(model_idx) - 1]
                    else:
                        class_name = list(CLASS2LABEL_MAPPING.keys())[int(model_idx)]
                    result_str = f'model_path:{args.model_path}, post_aug_mode:{args.post_aug_mode}, class_name:{class_name}\n' \
                                f'auc:{auc:.4f}, acc:{acc:.4f},  recall:{recall:.4f}, precision:{precision:.4f}, ' \
                                f'f1:{f1:.4f}, fnr: {fnr}\n'
                    file.write(result_str)
                print(f'The result was saved in {args.save_txt}')

        avg_metrics = {k: np.mean(v) for k, v in all_model_metrics.items()}

        print("\n" + "="*50)
        print(f"🔥 全模型平均指标 (Average Metrics over {len(gen_model_indexes)} models):")
        print(f"Avg AUC:       {avg_metrics['auc']:.4f}")
        print(f"Avg Accuracy:  {avg_metrics['acc']:.4f}")
        print(f"Avg Recall:    {avg_metrics['recall']:.4f}")
        print(f"Avg Precision: {avg_metrics['precision']:.4f}")
        print(f"Avg F1 Score:  {avg_metrics['f1']:.4f}")
        print(f"Avg FNR:       {avg_metrics['fnr']:.4f}")
        print("="*50 + "\n")

        if args.save_txt is not None:
            with open(args.save_txt, 'a') as file:
                avg_str = f"\n{'='*20} Average Metrics {'='*20}\n" \
                          f"Model Path: {args.model_path}\n" \
                          f"Avg AUC: {avg_metrics['auc']:.4f}\n" \
                          f"Avg Acc: {avg_metrics['acc']:.4f}\n" \
                          f"Avg Recall: {avg_metrics['recall']:.4f}\n" \
                          f"Avg Precision: {avg_metrics['precision']:.4f}\n" \
                          f"Avg F1: {avg_metrics['f1']:.4f}\n" \
                          f"Avg FNR: {avg_metrics['fnr']:.4f}\n" \
                          f"{'='*56}\n\n"
                file.write(avg_str)
                print(f"✅ 平均指标已追加保存至: {args.save_txt}")

        if len(total_features) > 0:
            final_features = np.concatenate(total_features, axis=0)
            final_labels = np.concatenate(total_labels, axis=0)

            print(f"\n数据收集完毕! 总样本: {len(final_labels)}")

            plot_feature_tsne_multi(
                features=final_features,
                labels=final_labels,
                save_path=vis_dir,
                dataset_name="GenImage_Ordered_8_Models",
                epoch=ckpt_name,
                max_points=2000
            )
        print('Total time:', time.time() - start)
