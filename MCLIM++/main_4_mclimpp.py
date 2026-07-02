import datetime
import math
import sys
import time
import warnings
from functools import partial
from typing import List

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

import dist
from models import build_sparse_encoder, build_decoder
from sampler import DistInfiniteBatchSampler, worker_init_fn
from mclimpp import MCLIM
from utils import arg_util, misc, lamb
from utils.mri_4_mclimpp import build_dataset_to_pretrain
from utils.lr_control import lr_wd_annealing, get_param_groups

import SimpleITK as sitk


class LocalDDP(torch.nn.Module):
    def __init__(self, module):
        super(LocalDDP, self).__init__()
        self.module = module
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def main_pt():
    warnings.filterwarnings("ignore") 
    
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    print(f'initial args:\n{str(args)}')
    args.log_epoch()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # build data
    print(f'[build data for pre-training] ...\n')
    dataset_train = build_dataset_to_pretrain(args.adni_data_path, args.multi_data_path, args.input_size, args.mim_ratio, args.patch_size, args, False)
    data_loader_train = DataLoader(
        dataset=dataset_train, num_workers=args.dataloader_workers, pin_memory=True,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size,
            shuffle=True, filling=True, rank=dist.get_rank(), world_size=dist.get_world_size(),
        ), worker_init_fn=worker_init_fn, persistent_workers=True
    )
    itrt_train, iters_train = iter(data_loader_train), len(data_loader_train)
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size_per_gpu}, iters_train={iters_train}')
    
    model_without_ddp = MCLIM(rank=dist.get_rank(), world_size=dist.get_world_size(), local_batch_size=4, input_size=args.input_size, backbone=args.model, feature_size=args.feature_size).to(args.device)
    print(f'[PT model] model = {model_without_ddp}\n')
    if dist.initialized():
        model: DistributedDataParallel = DistributedDataParallel(model_without_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=True, broadcast_buffers=False)
    else:
        model = LocalDDP(model_without_ddp)
    
    # build optimizer and lr_scheduler
    param_groups: List[dict] = get_param_groups(model_without_ddp)
    opt_clz = {
        'sgd': partial(torch.optim.SGD, momentum=0.9, nesterov=True),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, args.ada)),
        'lamb': partial(lamb.TheSameAsTimmLAMB, betas=(0.9, args.ada), max_grad_norm=5.0),
    }[args.opt]
    optimizer = opt_clz(params=param_groups, lr=args.lr, weight_decay=0.0)
    print(f'[optimizer] optimizer({opt_clz}) ={optimizer}\n')
    
    # try to resume
    ep_start, performance_desc = misc.load_checkpoint(args.resume_from, model_without_ddp, optimizer)
    if ep_start >= args.ep: # load from a complete checkpoint file
        print(f'  [*] [PT already done]    Min/Last Loss: {performance_desc}')
    else:   # perform pre-training
        tb_lg = misc.TensorboardLogger(args.tb_lg_dir, is_master=dist.is_master(), prefix='pt')
        min_loss = 1e9
        print(f'[PT start] from ep{ep_start}')

        if args.amp:
            scaler = GradScaler()
        else:
            scaler = None
        
        pt_start_time = time.time()
        for ep in range(ep_start, args.ep):
            ep_start_time = time.time()
            tb_lg.set_step(ep * iters_train)
            if hasattr(itrt_train, 'set_epoch'):
                itrt_train.set_epoch(ep)
            
            stats = pre_train_one_ep(ep, args, tb_lg, itrt_train, iters_train, model, optimizer, scaler)
            last_loss = stats['last_loss']
            clip_loss = stats['clip_loss']
            match_loss = stats['match_loss']
            recon_im_loss = 0
            hire_loss = 0
            deform_loss = 0

            if args.hire:
                hire_loss = stats['hire_loss']
            if args.deform:
                deform_loss = stats['deform_loss']
            if args.weight_recon>0:
                recon_im_loss = stats['recon_im_loss']

            min_loss = min(min_loss, last_loss)
            performance_desc = f'{min_loss:.4f} {last_loss:.4f} {clip_loss:.4f} {match_loss:.4f} {recon_im_loss:.4f} {deform_loss:.4f} {hire_loss:.4f}'

            if args.hire:
                performance_desc += f' {stats["loss_l5"]:.4f} {stats["loss_l4"]:.4f} {stats["loss_l3"]:.4f} {stats["loss_l2"]:.4f} {stats["loss_l1"]:.4f}'

            misc.save_checkpoint(f'{args.model}_still_pretraining.pth', args, ep, performance_desc, model_without_ddp.state_dict(), optimizer.state_dict())
            if ep % 20 == 0 and ep != 0:
                misc.save_checkpoint(f'{args.model}_{ep}.pth', args, ep, performance_desc, model_without_ddp.state_dict(), optimizer.state_dict())
            
            ep_cost = round(time.time() - ep_start_time, 2) + 1    # +1s: approximate the following logging cost
            remain_secs = (args.ep-1 - ep) * ep_cost
            remain_time = datetime.timedelta(seconds=round(remain_secs))
            finish_time = time.strftime("%m-%d %H:%M", time.localtime(time.time() + remain_secs))
            print(f'  [*] [ep{ep}/{args.ep}]    Min/Last Loss {performance_desc},    Cost: {ep_cost}s,    Remain: {remain_time},    Finish @ {finish_time}')
            
            args.cur_ep = f'{ep + 1}/{args.ep}'
            args.remain_time, args.finish_time = str(remain_time), str(finish_time)
            args.last_loss = last_loss
            args.log_epoch()
            
            tb_lg.update(min_loss=min_loss, head='train', step=ep)
            tb_lg.update(rest_hours=round(remain_secs/60/60, 2), head='z_burnout', step=ep)
            tb_lg.flush()
        
        # finish pre-training
        tb_lg.update(min_loss=min_loss, head='result', step=ep_start)
        tb_lg.update(min_loss=min_loss, head='result', step=args.ep)
        tb_lg.flush()
        print(f'final args:\n{str(args)}')
        print('\n\n')
        print(f'  [*] [PT finished]    Min/Last Loss: {performance_desc},    Total Cost: {(time.time() - pt_start_time) / 60 / 60:.1f}h\n')
        print('\n\n')
        tb_lg.close()
        time.sleep(10)
    
    args.remain_time, args.finish_time = '-', time.strftime("%m-%d %H:%M", time.localtime(time.time()))
    args.log_epoch()


def pre_train_one_ep(ep, args: arg_util.Args, tb_lg: misc.TensorboardLogger, itrt_train, iters_train, model: DistributedDataParallel, optimizer, scaler):
    model.train()
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('max_lr', misc.SmoothedValue(window_size=1, fmt='{value:.5f}'))
    header = f'[PT] Epoch {ep}:'
    
    warnings.filterwarnings("ignore") 
    optimizer.zero_grad()
    early_clipping = args.clip > 0 and not hasattr(optimizer, 'global_grad_norm')
    print('Early Clipping:', early_clipping)
    late_clipping = hasattr(optimizer, 'global_grad_norm')
    if early_clipping:
        params_req_grad = [p for p in model.parameters() if p.requires_grad]
    
    for it, (inp) in enumerate(me.log_every(iters_train, itrt_train, 100, header)):
        # adjust lr and wd
        min_lr, max_lr, min_wd, max_wd = lr_wd_annealing(optimizer, args.lr, args.wd, args.wde, it + ep * iters_train, args.wp_ep * iters_train, args.ep * iters_train)
        
        # forward and backward
        inp_t = torch.cat([t['image'] for t in inp], dim=0).to(args.device, non_blocking=True)
        modality_t = torch.cat([t['modality'] for t in inp], dim=0).long().view(-1).to(args.device, non_blocking=True)
        label_t = torch.cat([t['label'] for t in inp], dim=0).to(args.device, non_blocking=True)
        patch_ratio_t = torch.cat([t['patch_ratio'] for t in inp], dim=0).to(args.device, non_blocking=True)
        coverage_ratio_t = torch.cat([t['coverage_ratio'] for t in inp], dim=0).to(args.device, non_blocking=True)
        ignore_mask_t = torch.cat([t['ignore_mask'] for t in inp], dim=0).to(args.device, non_blocking=True)

        label_tumor_class_t = None
        label_tumor_onehot_t = None
        jacobian_t = None
        labels_l4_t = None
        labels_l3_t = None
        labels_l2_t = None
        labels_l1_t = None

        if args.tumor_aware:
            label_tumor_class_t = torch.cat([t['label_tumor_class'] for t in inp], dim=0).to(args.device, non_blocking=True)
            label_tumor_onehot_t = torch.cat([t['label_tumor_onehot'] for t in inp], dim=0).to(args.device, non_blocking=True)

        if args.mim_ratio>0:
            mask_t = torch.cat([t['mask'] for t in inp], dim=0).to(args.device, non_blocking=True)
            mask_image_t = torch.cat([t['mask_image'] for t in inp], dim=0).to(args.device, non_blocking=True)
        else:
            mask_t = None
            mask_image_t = None

        if args.deform:
            jacobian_t = torch.cat([t['jacobian'] for t in inp], dim=0).to(args.device, non_blocking=True)

        if args.hire:
            labels_l4_t = torch.cat([t['label_l4'] for t in inp], dim=0).to(args.device, non_blocking=True)
            labels_l3_t = torch.cat([t['label_l3'] for t in inp], dim=0).to(args.device, non_blocking=True)
            labels_l2_t = torch.cat([t['label_l2'] for t in inp], dim=0).to(args.device, non_blocking=True)
            labels_l1_t = torch.cat([t['label_l1'] for t in inp], dim=0).to(args.device, non_blocking=True)

        with autocast(enabled=scaler is not None, dtype=torch.bfloat16):

            if (args.hire and args.deform) or (args.hire):
                loss, clip_loss, match_loss, recon_im_loss, deform_loss, hire_loss, loss_l5, loss_l4, loss_l3, loss_l2, loss_l1 = model(image=inp_t, modality=modality_t, mask=mask_t, mask_image=mask_image_t, label=label_t, ignore_mask=ignore_mask_t,
                patch_ratio=patch_ratio_t, coverage_ratio=coverage_ratio_t, weight_recon=args.weight_recon, label_tumor_class=label_tumor_class_t, label_tumor_onehot=label_tumor_onehot_t,
                jacobian=jacobian_t, labels_l4_all=labels_l4_t, labels_l3_all=labels_l3_t, labels_l2_all=labels_l2_t, labels_l1_all=labels_l1_t)
                grad_norm = None
            elif args.deform:
                loss, clip_loss, match_loss, recon_im_loss, deform_loss, _ = model(image=inp_t, modality=modality_t, mask=mask_t, mask_image=mask_image_t, label=label_t, ignore_mask=ignore_mask_t,
                patch_ratio=patch_ratio_t, coverage_ratio=coverage_ratio_t, weight_recon=args.weight_recon, label_tumor_class=label_tumor_class_t, label_tumor_onehot=label_tumor_onehot_t,
                jacobian=jacobian_t, labels_l4_all=labels_l4_t, labels_l3_all=labels_l3_t, labels_l2_all=labels_l2_t, labels_l1_all=labels_l1_t)
                grad_norm = None
            else:
                loss, clip_loss, match_loss, recon_im_loss, _, _ = model(image=inp_t, modality=modality_t, mask=mask_t, mask_image=mask_image_t, label=label_t, ignore_mask=ignore_mask_t,
                patch_ratio=patch_ratio_t, coverage_ratio=coverage_ratio_t, weight_recon=args.weight_recon, label_tumor_class=label_tumor_class_t, label_tumor_onehot=label_tumor_onehot_t,
                jacobian=jacobian_t, labels_l4_all=labels_l4_t, labels_l3_all=labels_l3_t, labels_l2_all=labels_l2_t, labels_l1_all=labels_l1_t)
                grad_norm = None

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if early_clipping: grad_norm = torch.nn.utils.clip_grad_norm_(params_req_grad, args.clip).item()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if early_clipping: grad_norm = torch.nn.utils.clip_grad_norm_(params_req_grad, args.clip).item()
                optimizer.step()
                if late_clipping: grad_norm = optimizer.global_grad_norm

            loss = loss.item()
            clip_loss = clip_loss.item()
            match_loss = match_loss.item()

            if args.mim_ratio>0:
                recon_im_loss = recon_im_loss.item()
            if args.hire:
                hire_loss = hire_loss.item()
                loss_l5 = loss_l5.item()
                loss_l4 = loss_l4.item() 
                loss_l3 = loss_l3.item()
                loss_l2 = loss_l2.item()
                loss_l1 = loss_l1.item()
            if args.deform:
                deform_loss = deform_loss.item()

            optimizer.zero_grad()
            
            torch.cuda.synchronize()
        
        # log
        me.update(last_loss=loss)
        me.update(clip_loss=clip_loss)
        me.update(match_loss=match_loss)

        if args.deform:
            me.update(deform_loss=deform_loss)
            tb_lg.update(loss=me.meters['deform_loss'].global_avg, head='deform_loss')
        if args.hire:
            me.update(hire_loss=hire_loss)
            tb_lg.update(loss=me.meters['hire_loss'].global_avg, head='hire_loss')
            me.update(loss_l5=loss_l5)
            me.update(loss_l4=loss_l4)
            me.update(loss_l3=loss_l3)
            me.update(loss_l2=loss_l2)
            me.update(loss_l1=loss_l1)
            tb_lg.update(loss=me.meters['loss_l5'].global_avg, head='loss_l5')
            tb_lg.update(loss=me.meters['loss_l4'].global_avg, head='loss_l4')
            tb_lg.update(loss=me.meters['loss_l3'].global_avg, head='loss_l3')
            tb_lg.update(loss=me.meters['loss_l2'].global_avg, head='loss_l2')
            tb_lg.update(loss=me.meters['loss_l1'].global_avg, head='loss_l1')
        if args.mim_ratio>0:
            me.update(recon_im_loss=recon_im_loss)
            tb_lg.update(loss=me.meters['recon_im_loss'].global_avg, head='recon_im_loss')

        me.update(max_lr=max_lr)
        tb_lg.update(loss=me.meters['last_loss'].global_avg, head='train_loss')
        tb_lg.update(loss=me.meters['clip_loss'].global_avg, head='clip_loss')
        tb_lg.update(loss=me.meters['match_loss'].global_avg, head='match_loss')
        
        tb_lg.update(sche_lr=max_lr, head='train_hp/lr_max')
        tb_lg.update(sche_lr=min_lr, head='train_hp/lr_min')
        tb_lg.update(sche_wd=max_wd, head='train_hp/wd_max')
        tb_lg.update(sche_wd=min_wd, head='train_hp/wd_min')
        
        if grad_norm is not None:
            me.update(orig_norm=grad_norm)
            tb_lg.update(orig_norm=grad_norm, head='train_hp')
        tb_lg.set_step()
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}


if __name__ == '__main__':
    main_pt()
