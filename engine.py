# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
import os
import json
import time
import argparse
from typing import Iterable, Optional

import torch
import torch.nn as nn
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import utils

# --- [NEW] Generic Hook for Gradient Monitoring ---
def _generic_monitoring_hook(module, grad_input, grad_output):
    """
    Generic backward hook to capture gradient norms for ANY layer (LN or DyT).
    Calculates In-Norm, Out-Norm, and the Ratio (Stability Metric).
    """
    if grad_input[0] is not None and grad_output[0] is not None:
        with torch.no_grad():
            g_in = torch.norm(grad_input[0]).item()
            g_out = torch.norm(grad_output[0]).item()
            ratio = g_in / (g_out + 1e-9)
            
            # Create a temporary storage dict on the module
            if not hasattr(module, 'monitor_stats'): module.monitor_stats = {}
            module.monitor_stats['grad_in'] = g_in
            module.monitor_stats['grad_out'] = g_out
            module.monitor_stats['grad_ratio'] = ratio

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False, args=None):
    
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    # -------------------------------------------------------------------------
    # [NEW] Setup History File (JSONL format)
    # -------------------------------------------------------------------------
    history_path = os.path.join(args.output_dir, "training_log.json")
    
    # If this is the very first epoch, delete old log to avoid appending to previous runs
    if epoch == 0 and os.path.exists(history_path): 
        try:
            os.remove(history_path)
        except OSError:
            pass # Handle race conditions

    # -------------------------------------------------------------------------
    # [NEW] Attach Hooks (Layer-Agnostic)
    # -------------------------------------------------------------------------
    print(f"Attaching gradient monitors for Epoch {epoch}...")
    for name, m in model.named_modules():
        is_ln = isinstance(m, nn.LayerNorm)
        is_dyt = "DynamicTanh" in str(type(m))
        
        if is_ln or is_dyt:
            # Ensure storage exists
            if not hasattr(m, 'monitor_stats'): m.monitor_stats = {}
            m.monitor_stats['name'] = name 
            # Register hook
            m.register_full_backward_hook(_generic_monitoring_hook)
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50 # Log every 50 steps to keep JSON size manageable
    start_time = time.time()

    optimizer.zero_grad()
    
    # Buffer to store logs in memory before flushing to disk at end of epoch
    epoch_logs = [] 

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        
        # Update LR & WD
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if use_amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        # ---------------------------------------------------------------------
        # [NEW] Logging Logic (Fast & Comprehensive)
        # ---------------------------------------------------------------------
        if data_iter_step % print_freq == 0:
            step_data = {"epoch": epoch, "step": data_iter_step, "modules": {}}
            
            # Harvest data from the hooks attached to modules
            for name, m in model.named_modules():
                # Only log if the hook has fired (monitor_stats has 'grad_ratio')
                if hasattr(m, 'monitor_stats') and 'grad_ratio' in m.monitor_stats:
                    mod_data = {
                        "grad_in": m.monitor_stats['grad_in'],
                        "grad_out": m.monitor_stats['grad_out'],
                        "grad_ratio": m.monitor_stats['grad_ratio'],
                    }
                    
                    # Also capture Alpha if it exists (DyT specific metric)
                    if hasattr(m, 'alpha'): 
                        mod_data['alpha'] = m.alpha.item()
                    
                    step_data["modules"][name] = mod_data
            
            # 1. Store detailed data in buffer (for JSON file)
            if step_data["modules"]:
                epoch_logs.append(step_data)
            
            # 2. Update Console Logger (Average Ratio only)
            ratios = [d['grad_ratio'] for d in step_data['modules'].values() if 'grad_ratio' in d]
            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                metric_logger.update(grad_ratio=avg_ratio)
            
            # 3. Update Elapsed Time
            metric_logger.update(time_elapsed=(time.time() - start_time)/3600.0)

        # ---------------------------------------------------------------------

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # -------------------------------------------------------------------------
    # [NEW] Flush Buffer to JSON at End of Epoch
    # -------------------------------------------------------------------------
    if utils.is_main_process():
        with open(history_path, "a") as f:
            for entry in epoch_logs:
                f.write(json.dumps(entry) + "\n")

    total_time = time.time() - start_time
    print(f"Epoch {epoch} time: {total_time:.2f}s")
    
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}