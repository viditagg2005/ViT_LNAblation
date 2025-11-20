# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from gradient_flow_diagnostics import run_stability_diagnostics
import time
import utils

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    # [ADD THIS BLOCK] --- Setup Axis 1 Hooks ---
    for m in model.modules():
        if "DynamicTanh" in str(type(m)) and hasattr(m, 'register_stability_hooks'):
            m.register_stability_hooks()
    # -------------------------------------------
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    # [ADD THIS LINE] --- Axis 2: Convergence Time ---
    start_time = time.time()

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
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

        # --- [FIX] Comprehensive Axis 1 Logging ---
        if data_iter_step % print_freq == 0:
            # Initialize lists to hold values from all DyT layers
            ratios = []
            alphas = []
            grad_in = []
            grad_out = []

            for m in model.modules():
                if "DynamicTanh" in str(type(m)):
                    # 1. Capture Alpha (The learnable parameter)
                    if hasattr(m, 'alpha'):
                        alphas.append(m.alpha.item())
                    
                    # 2. Capture Gradient Stats
                    if hasattr(m, 'stability_metrics'):
                         r = m.stability_metrics.get('grad_norm_ratio', None)
                         g_in = m.stability_metrics.get('grad_in_norm', None)
                         g_out = m.stability_metrics.get('grad_out_norm', None)
                         
                         if r is not None: ratios.append(r)
                         if g_in is not None: grad_in.append(g_in)
                         if g_out is not None: grad_out.append(g_out)
            
            # Compute Averages
            update_dict = {}
            if ratios:
                update_dict['grad_ratio'] = sum(ratios) / len(ratios)
            if alphas:
                update_dict['alpha'] = sum(alphas) / len(alphas)
            if grad_in:
                update_dict['g_in'] = sum(grad_in) / len(grad_in)
            if grad_out:
                update_dict['g_out'] = sum(grad_out) / len(grad_out)

            # Update the Loggers
            if update_dict:
                # Print to Terminal
                metric_logger.update(**update_dict)
                
                # Log to TensorBoard/WandB
                if log_writer is not None:
                    for k, v in update_dict.items():
                        log_writer.update({k: v}, head="perf")
        # -----------------------------------------------------


        # --- [FIX] Axis 1: Run Diagnostics (Rarely) ---
        if data_iter_step > 0 and data_iter_step % 500 == 0:
             print("\nRunning Stability Diagnostics... (this takes a moment)")
             diag_input = samples[:4].to(device)
             diag_stats = run_stability_diagnostics(model, diag_input)
             
             # Print specifically to console so you see it immediately
             print(f"Diagnostic Stats (Step {data_iter_step}): {diag_stats}")

             # Log to writers
             if log_writer is not None:
                 for k, v in diag_stats.items():
                     log_writer.update({k: v}, head="diagnostics")
             
             model.train()
        # -----------------------------------------------------

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
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

        # [ADD THIS BLOCK] --- Axis 1: Run Expensive Diagnostics (Rarely) ---
        # Run every 500 steps (adjust based on speed)
        if data_iter_step > 0 and data_iter_step % 500 == 0:
             # Use a small slice of samples to save time
             diag_input = samples[:4].to(device)
             diag_stats = run_stability_diagnostics(model, diag_input)
             
             if log_writer is not None:
                 for k, v in diag_stats.items():
                     log_writer.update(**{k: v}, head="diagnostics")
             
             model.train() # Ensure we return to train mode
        # -------------------------------------------------------------------

        torch.cuda.synchronize()

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

    # [ADD THIS LINE] --- Axis 2: Log Total Epoch Time ---
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