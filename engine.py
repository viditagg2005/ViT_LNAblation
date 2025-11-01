# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import time
from utils import save_json, save_tensor_dict, estimate_module_params_and_flops
import utils
import os

#######################
# Define a small function to walk the model and snapshot DynamicTanh modules
def snapshot_all_alphas(model):
    snapshots = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ == "DynamicTanh":
            try:
                snapshots[name] = module.snapshot_alpha()
            except Exception:
                # fallback: try reading attributes alpha/u/v
                d = {}
                if hasattr(module, "alpha"):
                    d["alpha"] = module.alpha.detach().cpu().tolist()
                if hasattr(module, "u") and hasattr(module, "v"):
                    d["u_mean"] = float(module.u.detach().cpu().mean())
                    d["v_mean"] = float(module.v.detach().cpu().mean())
                snapshots[name] = d
    return snapshots

# Register gradient monitors for all DyT modules (call at start of training)
def attach_all_grad_monitors(model, grad_container):
    for name, module in model.named_modules():
        if module.__class__.__name__ == "DynamicTanh":
            module.attach_gradient_monitor(name, grad_container)

def remove_all_grad_monitors(model, grad_container):
    for name, module in model.named_modules():
        if module.__class__.__name__ == "DynamicTanh":
            module.remove_gradient_monitor(name, grad_container)
#######################

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    model.train(True)
    grad_container = {} ## newly added
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    attach_all_grad_monitors(model, grad_container)
    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        start = time.time() ## newly added to monitor time

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
        ##########################################################################
        # after step: log grad_container if you like (pop latest entries)
        if step % log_interval == 0:
            # gather summary
            grad_summary = {}
            for k, v in grad_container.items():
                # for each module, we recorded lists of grads; read last entry
                input_grad_norm = v["input_grad_norms"][-1] if len(v.get("input_grad_norms", []))>0 else None
                output_grad_norm = v["output_grad_norms"][-1] if len(v.get("output_grad_norms", []))>0 else None
                grad_summary[k] = {"in_grad": input_grad_norm, "out_grad": output_grad_norm}
            # log grad_summary to TB or JSON
            logger.log({f"grad_stats/step_{step}": grad_summary})

        # optional: periodically snapshot alpha parameters (every N steps)
        if step % alpha_snapshot_interval == 0:
            alpha_snap = snapshot_all_alphas(model)
            save_json(alpha_snap, os.path.join(output_dir, f"alpha_snapshot_step_{global_step}.json"))

        # optional: estimate DyT param/flops overhead once per epoch
        if step == 0 and hasattr(model, 'named_modules'):
            dyt_costs = {}
            for name, m in model.named_modules():
                if m.__class__.__name__ == "DynamicTanh":
                    info = estimate_module_params_and_flops(m, sample_shape=(batch_size, images.shape[1], images.shape[-1] if images.ndim==3 else m.hidden_dim if hasattr(m,'hidden_dim') else 1))
                    dyt_costs[name] = info
            save_json(dyt_costs, os.path.join(output_dir, f"dyt_costs_epoch_{epoch}.json"))

    # remove monitors at end of epoch to avoid unexpected hooks surviving
    remove_all_grad_monitors(model, grad_container)
    ##########################################################################     

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
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
