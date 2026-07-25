"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import contextlib
import logging
import math
import os
import time

import torch
import torch.distributed as dist
from model.lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from model.lavis.common.logger import MetricLogger, SmoothedValue
from model.lavis.common.registry import registry
from model.lavis.datasets.data_utils import prepare_sample
from torch.nn.utils import clip_grad_norm_

# Opt-in per-rank DDP tracing. Off by default (production is unaffected); enable
# with META_DDP_DEBUG=1 to pinpoint exactly which collective a resumed 2-GPU run
# stalls on. Every marker flushes and (with sync=True) drives a cuda.synchronize
# first, so a marker that fails to print localises the hang to the op just after
# the last printed marker on that rank.
_DDP_DEBUG = os.environ.get("META_DDP_DEBUG", "0").strip().lower() not in ("", "0", "false", "no")


def _ddp_marker(tag, *, epoch=None, iter_idx=None, loss=None, batch_size=None,
                extra="", sync=False):
    if not _DDP_DEBUG:
        return
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    rank = get_rank()
    local_rank = os.environ.get("LOCAL_RANK", "?")
    dev = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    loss_str = ""
    if loss is not None:
        lv = loss.item() if torch.is_tensor(loss) else float(loss)
        loss_str = f" loss={lv:.4f} finite={math.isfinite(lv)}"
    bs_str = f" bs={batch_size}" if batch_size is not None else ""
    extra_str = f" {extra}" if extra else ""
    print(
        f"[DDP_DBG rank{rank} lrank{local_rank} cuda:{dev}] {tag} "
        f"epoch={epoch} iter={iter_idx}{loss_str}{bs_str}{extra_str}",
        flush=True,
    )


def _batch_gate_summary(samples):
    """Compact per-rank view of the branch gates that historically diverged."""
    if not _DDP_DEBUG:
        return ""
    parts = []
    img = samples.get("image")
    if torch.is_tensor(img) and img.ndim > 0:
        parts.append(f"bs={img.shape[0]}")
    aux_mask = samples.get("aux_mask")
    if torch.is_tensor(aux_mask):
        parts.append(f"aux_any={bool(aux_mask.any().item()) if aux_mask.numel() else False}")
    cls_mask = samples.get("classification_mask")
    gen_mask = samples.get("generation_mask")
    if torch.is_tensor(cls_mask) and torch.is_tensor(gen_mask):
        parts.append(
            f"teacher_any={bool((cls_mask.bool() & gen_mask.bool()).any().item())}"
        )
    return " ".join(parts)


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        output = model(samples)
        loss_dict = {}
        for k,v in output.items():
            if "loss" in k:
                loss_dict[k] = v
        return output["loss"], loss_dict

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
        max_grad_norm=1.0,
        run_role="train",
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=len(data_loader),
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
            max_grad_norm=max_grad_norm,
            run_role=run_role,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
        max_grad_norm=1.0,
        run_role="train",
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
            max_grad_norm=max_grad_norm,
            run_role=run_role,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
        max_grad_norm=1.0,
        run_role="train",
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None
        epoch_started = time.perf_counter()
        processed_examples = 0
        if cuda_enabled:
            torch.cuda.reset_peak_memory_stats()

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        # Role prefix ("[preflight]" vs "[train]") so the 2-step preflight probe,
        # which always runs a fresh epoch 0, is never mistaken in the logs for
        # the real run restarting at epoch 0 after a resume.
        header = "[{}] Train: data epoch: [{}]".format(run_role, epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        if accum_grad_iters < 1:
            raise ValueError("accum_grad_iters must be >= 1")
        updates_per_epoch = math.ceil(iters_per_epoch / accum_grad_iters)
        optimizer.zero_grad(set_to_none=True)
        window_had_nonfinite = False
        current_lr = optimizer.param_groups[0]["lr"]
        loss_dict = {}

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            _ddp_marker("BEFORE_FETCH_BATCH", epoch=inner_epoch, iter_idx=i)
            samples = next(data_loader)
            batch_probe = samples.get("classification_labels", samples.get("image"))
            local_bs = None
            if torch.is_tensor(batch_probe) and batch_probe.ndim > 0:
                local_bs = int(batch_probe.shape[0])
                processed_examples += local_bs
            _ddp_marker("AFTER_FETCH_BATCH", epoch=inner_epoch, iter_idx=i,
                        batch_size=local_bs)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            window_start = (i // accum_grad_iters) * accum_grad_iters
            window_size = min(accum_grad_iters, iters_per_epoch - window_start)
            is_sync_step = i + 1 == window_start + window_size
            if i == window_start:
                try:
                    lr_scheduler.step(
                        cur_epoch=inner_epoch,
                        cur_step=i // accum_grad_iters,
                        steps_per_epoch=updates_per_epoch,
                    )
                except TypeError:
                    # Backward compatibility for third-party schedulers using
                    # the original two-argument LAVIS interface.
                    lr_scheduler.step(
                        cur_epoch=inner_epoch,
                        cur_step=i // accum_grad_iters,
                    )

            sync_ctx = (
                contextlib.nullcontext()
                if is_sync_step or not hasattr(model, "no_sync")
                else model.no_sync()
            )
            with sync_ctx:
                # DDP's no_sync must encompass *both* forward and backward;
                # entering it only for backward still schedules the all-reduce
                # from DDP's forward hooks.
                if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                    amp_context = torch.amp.autocast(
                        device_type="cuda", enabled=use_amp
                    )
                elif use_amp:
                    amp_context = torch.cuda.amp.autocast()
                else:
                    amp_context = contextlib.nullcontext()
                with amp_context:
                    _ddp_marker("BEFORE_FORWARD", epoch=inner_epoch, iter_idx=i,
                                extra=_batch_gate_summary(samples), sync=True)
                    loss, loss_dict = self.train_step(model=model, samples=samples)
                    _ddp_marker("AFTER_FORWARD", epoch=inner_epoch, iter_idx=i,
                                loss=loss, sync=True)
                    # The final accumulation window is often shorter on the full
                    # dataset. Divide by its actual size so its update has the same
                    # scale as every complete window.
                    scaled_loss = loss / window_size

                # Skip iter if loss is NaN/Inf so corruption can't propagate through
                # accumulated grads; AMP overflow handling alone doesn't clear .grad.
                # Finiteness is a per-rank observation, but backward() drives a
                # collective, so every rank makes the same decision first.
                skip_iter = torch.tensor(
                    float(not torch.isfinite(loss)), device=loss.device
                )
                if is_dist_avail_and_initialized():
                    dist.all_reduce(skip_iter, op=dist.ReduceOp.MAX)
                if skip_iter.item() > 0:
                    logging.warning(
                        f"Non-finite loss at iter {i} on at least one rank "
                        f"(local value={loss.item()}); skipping step on all ranks."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    window_had_nonfinite = True
                    if is_sync_step:
                        window_had_nonfinite = False
                    continue

                if window_had_nonfinite:
                    # Discard the complete accumulation window on every rank. A
                    # partial update after one non-finite micro-batch is biased and
                    # has the wrong effective scale.
                    if is_sync_step:
                        optimizer.zero_grad(set_to_none=True)
                        window_had_nonfinite = False
                    continue

                _ddp_marker("BEFORE_BACKWARD", epoch=inner_epoch, iter_idx=i,
                            loss=loss, sync=True)
                if use_amp:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                _ddp_marker("AFTER_BACKWARD", epoch=inner_epoch, iter_idx=i, sync=True)

            # update gradients every accum_grad_iters iterations
            if is_sync_step:
                _ddp_marker("BEFORE_OPTIMIZER_STEP", epoch=inner_epoch, iter_idx=i,
                            sync=True)
                if use_amp:
                    # First unscale the gradients before clipping
                    scaler.unscale_(optimizer)
                    if max_grad_norm > 0:
                        clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Directly clip the gradients for non-AMP training
                    if max_grad_norm > 0:
                        clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                # Clear grads after each optimizer step; without this, residual
                # inf/NaN from an AMP overflow persist across iters and corrupt
                # all subsequent updates.
                optimizer.zero_grad(set_to_none=True)
                _ddp_marker("AFTER_OPTIMIZER_STEP", epoch=inner_epoch, iter_idx=i,
                            sync=True)

            current_lr = optimizer.param_groups[0]["lr"]
            # print(f"current lr is {current_lr}")
            metric_logger.update(**loss_dict)
            metric_logger.update(lr=current_lr)

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        stats = {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }
        stats["optimizer_steps"] = updates_per_epoch
        elapsed = time.perf_counter() - epoch_started
        peak_vram = torch.cuda.max_memory_allocated() if cuda_enabled else 0
        local_runtime = torch.tensor(
            [float(processed_examples), elapsed, float(peak_vram)],
            dtype=torch.float64,
            device=next(model.parameters()).device,
        )
        if is_dist_avail_and_initialized():
            gathered = [torch.zeros_like(local_runtime) for _ in range(get_world_size())]
            dist.all_gather(gathered, local_runtime)
        else:
            gathered = [local_runtime]
        values = [item.detach().cpu().tolist() for item in gathered]
        global_examples = sum(item[0] for item in values)
        max_elapsed = max(item[1] for item in values)
        stats["epoch_wall_seconds"] = max_elapsed
        stats["optimizer_step_seconds"] = max_elapsed / updates_per_epoch
        stats["samples_per_second"] = global_examples / max_elapsed
        for rank, item in enumerate(values):
            stats[f"peak_vram_rank{rank}_bytes"] = int(item[2])
        return stats

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
