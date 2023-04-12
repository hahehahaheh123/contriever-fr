import os
import time
import sys
import torch
import logging
import json
import numpy as np
import random
import pickle

import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler

from src.options import Options
from src import data, beir_utils, slurm, dist_utils, utils
from src import moco, inbatch


logger = logging.getLogger(__name__)


def train(opt, model, optimizer, scheduler, step):

    run_stats = utils.WeightedAvgStats()

    tb_logger = utils.init_tb_logger(opt.output_dir)

    logger.info("Data loading")
    
    # distributed mode or single mode
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        tokenizer = model.module.tokenizer
    else:
        tokenizer = model.tokenizer
        
    # data loading
    collator = data.Collator(opt=opt)
    train_dataset = data.load_data(opt, tokenizer)
    logger.warning(f"Data loading finished for rank {dist_utils.get_rank()}")

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=opt.num_workers,  # number of sub-processes used by the graphics card to collect batches
        collate_fn=collator,
    )

    epoch = 0

    model.train()
    while epoch < opt.total_epochs:  # i corrected something
        train_dataset.generate_offset()

        logger.info(f"Start epoch {epoch}")
        for i, batch in enumerate(train_dataloader):
            step += 1

            # move batch values to gpu if tensor, else remain the same
            batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            train_loss, iter_stats = model(**batch, stats_prefix="train")
            # remember that the bert model is wrapped in contriever class, and then wrapped again in MoCo class.

            train_loss.backward()
            optimizer.step()

            scheduler.step()
            model.zero_grad()

            run_stats.update(iter_stats)

            # INFO: step progression, update stats, get lr and memory of the process
            if step % opt.log_freq == 0:
                log = f"{step} / {opt.total_steps}"
                for k, v in sorted(run_stats.average_stats.items()):
                    log += f" | {k}: {v:.3f}"
                    if tb_logger:
                        tb_logger.add_scalar(k, v, step)
                log += f" | lr: {scheduler.get_last_lr()[0]:0.3g}"
                log += f" | Memory: {torch.cuda.max_memory_allocated()//1e9} GiB"

                logger.info(log)
                run_stats.reset()

            # evaluate and save the model every now and then
            if step % opt.eval_freq == 0:
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    encoder = model.module.get_encoder()
                else:
                    encoder = model.get_encoder()
                eval_model(
                    opt, query_encoder=encoder, doc_encoder=encoder, tokenizer=tokenizer, tb_logger=tb_logger, step=step
                )

                if dist_utils.is_main():
                    utils.save(model, optimizer, scheduler, step, opt, opt.output_dir, f"lastlog")

                model.train()

            if dist_utils.is_main() and step % opt.save_freq == 0:
                utils.save(model, optimizer, scheduler, step, opt, opt.output_dir, f"step-{step}")

            if step > opt.total_steps:
                break
        epoch += 1
