import os
import sys  # access to parameters and functionalities specific to python runtime environment
import logging
import torch
import errno  # error indication
from typing import Union, Tuple, List, Dict
from collections import defaultdict

from src import dist_utils

Number = Union[float, int]

logger = logging.getLogger(__name__)


def init_logger(args, stdout_only=False):
# initialize logging system (where logger outputs are, its handlers, config settings, all that jazz)
  if torch.distributed.is_initialized():
    torch.distributed.barrier()  # pause process until others have reached this point
  stdout_handler = logging.StreamHandler(sys.stdout)  # specify logging output (to file-like object)
  handlers = [stdout_handler]
  if not stdout_only:
    file_handler = logging.FileHandler(filename=os.path.join(args.output_dir, "run.log"))
    # second logging output (to disk file)
    handlers.append(file_handler)
    
  logging.basicConfig(
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if dist_utils.is_main() else logging.WARN,  # set log message level
    format="[%(asctime)s] {%(filename)s:%(linedo)d} %(levelname)s - %(message)s",
    handlers=handlers,
  )
  return logger
    
  
def symlink_force(target, link_name):
# establish a soft link (unix file pointer)
  try:
    os.symlink(target, link_name)
  except OSError as e:
    if e.errno == errno.EEXIST:
      os.remove(link_name)
      os.symlink(target, link_name)
    else:
      raise e
      
      
def save(model, optimizer, scheduler, step, opt, dir_path, name):
  
  # establish all directories
  model_to_save = model.module if hasattr(model, "module") else model  # ?
  path = os.path.join(dir_path, "checkpoint")
  epoch_path = os.path.join(path, name)  # "step-%s" % step)
  os.makedirs(epoch_path, exist_ok=True)  # creates any intermediate missing directories
  cp = os.path.join(path, "latest")
  fp = os.path.join(epoch_path, "checkpoint.pth")
  
  # save checkpoint dict in dir_path/checkpoint/name/checkpoint.pth
  checkpoint = {
    "step": step
    "model": model_to_save.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "opt": opt,
  }
  torch.save(checkpoint, fp)
  # establish dir_path/checkpoint/latest as a pointer to dir_path/checkpoint/name
  symlink_force(epoch_path, cp)
  if not name == "lastlog":
    logger.info(f"Saving model to {epoch_path}")
    
    
def load(model_class, dir_path, opt, reset_params=False):
  """
  1. establish directories to receive from
  2. load opt
  3. load model state_dict
  4. load optimizer & scheduler state_dict
  """
    epoch_path = os.path.realpath(dir_path)  # eliminates any symlinks, returns r e a l path
    checkpoint_path = os.path.join(epoch_path, "checkpoint.pth")
    logger.info(f"loading checkpoint {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    opt_checkpoint = checkpoint["opt"]
    state_dict = checkpoint["model"]
    
    model = model_class(opt_checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model = model.cuda()
    step = checkpoint["step"]
    if not reset_params:
      optimizer, scheduler = set_optim(opt_checkpoint, model)
      scheduler.load_state_dict(checkpoint["scheduler"])
      optimizer.load_state_dict(checkpoint["optimizer"])
    else:
      optimizer, scheduler = set_optim(opt, model)
      
    return model, optimizer, scheduler, opt_checkpoint, step
  
  
############ OPTIM


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
  def __init__(self, optimizer, warmup, total, ratio, last_epoch=-1):
    self.warmup = warmup
    self.total = total
    self.ratio = ratio
    super(WarmupLinearScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
    
  def lr_lambda(self, step):
    if step < self.warmup:
      return (1 - self.ratio) * step / float(max(1, self.warmup))
    
    return max(
      0.0,
      1.0 + (self.ratio - 1) * (step - self.warmup) / float(max(1.0, self.total - self.warmup)),
    )
  

class CosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio=0.1, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(CosineScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return float(step) / self.warmup
        s = float(step - self.warmup) / (self.total - self.warmup)
        return self.ratio + (1.0 - self.ratio) * math.cos(0.5 * math.pi * s)


def set_optim(opt, model):
    if opt.optim == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay
        )
    else:
        raise NotImplementedError("optimizer class not implemented")

    scheduler_args = {
        "warmup": opt.warmup_steps,
        "total": opt.total_steps,
        "ratio": opt.lr_min_ratio,
    }
    if opt.scheduler == "linear":
        scheduler_class = WarmupLinearScheduler
    elif opt.scheduler == "cosine":
        scheduler_class = CosineScheduler
    else:
        raise ValueError
    scheduler = scheduler_class(optimizer, **scheduler_args)
    return optimizer, scheduler


def get_parameters(net, verbose=False):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    message = "[Network] Total number of parameters : %.6f M" % (num_params / 1e6)
    return message


class WeightedAvgStats:
    """provides an average over a bunch of stats"""

    def __init__(self):
        self.raw_stats: Dict[str, float] = defaultdict(float)
        self.total_weights: Dict[str, float] = defaultdict(float)

    def update(self, vals: Dict[str, Tuple[Number, Number]]) -> None:
        for key, (value, weight) in vals.items():
            self.raw_stats[key] += value * weight
            self.total_weights[key] += weight

    @property
    def stats(self) -> Dict[str, float]:
        return {x: self.raw_stats[x] / self.total_weights[x] for x in self.raw_stats.keys()}

    @property
    def tuple_stats(self) -> Dict[str, Tuple[float, float]]:
        return {x: (self.raw_stats[x] / self.total_weights[x], self.total_weights[x]) for x in self.raw_stats.keys()}

    def reset(self) -> None:
        self.raw_stats = defaultdict(float)
        self.total_weights = defaultdict(float)

    @property
    def average_stats(self) -> Dict[str, float]:
        keys = sorted(self.raw_stats.keys())
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(keys, src=0)
        global_dict = {}
        for k in keys:
            if not k in self.total_weights:
                v = 0.0
            else:
                v = self.raw_stats[k] / self.total_weights[k]
            v, _ = dist_utils.weighted_average(v, self.total_weights[k])
            global_dict[k] = v
        return global_dict


def load_hf(object_class, model_name):
    try:
        obj = object_class.from_pretrained(model_name, local_files_only=True)
    except:
        obj = object_class.from_pretrained(model_name, local_files_only=False)
    return obj


def init_tb_logger(output_dir):
    try:
        from torch.utils import tensorboard

        if dist_utils.is_main():
            tb_logger = tensorboard.SummaryWriter(output_dir)
        else:
            tb_logger = None
    except:
        logger.warning("Tensorboard is not available.")
        tb_logger = None

    return tb_logger
