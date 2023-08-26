import os
import random
import time
import tqdm
import socket
import visdom

import torch
import torch.distributed as dist
import numpy as np

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from conerf.base.checkpoint_manager import CheckPointManager


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def check_socket_open(hostname, port):
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    is_open = False
    try:
        s.bind((hostname,port))
    except socket.error:
        is_open = True
    finally:
        s.close()
    
    return is_open


def seed_worker(worker_id):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleSampler(object):
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


class BaseTrainer(object):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.trainer_name = 'BaseTrainer'
        self.config = config
        self.device = f"cuda:{config.local_rank}"

        self.output_path = os.path.join(config.root_dir, 'out', config.expname)
        if self.config.local_rank == 0:
            os.makedirs(self.output_path, exist_ok=True)
            print(f'[INFO] Outputs will be saved to {self.output_path}')

        self.log_file = open(os.path.join(self.output_path, 'log.txt'), 'w')
        self.scheduler = None
        self.model = None
        self.scalars_to_log = dict()
        self.ckpt_manager = CheckPointManager(
            save_path=self.output_path,
            max_to_keep=1000,
            keep_checkpoint_every_n_hours=0.5
        )
        
        self.train_done = False
        self._setup_visualizer()
        self.load_dataset()

        # Functions need to be overwritten.
        self.build_networks()
        self.setup_optimizer()
        self.setup_loss_functions()
        self.compose_state_dicts()

    def __del__(self):
        # if not self.train_done:
        #     score = self.validate()
        #     self.save_checkpoint(score=score)
        
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        
        self.log_file.close()

    def _check(self):
        assert self.train_dataset is not None
        # assert self.train_loader is not None
        assert self.val_dataset is not None
        # assert self.val_loader is not None
        assert self.model is not None
        assert os.path.exists(self.output_path) is True

        if self.config.distributed:
            assert self.train_sampler is not None

        if self.config.enable_tensorboard and self.config.local_rank == 0:
            assert self.writer is not None
        
        if self.config.enable_visdom and self.config.local_rank == 0:
            assert self.visdom is not None

    def load_dataset(self):
        raise NotImplementedError

    def _setup_visualizer(self):
        print('[INFO] Setting up visualizers...', file=self.log_file)
        self.writer = None
        self.visdom = None

        # Setup tensorboard.
        if self.config.enable_tensorboard and self.config.local_rank == 0:
            log_dir = os.path.join(self.config.root_dir, 'logs', self.config.expname)
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            print(f'[INFO] Saving tensorboard files to {log_dir}.')

        # Setup visdom.
        if self.config.enable_visdom and self.config.local_rank == 0:
            # check if visdom server is runninng
            is_open = check_socket_open(self.config.visdom_server, self.config.visdom_port)
            retry = None
            while not is_open:
                retry = input(f"visdom port ({self.config.visdom_port}) not open, retry? (y/n) ")
                if retry not in ["y", "n"]:
                    continue
                if retry == "y":
                    is_open = check_socket_open(self.config.visdom_server, self.config.visdom_port)
                else:
                    break

            self.visdom = visdom.Visdom(
                server=self.config.visdom_server,
                port=self.config.visdom_port,
                env='conerf'
            )
            print(f'[INFO] Visualizing camera poses at {self.config.visdom_server}:{self.config.visdom_port}')

    def build_networks(self):
        """
            Implement this function.
        """
        raise NotImplementedError

    def setup_optimizer(self):
        """
            Implement this function.
        """
        raise NotImplementedError

    def setup_loss_functions(self):
        """
            Implement this function.
        """
        raise NotImplementedError

    def train(self):
        # assert self.train_loader is not None
        # assert self.val_loader is not None

        pbar = tqdm.trange(self.config.n_iters, desc=f"Training {self.config.expname}", leave=False)

        iter_start = self.load_checkpoint(load_optimizer=not self.config.no_load_opt,
                                          load_scheduler=not self.config.no_load_scheduler)

        if self.config.distributed:
            # NOTE: Distributed mode can only be activated after loading models.
            self.model.to_distributed()
        
        self.epoch  = 0
        self.iteration = 0
        while self.iteration < iter_start:
            pbar.update(1)
            self.iteration += 1

        while self.iteration < self.config.n_iters + 1:
            for self.train_data in self.train_loader:
                if self.config.distributed:
                    self.train_sampler.set_epoch(self.epoch)
                
                # Main training logic.
                self.train_iteration(data_batch=self.train_data)

                if self.config.local_rank == 0:
                    # Main validation logic.
                    if self.iteration % self.config.n_validation == 0:
                        score = self.validate()
                    
                    # log to tensorboard.
                    if self.iteration % self.config.n_tensorboard == 0:
                        self.log_info()

                    # save checkpoint.
                    if self.iteration % self.config.n_checkpoint == 0:
                        score = self.validate()
                        self.save_checkpoint(score=score)
                
                pbar.update(1)
                
                self.iteration += 1
                if self.iteration > self.config.n_iters + 1:
                    break
            self.epoch += 1
        
        self.train_done = True

    def train_iteration(self, data_batch) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def validate(self) -> float:
        score = 0.
        """
            self.model.switch_to_eval()
            ... (implement validation logic here)
            self.model.switch_to_train()
        """
        
        return score

    def compose_state_dicts(self) -> None:
        """
            Implement this function and follow the format below:
            self.state_dicts = {'models': None, 'optimizers': None, 'schedulers': None}
        """
        
        raise NotImplementedError

    @torch.no_grad()
    def log_info(self) -> None:
        log_str = f'{self.config.expname} Epoch: {self.epoch}  step: {self.iteration} '
        
        for key in self.scalars_to_log.keys():
            log_str += ' {}: {:.6f}'.format(key, self.scalars_to_log[key])
            self.writer.add_scalar(key, self.scalars_to_log[key], self.iteration)
        
        print(log_str, file=self.log_file)

    def save_checkpoint(self, score: float = 0.0) -> None:
        assert self.state_dicts is not None

        self.ckpt_manager.save(
            models=self.state_dicts['models'],
            optimizers=self.state_dicts['optimizers'],
            schedulers=self.state_dicts['schedulers'],
            meta_data=self.state_dicts['meta_data'],
            step=self.iteration,
            score=score
        )

    def load_checkpoint(self, load_model=True, load_optimizer=True, load_scheduler=True, load_meta_data=False) -> int:
        iter_start = self.ckpt_manager.load(
            config=self.config,
            models=self.state_dicts['models'] if load_model else None,
            optimizers=self.state_dicts['optimizers'] if load_optimizer else None,
            schedulers=self.state_dicts['schedulers'] if load_scheduler else None,
            meta_data=self.state_dicts['meta_data'] if load_meta_data else None
        )

        return iter_start
