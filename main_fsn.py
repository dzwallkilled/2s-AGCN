#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from utils.config import get_parser
from utils.msic import *
from utils.metrics import Metrics, AverageMeter
from data.h36m.definitions import VAR_NAMES, JOINTS_OF_PARTS, CLASS_NUMS


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def init_seed(_):
    torch.cuda.manual_seed_all(123)
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, args):
        self.args = args
        self.save_arg()
        self.body_parts = list(JOINTS_OF_PARTS.keys())
        self.body_part_names = VAR_NAMES
        self.num_classes = [CLASS_NUMS[b] for b in self.body_parts]
        if args.phase == 'train':
            if not args.train_feeder_args['debug']:
                if os.path.isdir(args.model_saved_name):
                    print('log_dir: ', args.model_saved_name, 'already exist')
                    # answer = input('delete it? y/n:')
                    # if answer == 'y':
                    #     shutil.rmtree(args.model_saved_name)
                    #     print('Dir removed: ', args.model_saved_name)
                    #     input('Refresh the website of tensorboard by pressing any keys')
                    # else:
                    #     print('Dir not removed: ', args.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(args.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(args.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(args.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.args.base_lr
        self.best_acc = 0

    def load_data(self):
        Feeder = import_class(self.args.feeder)
        self.data_loader = dict()
        if self.args.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.args.train_feeder_args, phase='train', body_parts=self.body_parts),
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.args.test_feeder_args, phase='test', body_parts=self.body_parts),
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=self.args.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = get_available_device(self.args.num_gpus, )
        self.output_device = output_device
        Model = import_class(self.args.model)
        shutil.copy2(inspect.getfile(Model), self.args.work_dir)
        print(Model)
        self.model = Model(self.num_classes, **self.args.model_args).to(self.output_device)
        print(self.model)
        # loss implemented in network forward function
        # self.loss = nn.CrossEntropyLoss().cuda(output_device)

        if self.args.weights:
            self.global_step = int(args.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.args.weights))
            if '.pkl' in self.args.weights:
                with open(self.args.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.args.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.args.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if self.args.num_gpus > 1:
            self.model = nn.DataParallel(self.model).to(self.output_device)

    def load_optimizer(self):
        if self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.base_lr,
                momentum=0.9,
                nesterov=self.args.nesterov,
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.base_lr,
                weight_decay=self.args.weight_decay)
        else:
            raise ValueError()

        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.args.step, gamma=0.1)

        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=self.args.warm_up_epoch,
                                                   after_scheduler=lr_scheduler_pre)
        self.print_log('using warm up, epoch: {}'.format(self.args.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.args)
        if not os.path.exists(self.args.work_dir):
            os.makedirs(self.args.work_dir)
        with open('{}/config.yaml'.format(self.args.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.args.optimizer == 'SGD' or self.args.optimizer == 'Adam':
            if epoch < self.args.warm_up_epoch:
                lr = self.args.base_lr * (epoch + 1) / self.args.warm_up_epoch
            else:
                lr = self.args.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.args.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.args.print_log:
            with open('{}/log.txt'.format(self.args.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        # for name, param in self.model.named_parameters():
        #     self.train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        loss_value = []
        loss_value_joint = []
        loss_value_bone = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        for batch_idx, (joint_data, bone_data, _, label) in enumerate(process):
            self.global_step += 1
            # get data
            joint_data = joint_data.to(self.output_device)
            bone_data = bone_data.to(self.output_device)
            label = [l.to(self.output_device) for l in label]
            # data = Variable(data.float().cuda(self.output_device), requires_grad=False)
            # label = Variable(label.long().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()

            # forward
            output1, loss1, output2, loss2 = self.model(joint_data, bone_data, label)
            # if batch_idx == 0 and epoch == 0:
            #     self.train_writer.add_graph(self.model, output)

            loss = loss1 + loss2
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(to_item(loss))
            loss_value_joint.append(to_item(loss1))
            loss_value_bone.append(to_item(loss2))
            timer['model'] += self.split_time()

            self.train_writer.add_scalar('loss', to_item(loss), self.global_step)
            self.train_writer.add_scalar('loss_joint', to_item(loss1), self.global_step)
            self.train_writer.add_scalar('loss_bone', to_item(loss2), self.global_step)
            # self.train_writer.add_scalar('batch_time', process.iterable.last_duration, self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            # if self.global_step % self.arg.log_interval == 0:
            #     self.print_log(
            #         '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
            #             batch_idx, len(loader), loss.data[0], lr))
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                **proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],
                                    v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, self.args.model_saved_name + '-' + str(epoch) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, loader_name='test'):
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        self.model.eval()
        phase = loader_name
        loader = self.data_loader[loader_name]
        pbar = tqdm(loader)
        metrics = {'joint': Metrics(body_parts=self.body_parts, num_classes=self.num_classes),
                   'bone': Metrics(body_parts=self.body_parts, num_classes=self.num_classes),
                   '2stream': Metrics(body_parts=self.body_parts, num_classes=self.num_classes)}
        loss_meters = {'joint': AverageMeter(),
                       'bone': AverageMeter(),
                       '2stream': AverageMeter()}
        with torch.no_grad():
            for batch_index, (joint_data, bone_data, scores, label) in enumerate(pbar):
                joint_data = joint_data.to(self.output_device)
                bone_data = bone_data.to(self.output_device)
                scores = scores.to(self.output_device)
                label = [l.to(self.output_device) for l in label]
                output1, loss1, output2, loss2, output3, loss3 = self.model(joint_data, bone_data, label)
                loss_meters['2stream'].update(to_item(loss1+loss2), joint_data.size(0))
                loss_meters['bone'].update(to_item(loss2), joint_data.size(0))
                loss_meters['joint'].update(to_item(loss1), joint_data.size(0))
                metrics['joint'].update2(preds=output1, scores=scores)
                metrics['bone'].update2(output2, scores)
                metrics['2stream'].update2(output3, scores)

        for stream, metric in metrics.items():
            self.print_log(f'{epoch} | {phase} | {stream}')
            mae, kappa, accuracy = metric.metrics()
            self.print_log(f'loss reg: {loss_meters[stream].avg:.6f}')
            for b, m, k, a in zip(self.body_parts, mae, kappa, accuracy):
                self.print_log(f'{phase}| {self.body_part_names[b]} '
                            f'| MAE {m:.4f} | Kappa {k:.4f} | Acc {a:.2f}')

            self.print_log(
            f'{phase}| Mean: {sum(mae) / len(mae):.4f} {sum(kappa) / len(kappa):.4f} {sum(accuracy) / len(accuracy):.2f}')

            self.print_log('| '.join(self.body_part_names[b] for b in self.body_parts))
            self.print_log('& '.join(f'{m:.3f}/{k:.3f}' for m, k in zip(mae, kappa)))

            self.val_writer.add_scalar(f'{phase}/ loss', loss_meters[stream].avg, epoch)
            for b, m, k, a in zip(self.body_parts, mae, kappa, accuracy):
                self.val_writer.add_scalar(f'{phase}/{self.body_part_names[b]}/MAE', m, epoch)
                self.val_writer.add_scalar(f'{phase}/{self.body_part_names[b]}/kappa', k, epoch)
                self.val_writer.add_scalar(f'{phase}/{self.body_part_names[b]}/Acc', a, epoch)

    def start(self):
        if self.args.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.args))))
            self.global_step = self.args.start_epoch * len(self.data_loader['train']) / self.args.batch_size
            for epoch in range(self.args.start_epoch, self.args.num_epoch):
                if self.lr < 1e-3:
                    break
                save_model = ((epoch + 1) % self.args.save_interval == 0) or (
                        epoch + 1 == self.args.num_epoch)

                self.train(epoch, save_model=save_model)
                self.eval(epoch, loader_name='test')
            print('best accuracy: ', self.best_acc, ' model_name: ', self.args.model_saved_name)

        elif self.args.phase == 'test':
            if self.args.weights is None:
                raise ValueError('Please appoint --weights.')
            self.args.print_log = False
            self.print_log('Model:   {}.'.format(self.args.model))
            self.print_log('Weights: {}.'.format(self.args.weights))
            self.eval(epoch=0, loader_name='test')
            self.print_log('Done.\n')


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    args = parser.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    init_seed(123)
    processor = Processor(args)
    processor.start()
