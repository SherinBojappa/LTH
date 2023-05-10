# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import io
import csv
import torch
import torch.nn as nn
import numpy as np

from cli import shared_args
from dataclasses import dataclass
from foundations.runner import Runner
import models.registry
from lottery.desc import LotteryDesc
from platforms.platform import get_platform
import pruning.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train


@dataclass
class LotteryRunner(Runner):
    replicate: int
    levels: int
    desc: LotteryDesc
    verbose: bool = True
    evaluate_every_epoch: bool = True

    @staticmethod
    def description():
        return 'Run a lottery ticket hypothesis experiment.'

    @staticmethod
    def _add_levels_argument(parser):
        help_text = \
            'The number of levels of iterative pruning to perform. At each level, the network is trained to ' \
            'completion, pruned, and rewound, preparing it for the next lottery ticket iteration. The full network ' \
            'is trained at level 0, and level 1 is the first level at which pruning occurs. Set this argument to 0 ' \
            'to just train the full network or to N to prune the network N times.'
        parser.add_argument('--levels', required=True, type=int, help=help_text)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        # Get preliminary information.
        defaults = shared_args.maybe_get_default_hparams()

        # Add the job arguments.
        shared_args.JobArgs.add_args(parser)
        lottery_parser = parser.add_argument_group(
            'Lottery Ticket Hyperparameters', 'Hyperparameters that control the lottery ticket process.')
        LotteryRunner._add_levels_argument(lottery_parser)
        LotteryDesc.add_args(parser, defaults)

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'LotteryRunner':
        return LotteryRunner(args.replicate, args.levels, LotteryDesc.create_from_args(args),
                             not args.quiet, not args.evaluate_only_at_end)

    def display_output_location(self):
        print(self.desc.run_path(self.replicate, 0))

    def run(self) -> None:
        #import pdb; pdb.set_trace()
        if self.verbose and get_platform().is_primary_process:
            print('='*82 + f'\nLottery Ticket Experiment (Replicate {self.replicate})\n' + '-'*82)
            print(self.desc.display)
            print(f'Output Location: {self.desc.run_path(self.replicate, 0)}' + '\n' + '='*82 + '\n')

        if get_platform().is_primary_process: self.desc.save(self.desc.run_path(self.replicate, 0))
        if self.desc.pretrain_training_hparams: self._pretrain()
        if get_platform().is_primary_process: self._establish_initial_weights()
        get_platform().barrier()

        output_filename = "ranks_output_svd.txt"
        singular_values_filename = "singular_values.csv"
        mask_filename = "mask.txt"
        if os.path.exists(output_filename):
            os.remove(output_filename)

        if os.path.exists(singular_values_filename):
            os.remove(singular_values_filename)


        fh = open(output_filename, 'a')
        ch = open(singular_values_filename, 'a')
        mh = open(mask_filename, 'a')
        csv_writer = csv.writer(ch)

        for level in range(self.levels+1):
            #import pdb; pdb.set_trace()
            if get_platform().is_primary_process: self._prune_level(level)
            #if level == 1:
                # write the masks to a file
                #new_location = self.desc.run_path(self.replicate, level)
                #mh.write(f"Mask for magnitude pruning\n")
                #mask = Mask.load(new_location)
                # print the values in mask
                #for k,v in mask.items():
                    #mh.write(f"{k} : {v.tolist()}\n")
            get_platform().barrier()
            self._train_level(level, output_filename, fh, csv_writer)

        # prune randomly
        #import pdb; pdb.set_trace()
        self._prune_random(output_filename, fh, csv_writer, mh)


    # Helper methods for running the lottery.
    def _pretrain(self):
        location = self.desc.run_path(self.replicate, 'pretrain')
        if models.registry.exists(location, self.desc.pretrain_end_step): return

        if self.verbose and get_platform().is_primary_process: print('-'*82 + '\nPretraining\n' + '-'*82)
        model = models.registry.get(self.desc.model_hparams, outputs=self.desc.pretrain_outputs)
        train.standard_train(model, location, self.desc.pretrain_dataset_hparams, self.desc.pretrain_training_hparams,
                             verbose=self.verbose, evaluate_every_epoch=self.evaluate_every_epoch)

    def _establish_initial_weights(self):
        location = self.desc.run_path(self.replicate, 0)
        if models.registry.exists(location, self.desc.train_start_step): return

        new_model = models.registry.get(self.desc.model_hparams, outputs=self.desc.train_outputs)

        # If there was a pretrained model, retrieve its final weights and adapt them for training.
        if self.desc.pretrain_training_hparams is not None:
            pretrain_loc = self.desc.run_path(self.replicate, 'pretrain')
            old = models.registry.load(pretrain_loc, self.desc.pretrain_end_step,
                                       self.desc.model_hparams, self.desc.pretrain_outputs)
            state_dict = {k: v for k, v in old.state_dict().items()}

            # Select a new output layer if number of classes differs.
            if self.desc.train_outputs != self.desc.pretrain_outputs:
                state_dict.update({k: new_model.state_dict()[k] for k in new_model.output_layer_names})

            new_model.load_state_dict(state_dict)

        new_model.save(location, self.desc.train_start_step)

    def _train_level(self, level: int, output_filename: str, fh: io.TextIOWrapper, csv_writer: csv.writer):
        #import pdb; pdb.set_trace()
        location = self.desc.run_path(self.replicate, level)
        if models.registry.exists(location, self.desc.train_end_step): return

        model = models.registry.load(self.desc.run_path(self.replicate, 0), self.desc.train_start_step,
                                     self.desc.model_hparams, self.desc.train_outputs)
        # compute the rank before pruning; level 0 compares to just the initialized model
        # before training, level 1 first level of pruning
        if level == 0:
            fh.write(f"Model architecture\n")
            fh.write(f"{model}")
            fh.write(f"Rank of the unpruned model\n")
            self._compute_rank_svd(model, level, output_filename, fh, csv_writer)
            csv_writer.writerow([])
        pruned_model = PrunedModel(model, Mask.load(location))
        # compute rank after pruning including the level of pruning
        # level 0 is just full training of the network
        if level != 0:
            fh.write(f"Rank of the pruned model before training\n")
            self._compute_rank_svd(pruned_model, level, output_filename, fh, csv_writer)
            csv_writer.writerow([])

        pruned_model.save(location, self.desc.train_start_step)
        if self.verbose and get_platform().is_primary_process:
            print('-'*82 + '\nPruning Level {}\n'.format(level) + '-'*82)
        train.standard_train(pruned_model, location, self.desc.dataset_hparams, self.desc.training_hparams,
                             start_step=self.desc.train_start_step, verbose=self.verbose,
                             evaluate_every_epoch=self.evaluate_every_epoch)
        if level != 0:
            fh.write(f"Rank of the pruned model after training\n")
        else:
            fh.write(f"Rank of the unpruned model after training\n")
        self._compute_rank_svd(pruned_model, level, output_filename, fh, csv_writer)
        csv_writer.writerow([])

    def _prune_level(self, level: int):
        new_location = self.desc.run_path(self.replicate, level)
        if Mask.exists(new_location): return

        if level == 0:
            Mask.ones_like(models.registry.get(self.desc.model_hparams)).save(new_location)
        else:
            old_location = self.desc.run_path(self.replicate, level-1)
            model = models.registry.load(old_location, self.desc.train_end_step,
                                         self.desc.model_hparams, self.desc.train_outputs)
            pruning.registry.get(self.desc.pruning_hparams)(model, Mask.load(old_location)).save(new_location)

    def _prune_random(self, output_filename, fh, csv_writer, mh):
        # this is to bypass writing to an exisiting mask file
        level = 1
        # indicates random pruning
        random = True
        new_location = self.desc.run_path(self.replicate, level+1)
        if Mask.exists(new_location): return

        # create a new mask file with all ones first
        Mask.ones_like(models.registry.get(self.desc.model_hparams)).save(new_location)
        # old_location now points to the initial model
        old_location = self.desc.run_path(self.replicate, level-1)
        # load the initial model
        model = models.registry.load(self.desc.run_path(self.replicate, 0), self.desc.train_start_step,
                                     self.desc.model_hparams, self.desc.train_outputs)
        # get random mask
        pruning.registry.get_random(self.desc.pruning_hparams)(model, Mask.load(new_location)).save(new_location)

        # write the masks to a file
        mh.write(f"Mask for random pruning\n")
        mh.write(f"{Mask.load(new_location)}")

        pruned_model = PrunedModel(model, Mask.load(new_location))
        pruned_model.save(new_location, self.desc.train_start_step)
        # compute rank after pruning including the level of pruning
        fh.write(f"Rank of a random subnetwork\n")
        self._compute_rank_svd(pruned_model, level, output_filename, fh, csv_writer)
        # evaluate the performance of the random subnetwork
        train.standard_train(pruned_model, new_location, self.desc.dataset_hparams, self.desc.training_hparams,
                        start_step=self.desc.train_start_step, verbose=self.verbose,
                        evaluate_every_epoch=self.evaluate_every_epoch)


    def _compute_rank(self, model, level, output_filename, fh):
        ranks = {}

        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                # For linear layers, use the weight matrix directly
                if isinstance(layer, nn.Linear):
                    weight_matrix = layer.weight.detach().cpu()
                # For convolutional layers, reshape the weight tensor into a 2D matrix
                else:
                    weight_matrix = layer.weight.view(layer.weight.size(0), -1).detach().cpu()

                l1_norm = torch.sum(torch.abs(weight_matrix)).item()
                rank = torch.matrix_rank(weight_matrix)
                ranks[name] = rank
                file_op = f"L1 norm of layer {name} is {l1_norm}"
                fh.write(file_op)
                output_line = f"Level_{level}: Rank of layer '{name}': {rank}\n"
                fh.write(output_line)

    def _compute_rank_svd(self, model, level, output_filename, fh, csv_writer):
        ranks = {}

        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                # For linear layers, use the weight matrix directly
                if isinstance(layer, nn.Linear):
                    weight_matrix = layer.weight.detach().cpu()
                # For convolutional layers, reshape the weight tensor into a 2D matrix
                else:
                    weight_matrix = layer.weight.view(layer.weight.size(0), -1).detach().cpu()

                l1_norm = torch.sum(torch.abs(weight_matrix)).item()
                u, s, v = torch.svd(weight_matrix)
                # threshold 1e-5 and 1e-4 not much difference in the rank between pruned and unpruned
                threshold = torch.max(s)*0.5
                # log the singular values
                csv_writer.writerow(s.tolist())
                rank = (s > threshold).sum().item()
                ranks[name] = rank
                file_op = f"L1 norm of layer {name} is {l1_norm}"
                fh.write(file_op)
                output_line = f"Level_{level}: Rank of layer '{name}': {rank}\n"
                fh.write(output_line)