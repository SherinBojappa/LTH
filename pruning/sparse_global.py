# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import numpy as np

from foundations import hparams
import models.base
from pruning import base
from pruning.mask import Mask


@dataclasses.dataclass
class PruningHparams(hparams.PruningHparams):
    pruning_fraction: float = 0.9
    pruning_layers_to_ignore: str = None

    _name = 'Hyperparameters for Sparse Global Pruning'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_fraction = 'The fraction of additional weights to prune from the network.'
    _layers_to_ignore = 'A comma-separated list of addititonal tensors that should not be pruned.'


class Strategy(base.Strategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    def prune(pruning_hparams: PruningHparams, trained_model: models.base.Model, current_mask: Mask = None):
        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()

        # Determine the number of weights that need to be pruned.
        number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
        number_of_weights_to_prune = np.ceil(
            pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)

        # Determine which layers can be pruned.
        prunable_tensors = set(trained_model.prunable_layer_names)
        if pruning_hparams.pruning_layers_to_ignore:
            prunable_tensors -= set(pruning_hparams.pruning_layers_to_ignore.split(','))

        # Get the model weights.
        weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in trained_model.state_dict().items()
                   if k in prunable_tensors}

        # Create a vector of all the unpruned weights in the model.
        weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
        threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune]

        new_mask = Mask({k: np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))
                         for k, v in weights.items()})
        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]

        return new_mask

    def prune_random(pruning_hparams: PruningHparams, trained_model: models.base.Model, current_mask: Mask = None):
        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()

        # Determine the number of weights that need to be pruned.
        number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
        number_of_weights_to_prune = np.ceil(
            pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)

        # Determine which layers can be pruned.
        prunable_tensors = set(trained_model.prunable_layer_names)
        if pruning_hparams.pruning_layers_to_ignore:
            prunable_tensors -= set(pruning_hparams.pruning_layers_to_ignore.split(','))

        # Get the model weights.
        weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in trained_model.state_dict().items()
                   if k in prunable_tensors}

        # Randomly select weights to prune

        # Find the indices of unmasked elements in all layers and store them in a list
        unmasked_indices_list = [np.flatnonzero(current_mask[k] == 1) for k, v in weights.items()]

        # Concatenate the indices from all layers into a single 1D array
        all_unmasked_indices = np.concatenate(unmasked_indices_list)

        # Randomly select indices to prune from the 1D array
        indices_to_prune = np.random.choice(all_unmasked_indices, size=min(number_of_weights_to_prune, len(all_unmasked_indices)), replace=False)

        # Create a boolean mask indicating which indices have been selected for pruning
        selected_indices_mask = np.isin(all_unmasked_indices, indices_to_prune)

        # Initialize an empty dictionary to store the new mask
        new_mask = {}

        # Iterate through the layers and create the new mask based on the selected indices
        layer_start = 0
        for idx, (k, v) in enumerate(weights.items()):
            layer_mask = np.ones_like(v)

            # Determine the layer's selected indices for pruning
            layer_end = layer_start + len(unmasked_indices_list[idx])
            layer_selected_indices = np.flatnonzero(selected_indices_mask[layer_start:layer_end])

            # Update the layer_start for the next iteration
            layer_start = layer_end

            # Set the layer's pruned indices to 0 in the new mask
            layer_mask[np.unravel_index(layer_selected_indices, v.shape)] = 0
            new_mask[k] = layer_mask

        # Pass the new mask to the Mask class constructor
        new_mask = Mask(new_mask)

        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]

        return new_mask
