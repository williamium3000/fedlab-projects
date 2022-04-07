# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math


class Aggregators(object):
    """Define the algorithm of parameters aggregation"""

    @staticmethod
    def fedavg_aggregate(serialized_params_list, weights=None):
        """FedAvg aggregator

        Paper: http://proceedings.mlr.press/v54/mcmahan17a.html

        Args:
            serialized_params_list (list[torch.Tensor])): Merge all tensors following FedAvg.
            weights (list, numpy.array or torch.Tensor, optional): 
            Weights for each params, the length of weights need to 
            be same as length of ``serialized_params_list``. Default: None.

        Returns:
            torch.Tensor
        """
        if weights is None:
            weights = torch.ones(len(serialized_params_list))

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)

        weights = weights / torch.sum(weights)
        assert torch.all(weights >= 0), "weights should be non-negative values"
        serialized_parameters = torch.sum(torch.stack(serialized_params_list, dim=-1) * weights[None, ...],
                                          dim=-1)

        return serialized_parameters
    
    
    
    @staticmethod
    def fedavg_bounded_aggregate(serialized_params_list, weights=None, max_norm=None, p="fro"):
        """FedAvg aggregator bounded with norm. Notice this aggregator requires differential input.

        Paper: http://proceedings.mlr.press/v54/mcmahan17a.html

        Args:
            serialized_params_list (list[torch.Tensor])): Each tensor represent one client update (in differential form).
            weights (list, numpy.array or torch.Tensor, optional): 
            Weights for each params, the length of weights need to be same as 
            length of ``serialized_params_list``. Default: None.
            max_norm (float): Maximum norm. Update is norm bounded by this. Default: None.
            p (int, float, inf, -inf, 'fro', 'nuc', optional): the order of norm. Default: 'fro'.
        Returns:
            torch.Tensor
        """
        if weights is None:
            weights = torch.ones(len(serialized_params_list))

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)

        weights = weights / torch.sum(weights)
        assert torch.all(weights >= 0), "weights should be non-negative values"
        
        params = torch.stack(serialized_params_list, dim=-1)
        if max_norm is not None:
            params_norm = torch.norm(params, p=p, dim=-1, keepdim=True)
            params = params * torch.minimum(max_norm / params_norm, torch.ones_like(params_norm))
        
        serialized_parameters = torch.sum(params * weights[None, ...],
                                          dim=-1)

        return serialized_parameters
    
    @staticmethod
    def krum_aggregate(serialized_params_list, weights=None, discard_fraction=None):
        """FedAvg aggregator bounded with norm. Notice this aggregator requires differential input.

        Paper: http://proceedings.mlr.press/v54/mcmahan17a.html

        Args:
            serialized_params_list (list[torch.Tensor])): Each tensor represent one client update (in differential form).
            weights (list, numpy.array or torch.Tensor, optional): 
            Weights for each params, the length of weights need to be same as 
            length of ``serialized_params_list``. Default: None.
            max_norm (float): Maximum norm. Update is norm bounded by this. Default: None.
            p (int, float, inf, -inf, 'fro', 'nuc', optional): the order of norm. Default: 'fro'.
        Returns:
            torch.Tensor
        """
        if weights is None:
            weights = torch.ones(len(serialized_params_list))

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)
        assert torch.all(weights >= 0), "weights should be non-negative values"
        n = len(serialized_params_list)
        # f: expected number of corrupted updates; cap it at n // 2
        num_corrupt = min(len(serialized_params_list) // 2 - 1, math.ceil(len(serialized_params_list) * discard_fraction))
        num_good = n - num_corrupt - 2  # n - f - 2
        m = n - num_corrupt  # parameter `m` in the paper
        params = torch.stack(serialized_params_list, dim=0)
        sqdist = torch.cdist(params.unsqueeze(0), params.unsqueeze(0))[0] ** 2
        scores = torch.zeros(n)
        
        for i in range(n):
            scores[i] = torch.sort(sqdist[i])[0][:num_good + 1].sum()  # exclude k = i
        good_idxs = torch.argsort(scores)[:m]
        params = params[good_idxs]
        weights = weights[good_idxs]
        weights = weights / torch.sum(weights)
        serialized_parameters = torch.sum(params * weights[..., None],
                                          dim=0)

        return serialized_parameters
    
    @staticmethod
    def fedavg_sign_aggregate(serialized_params_list, weights=None, gamma=1e-3):
        """Sign FedAvg aggregator. Notice this aggregator requires differential input.

        Paper: https://arxiv.org/pdf/1802.04434.pdf

        Args:
            serialized_params_list (list[torch.Tensor])): Each tensor represent one client update (in differential form).
            weights (list, numpy.array or torch.Tensor, optional): 
            Weights for each params, the length of weights need to be same as 
            length of ``serialized_params_list``. Default: None.

        Returns:
            torch.Tensor
        """
        if weights is None:
            weights = torch.ones(len(serialized_params_list))

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)

        weights = weights / torch.sum(weights)
        assert torch.all(weights >= 0), "weights should be non-negative values"
        
        params = torch.sign(torch.stack(serialized_params_list, dim=-1))
        serialized_parameters = torch.sign(torch.sum(params * weights[None, ...],
                                          dim=-1))

        return serialized_parameters * gamma
    @staticmethod
    def trimmed_mean_aggregate(serialized_params_list, weights=None, discard_fraction=None):
        """Trimmed Mean aggregator. Notice this aggregator requires differential input.

        Paper: 

        Args:
            serialized_params_list (list[torch.Tensor])): Each tensor represent one client update (in differential form).
            weights (list, numpy.array or torch.Tensor, optional): Weights for each params, the length of weights need to be same as length of ``serialized_params_list``
            discard_fraction (float [0, 1]): Fraction to discard.
        Returns:
            torch.Tensor
        """
        if weights is None:
            weights = torch.ones(len(serialized_params_list))

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)

        assert torch.all(weights >= 0), "weights should be non-negative values"
    
        if discard_fraction == 0.0:
            # if no trim, then returns weighted average of updates
            weights = weights / torch.sum(weights)
            return torch.sum(torch.stack(serialized_params_list, dim=-1) * weights[None, ...], dim=-1)

        half_discard = min(len(serialized_params_list) // 2 - 1, math.ceil(len(serialized_params_list) * discard_fraction) // 2)
        aggregated_update = torch.zeros_like(serialized_params_list[0])
        for i in range(aggregated_update.shape[0]):
            p_i = torch.stack([p[i] for p in serialized_params_list])
            idxs = torch.argsort(p_i)[half_discard: -half_discard]
            weights_i = weights[idxs] / torch.sum(weights[idxs])
            aggregated_update[i] = (p_i[idxs] * weights_i).sum()
        return aggregated_update
    
    @staticmethod
    def fedasync_aggregate(server_param, new_param, alpha):
        """FedAsync aggregator
        
        Paper: https://arxiv.org/abs/1903.03934
        """
        serialized_parameters = torch.mul(1 - alpha, server_param) + \
                                torch.mul(alpha, new_param)
        return serialized_parameters
