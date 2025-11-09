import json
import math

import ray
import argparse
import torch
from retrying import retry
import time
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os
import copy
from torch.distributions import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_set = [
    # (-10, -10), (-10, -5), (-10, 0), (-10, 5), (-10, 10),
    # (-10, -5), (-10, 5),
    (-5, -10), (-5, -5), (-5, 0), (-5, 5), (-5, 10),
    # (0, -10), (0, -5), (0, 5), (0, 10),
    (5, -10), (5, -5), (5, 0), (5, 5), (5, 10),
    # (10, -10), (10, -5), (10, 0), (10, 5), (10, 10)
    # (10, -5), (10, 5)
]


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class LocalSignalMixing(nn.Module):
    def __init__(self, pad, fixed_batch=False):
        """LIX regularization layer

        pad : float
            maximum regularization shift (maximum S)
        fixed batch : bool
            compute independent regularization for each sample (slower)
        """
        super().__init__()
        # +1 to avoid that the sampled values at the borders get smoothed with 0
        self.pad = int(math.ceil(pad)) + 1
        self.base_normalization_ratio = (2 * pad + 1) / (2 * self.pad + 1)
        self.fixed_batch = fixed_batch

    def get_random_shift(self, n, c, h, w, x):
        if self.fixed_batch:
            return torch.rand(size=(1, 1, 1, 2), device=x.device, dtype=x.dtype)
        else:
            return torch.rand(size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)

    def forward(self, x, max_normalized_shift=1.0):
        """
        x : Tensor
            input features
        max_normalized_shift : float
            current regularization shift in relative terms (current S)
        """
        if self.training:
            max_normalized_shift = max_normalized_shift * self.base_normalization_ratio
            n, c, h, w = x.size()
            assert h == w
            padding = tuple([self.pad] * 4)
            x = F.pad(x, padding, 'replicate')
            arange = torch.arange(h, device=x.device, dtype=x.dtype)  # from 0 to eps*h
            arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)  # shape=(h, h, 1)
            base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
            base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)  # 2d grid, shape=(n, h, h, 2)
            shift = self.get_random_shift(n, c, h, w, x)  # shape=(n,1,1,2) if not self.fixed_batch else (1,1,1,2)
            shift_offset = (1 - max_normalized_shift) / 2
            shift = (shift * max_normalized_shift) + shift_offset
            shift *= (2 * self.pad + 1)  # can start up to idx 2*pad + 1 - ignoring the left pad
            grid = base_grid + shift
            # normalize in [-1, 1]
            grid = grid * 2.0 / (h + 2 * self.pad) - 1
            return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        else:
            return x

    def get_grid(self, x, max_normalized_shift=1.0):
        max_normalized_shift = max_normalized_shift * self.base_normalization_ratio
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        arange = torch.arange(h, device=x.device, dtype=x.dtype)  # from 0 to eps*h
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)  # 2d grid
        shift = self.get_random_shift(n, c, h, w, x)
        shift_offset = (1 - max_normalized_shift) / 2
        shift = (shift * max_normalized_shift) + shift_offset
        shift *= (2 * self.pad + 1)
        grid = base_grid + shift
        # normalize in [-1, 1]
        grid = grid * 2.0 / (h + 2 * self.pad) - 1
        return grid


def get_local_patches_kernel(kernel_size, device):
    patch_dim = kernel_size ** 2
    k = torch.eye(patch_dim, device=device).view(patch_dim, 1,
                                                 kernel_size, kernel_size)
    return k


def extract_local_patches(input, kernel, N=None, padding=0, stride=1):
    b, c, _, _ = input.size()
    if kernel is None:
        kernel = get_local_patches_kernel(kernel_size=N, device=input.device)
    flinput = input.flatten(0, 1).unsqueeze(1)
    patches = F.conv2d(flinput, kernel, padding=padding, stride=stride)
    _, _, h, w = patches.size()
    return patches.view(b, c, -1, h, w)


class LearnS(torch.autograd.Function):
    '''Uses neighborhood around each feature gradient position to calculate the
        spatial divergence of the gradients, and uses it to update the param S,'''

    @staticmethod
    def forward(ctx, input, param, N, target_capped_ratio, eps):
        """
        input : Tensor
            representation to be processed (used for the gradient analysis).
        param : Tensor
            ALIX parameter S to be optimized.
        N : int
            filter size used to approximate the spatial divergence as a
            convolution (to calculate the ND scores), should be odd, >= 3
        target_capped_ratio : float
            target ND scores used to adaptively tune S
        eps : float
            small stabilization constant for the ND scores
        """
        ctx.save_for_backward(param)
        ctx.N = N
        ctx.target_capped_ratio = target_capped_ratio
        ctx.eps = eps
        return input

    @staticmethod
    def backward(ctx, dy):
        """
        compute the gradient of ALIX parameter S, which is indicted by param_grad
        dy : grad of output, shape=(batch, channel, h, w). The output is the input in the forward function,
            dy is the gradient of input in the forward function.
        """
        N = ctx.N
        target_capped_ratio = ctx.target_capped_ratio
        eps = ctx.eps
        dy_mean_B = dy.mean(0, keepdim=True)  # shape=(1, c, h, w)
        ave_dy_abs = torch.abs(dy_mean_B)
        pad_Hl = (N - 1) // 2
        pad_Hr = (N - 1) - pad_Hl
        pad_Wl = (N - 1) // 2
        pad_Wr = (N - 1) - pad_Wl
        pad = (pad_Wl, pad_Wr, pad_Hl, pad_Hr)
        padded_ave_dy = F.pad(dy_mean_B, pad, mode='replicate')
        loc_patches_k = get_local_patches_kernel(kernel_size=N, device=dy.device)

        local_patches_dy = extract_local_patches(
            input=padded_ave_dy, kernel=loc_patches_k, stride=1, padding=0)  # shape=(batch, channel, N, h, w)
        ave_dy_sq = ave_dy_abs.pow(2)  # shape=(batch, channel, h, w)
        patch_normalizer = (N * N) - 1

        # avoiding the center of kernel, As center to center doesn't correspond to any spatial direction
        unbiased_sq_signal = (local_patches_dy.pow(2).sum(
            dim=2) - ave_dy_sq) / patch_normalizer  # expected squared signal, delta z_{cij}^2 in equation (4)
        unbiased_sq_noise_signal = (local_patches_dy - dy_mean_B.unsqueeze(2)).pow(2).sum(
            2) / patch_normalizer  # 1 x C x x H x W expected squared noise

        unbiased_sqn2sig = (unbiased_sq_noise_signal) / (unbiased_sq_signal + eps)

        unbiased_sqn2sig_lp1 = torch.log(1 + unbiased_sqn2sig).mean()  # equation (4) in the paper
        param_grad = target_capped_ratio - unbiased_sqn2sig_lp1

        return dy, param_grad, None, None, None


class ParameterizedReg(nn.Module):
    """Augmentation/Regularization wrapper where the strength parameterized
       and is tuned with a custom autograd function

        aug : nn.Module
            augmentation/Regularization layer
        parameter_init : float
            initial strength value
        param_grad_fn : str
            custom autograd function to tune the parameter
        param_grad_fn_args : list
            arguments for the custom autograd function
        """

    def __init__(self, aug, parameter_init, param_grad_fn, param_grad_fn_args):
        super().__init__()
        self.aug = aug
        self.P = nn.Parameter(data=torch.tensor(parameter_init))
        self.param_grad_fn_name = param_grad_fn
        if param_grad_fn == 'alix_param_grad':
            self.param_grad_fn = LearnS.apply
        else:
            raise NotImplementedError
        self.param_grad_fn_args = param_grad_fn_args

    def forward(self, x):
        with torch.no_grad():
            self.P.copy_(torch.clamp(self.P, min=0, max=1))
        out = self.aug(x, self.P.detach())
        out = self.param_grad_fn(out, self.P, *self.param_grad_fn_args)
        return out

    def forward_no_learn(self, x):
        with torch.no_grad():
            self.P.copy_(torch.clamp(self.P, min=0, max=1))
        out = self.aug(x, self.P.detach())
        return out

    def forward_no_aug(self, x):
        with torch.no_grad():
            self.P.copy_(torch.clamp(self.P, min=0, max=1))
        out = x
        out = self.param_grad_fn(out, self.P, *self.param_grad_fn_args)
        return out


class NonLearnableParameterizedRegWrapper(nn.Module):
    def __init__(self, aug):
        super().__init__()
        self.aug = aug
        assert isinstance(aug, ParameterizedReg)

    def forward(self, x):
        return self.aug.forward_no_learn(x)


class ALIXEncoder1(nn.Module):
    '''Encoder with the same regularization applied after every layer, and with the
       regularization parameter tuned only with the final layer's feature gradients.'''

    def __init__(self, state_dim, aug):
        nn.Module.__init__(self)
        self.aug = aug
        assert len(state_dim) == 3

        # self.repr_dim = 32 * 35 * 35
        self.convnet = nn.Sequential(nn.Conv2d(state_dim[-1], 32, 3, stride=2),
                                     nn.ReLU(),
                                     self.aug,
                                     # NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(32, 32, 3, stride=2),
                                     nn.ReLU(),
                                     # NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(),
                                     # NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        # self.aug)

        # height = state_dim[0]
        # width = state_dim[1]
        # for _ in range(len(channels) - 1):
        #     height = int(((height - 4) + 1) / 2)
        #     width = int(((width - 4) + 1) / 2)
        self.feature_dim = 32 * 30 * 30

    def forward(self, state):
        """

        :param state: shape=(batch_size, 3, height, width)
        :return:
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(np.moveaxis(state, -1, -3)).unsqueeze(0).to(self.device)
        x = self.convnet(state)
        return x


class StackPolicyNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim, init_w=3e-3, log_std_min=-50, log_std_max=10):
        super(StackPolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # self.encoder = Encoder(state_dim)

        self.linear1 = nn.Linear(feature_dim + action_dim + 1, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.linear3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, feature, last_action):
        # if detach_encoder:
        #     with torch.no_grad():
        #         feature = self.encoder(obs)
        #         feature = feature.reshape((feature.shape[0], -1))
        #         last_feature = self.encoder(last_obs)
        #         last_feature = feature.reshape((last_feature.shape[0], -1))
        # else:
        #     feature = self.encoder(obs)
        #     feature = feature.reshape((feature.shape[0], -1))
        #     last_feature = self.encoder(last_obs)
        #     last_feature = feature.reshape((last_feature.shape[0], -1))
        # print(feature.mean())
        feature = feature.reshape((feature.shape[0], -1))
        # last_feature = last_feature.reshape((last_feature.shape[0], -1))
        last_action = nn.functional.one_hot(last_action.type(torch.int64).squeeze(-1), num_classes=11)
        q_input = torch.cat([feature, last_action], dim=1)
        x = F.relu(self.linear1(q_input))
        x = F.relu(self.linear2(x))
        x = F.softmax(self.linear3(x), dim=-1)

        return x

    def evaluate(self, feature, last_action, epsilon=1e-10):
        prob = self.forward(feature, last_action)
        dist = Categorical(prob)
        action = dist.sample().view(-1, 1)
        z = (prob == 0.0).float() * epsilon
        log_prob = torch.log(prob + z)

        return action, prob, log_prob

    def get_action(self, feature, last_action):
        # if isinstance(obs, np.ndarray):
        #     obs = torch.FloatTensor(np.moveaxis(obs, -1, -3)).unsqueeze(0).to(device)
        # if isinstance(last_obs, np.ndarray):
        #     last_obs = torch.FloatTensor(np.moveaxis(last_obs, -1, -3)).unsqueeze(0).to(device)
        if isinstance(last_action, np.ndarray):
            last_action = torch.FloatTensor(last_action).unsqueeze(0).to(device)
        prob = self.forward(feature, last_action)
        dist = Categorical(prob)
        action = dist.sample().view(-1, 1)
        action = action.detach().cpu().numpy()
        return action[0]


class SoftQNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim, init_w=3e-3):
        """

        :param state_dim:
        :param action_dim:
        :param hidden_dim:
        :param init_w:
        """
        super(SoftQNetwork, self).__init__()
        self.linear1 = nn.Linear(feature_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, feature):
        x = F.relu(self.linear1(feature))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class StackMultiQNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim, num=2, init_w=3e-3):
        super(StackMultiQNetwork, self).__init__()

        self.Qs = nn.ModuleList(
            [SoftQNetwork(feature_dim + action_dim + 1, action_dim, hidden_dim, init_w).to(device) for _ in
             range(num)])

    def forward(self, feature, last_action):
        feature = feature.reshape((feature.shape[0], -1))
        # last_feature = last_feature.reshape((last_feature.shape[0], -1))
        last_action = nn.functional.one_hot(last_action.type(torch.int64).squeeze(-1), num_classes=11)
        q_input = torch.cat([feature, last_action], dim=1)
        out = [q_net(q_input) for q_net in self.Qs]
        return torch.stack(out, dim=0)

    def qLoss(self, target, feature, last_action, action, criterion):
        loss = 0
        action = action.squeeze(-1).long()
        feature = feature.reshape((feature.shape[0], -1))
        last_action = nn.functional.one_hot(last_action.type(torch.int64).squeeze(-1), num_classes=11)
        q_input = torch.cat([feature, last_action], dim=1)
        assert action.shape[0] == feature.shape[0]
        for q_net in self.Qs:
            # for i in range(action.shape[0]):

            #     loss1 += criterion(q_net(feature[i])[action[i]], target[i])
            loss += criterion(q_net(q_input)[torch.tensor(range(action.shape[0])), action], target.view(-1))
        return loss


def l2_projection(constraint):
    @torch.no_grad()
    def fn(module):
        if hasattr(module, 'weight') and constraint > 0:
            w = module.weight
            norm = torch.norm(w)
            w.mul_(torch.clip(constraint / norm, max=1))

    return fn


def custom_parameterized_aug_optimizer_builder(encoder, encoder_lr, **kwargs):
    """Apply different optimizer parameters for S"""
    # assert isinstance(encoder, AllFeatTiedRegularizedEncoder)
    assert isinstance(encoder.aug, ParameterizedReg)
    encoder_params = list(encoder.parameters())
    encoder_aug_parameters = list(encoder.aug.parameters())
    encoder_non_aug_parameters = [p for p in encoder_params if
                                  all([p is not aug_p for aug_p in
                                       encoder_aug_parameters])]
    return torch.optim.Adam([{'params': encoder_non_aug_parameters},
                          {'params': encoder_aug_parameters, **kwargs}],
                         lr=encoder_lr)


@retry(stop_max_attempt_number=5, wait_fixed=2000)
def get_remote_class(name: str, namespace: str):
    """
    Try to get ray Remote Class
    :param name:
    :param namespace:
    :return:
    """
    remote_class = ray.get_actor(name=name, namespace=namespace)
    return remote_class


@retry(stop_max_attempt_number=500, wait_fixed=2000)
def check_transition_num(dataset, batch):
    if ray.get(dataset.__len__.remote()) < batch:
        raise Exception(f'There are less than {batch} transitions in the dataset after 1000 seconds, please check the '
                        f'actor!')
    else:
        return


@ray.remote(num_cpus=2, num_gpus=1)
class remote_trainer:
    def __init__(self, load_path, action_dim=10, soft_q_lr=1e-3, policy_lr=3e-4,
                 alpha_lr=1e-4, encoder_lr=1e-3, soft_tau=5e-3, stack=True):
        super().__init__()
        with open(os.path.join(os.path.dirname(load_path), os.pardir, 'args.json'), 'r') as f:
            args = f.read()
            args = json.loads(args)
            args = argparse.Namespace(**args)
        if stack:
            channals = 4
        else:
            channals = 2
        aug = ParameterizedReg(aug=LocalSignalMixing(pad=2, fixed_batch=True),
                               parameter_init=0.5, param_grad_fn='alix_param_grad',
                               param_grad_fn_args=[3, 0.535, 1e-20])
        if 'encoder' in args:
            self.encoder = eval(args.encoder)((140, 140, channals), aug=aug).to(device)
        else:
            self.encoder = ALIXEncoder1((140, 140, channals), aug=aug).to(device)
        self.feature_dim = self.encoder.feature_dim
        self.actor = StackPolicyNetwork(feature_dim=self.feature_dim,
                                        action_dim=len(action_set),
                                        hidden_dim=args.actor_hidden_dim).to(device)
        self.soft_q_nets = StackMultiQNetwork(self.feature_dim, action_dim, args.critic_hidden_dim).to(device)
        self.target_soft_q_nets = StackMultiQNetwork(self.feature_dim, action_dim, args.critic_hidden_dim).to(device)
        for target_param, param in zip(self.target_soft_q_nets.parameters(), self.soft_q_nets.parameters()):
            target_param.data.copy_(param.data)
        self.log_alpha = torch.tensor(args.log_alpha).to(device)
        self.log_alpha.requires_grad = True

        self.target_entropy = 0.98 * np.log(action_dim) if args.target_entropy is None else args.target_entropy
        self.gamma = args.gamma
        self.soft_q_criterion = nn.MSELoss()
        self.soft_tau = soft_tau
        self.w = args.w
        self.beta = args.beta
        self.alpha_lr = alpha_lr
        self.step = 0

        if load_path is not None:
            self.load_paras(load_path)

        print(f'successfully load offline model "{load_path}"!')

        self.critic_optimizer = optim.Adam(self.soft_q_nets.parameters(), lr=soft_q_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.encoder_optimizer = custom_parameterized_aug_optimizer_builder(encoder=self.encoder,
                                                                            encoder_lr=encoder_lr,
                                                                            lr=2e-3, betas=[0.5, 0.999])

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_actor_paras(self):
        return self.step, list(para.cpu() for para in self.actor.parameters())

    def get_actor_and_encoder_paras(self):
        return self.step, list(para.cpu() for para in self.actor.parameters()), list(
            para.cpu() for para in self.encoder.parameters())

    def get_critic_paras(self):
        return self.step, list(para.cpu() for para in self.actor.parameters())

    def critic_loss(self, last_obs, obs, last_action, action, reward, next_obs, done):

        batch_size = last_obs.shape[0]

        feature = self.encoder(torch.cat([last_obs, obs], dim=1))
        next_feature = self.encoder(torch.cat([obs, next_obs], dim=1)).detach()

        # update critic
        with torch.no_grad():
            next_action, next_prob, next_log_prob = self.actor.evaluate(next_feature, action)
            target_next_q = self.target_soft_q_nets(next_feature, action).min(dim=0)[0]
            target_next_v = torch.einsum('ij,ij->i', target_next_q, next_prob)
            target_bellman_q = reward + (1 - done) * self.gamma * target_next_v.unsqueeze(1)
            target_bellman_q = target_bellman_q.view(-1)
            assert len(target_bellman_q.shape) == 1 and target_bellman_q.shape[0] == batch_size
        next_q = self.soft_q_nets(next_feature, action).min(dim=0)[0]
        next_v = torch.einsum('ij,ij->i', next_q, next_prob)
        bellman_q = reward + (1 - done) * self.gamma * next_v.unsqueeze(1)
        bellman_q = bellman_q.view(-1)  # shape=(batch_size)
        assert len(bellman_q.shape) == 1 and bellman_q.shape[0] == batch_size
        q = self.soft_q_nets(feature, last_action)
        q_1 = q[0][torch.tensor(range(action.shape[0])), action.squeeze(-1).long()]
        q_2 = q[1][torch.tensor(range(action.shape[0])), action.squeeze(-1).long()]
        assert q_1.shape == target_bellman_q.shape and q_1.shape == q_2.shape and q_1.shape == bellman_q.shape
        target_bellman_loss = (q_1 - target_bellman_q) ** 2 + (q_2 - target_bellman_q) ** 2
        bellman_loss = (q_1 - bellman_q) ** 2 + (q_2 - bellman_q) ** 2
        # target_bellman_loss = self.soft_q_nets.qLoss(target_bellman_q, last_feature, feature, last_action, action,
        #                                              self.soft_q_criterion)
        # bellman_loss = self.soft_q_nets.qLoss(bellman_q, last_feature, feature, last_action, action,
        #                                       self.soft_q_criterion)
        bellman_loss_sum = self.w * target_bellman_loss + (1 - self.w) * bellman_loss

        return bellman_loss_sum, target_bellman_loss, bellman_loss

    def actor_loss(self, last_obs, obs, last_action):
        feature = self.encoder(torch.cat([last_obs, obs], dim=1))

        _, prob, log_prob = self.actor.evaluate(feature.detach(), last_action)
        with torch.no_grad():
            expected_q_value = torch.min(self.soft_q_nets(feature, last_action), dim=0)[0]
            expected_q_value = expected_q_value - torch.mean(expected_q_value, dim=-1, keepdim=True) \
                .repeat(1, expected_q_value.shape[1])  # using advantage function for normalization
        expected_q_value = self.alpha.detach() * log_prob - expected_q_value
        actor_loss = torch.einsum('ij,ij->i', expected_q_value, prob)
        # actor_loss = actor_loss.mean()

        with torch.no_grad():
            # new_action, prob, log_prob = self.actor.evaluate(feature)
            entropy = -torch.einsum('ij,ij->i', log_prob, prob)  # .mean()

        alpha_loss = self.log_alpha.exp() * (entropy - self.target_entropy)
        return actor_loss, alpha_loss, entropy

    def transfer2tensor(self, *paras):
        # out = []
        for para in paras:
            if not isinstance(para, torch.Tensor):
                if len(para.shape) == 4:  # adjust axis for the image as np.ndarray
                    para = np.moveaxis(para, -1, -3)
                elif len(para.shape) == 1:  # add axis for action, reward and done to make the shape as (batch_size, 1)
                    para = np.expand_dims(para, axis=-1)
                para = torch.FloatTensor(para.copy()).to(device)
            yield para
            # out.append(para)
        # return out

    def train_step(self,
                   last_obs, obs, last_action, action, reward, next_obs, done,
                   mean_lambda=1e-3,
                   std_lambda=1e-3,
                   z_lambda=0.0,
                   gradient_steps=1
                   ):
        # state 原始形状 [140, 140, 3]
        # 注意这里的维度设定！！！！卷积层的输入为(N,C_in,H,W) [256, 2, 130, 130]
        last_obs, obs, last_action, action, reward, next_obs, done = self.transfer2tensor(last_obs, obs,
                                                                                          last_action, action,
                                                                                          reward, next_obs, done)
        for _ in range(gradient_steps):
            bellman_loss_sum, target_bellman_loss, bellman_loss = self.critic_loss(last_obs, obs, last_action, action,
                                                                                   reward, next_obs, done)

            actor_loss, alpha_loss, entropy = self.actor_loss(last_obs, obs, last_action)

            loss = bellman_loss_sum.mean() + actor_loss.mean() + alpha_loss.mean()
            self.critic_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            self.alpha_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()
            self.encoder_optimizer.step()
            self.actor_optimizer.step()
            self.alpha_optimizer.step()

            for target_param, param in zip(self.target_soft_q_nets.parameters(), self.soft_q_nets.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
                )

            return bellman_loss_sum.mean(), actor_loss.mean().item(), alpha_loss.mean().item(), self.alpha.item()

    def save_paras(self, path):
        torch.save({'actor': self.actor.state_dict(),
                    'critic': self.soft_q_nets.state_dict(),
                    'target_critic': self.target_soft_q_nets.state_dict(),
                    'encoder': self.encoder.state_dict(),
                    'log_alpha': self.log_alpha},
                   path)
        return

    def load_paras(self, path):
        payload = torch.load(path)
        self.actor.load_state_dict(payload['actor'])
        self.soft_q_nets.load_state_dict(payload['critic'])
        self.target_soft_q_nets.load_state_dict(payload['target_critic'])
        self.log_alpha = payload['log_alpha']
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
        self.encoder.load_state_dict(payload['encoder'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ray arguments
    parser.add_argument("--address", type=str, help="the address of ray head node", default='ray://172.18.22.9:10001')
    parser.add_argument("--namespace", type=str, help="the name of node", default='learner')
    parser.add_argument("--name", type=str, help="the name of learner", default='learner-node')
    parser.add_argument("--dataset-namespace", type=str, help="the namespace of dataset node", default='dataset')
    parser.add_argument("--dataset-name", type=str, help="the name of replay buffer", default='dataset-node')

    # RL arguments
    parser.add_argument("--save-path", type=str, help="the dir to save log and model", default='SAC-experiment-0404')
    parser.add_argument("--log-path", type=str, help="the log file", default='log.csv')
    parser.add_argument("--step", type=float, help="steps of training", default=5e6)
    # parser.add_argument("--hidden-size", type=int, help="number of hidden units per layer", default=512)
    parser.add_argument("--batch-size", type=int, help="number of samples per minibatch", default=256)
    parser.add_argument("--soft-tau", type=float, help="target smoothing coefficient", default=5e-3)
    # parser.add_argument("--gamma", type=float, help="gamma of reward", default=0.99)
    # parser.add_argument("--alpha", type=float, help="init alpha", default=1.0)
    parser.add_argument("--gradient-steps", type=int, help="the num of gradient step per SAC update", default=1)
    parser.add_argument("--encoder-lr", type=float, help="the learning rate of encoder", default=1e-3)
    parser.add_argument("--soft-q-lr", type=float, help="the learning rate of soft Q function", default=1e-3)
    parser.add_argument("--policy-lr", type=float, help="the learning rate of policy", default=1e-4)
    parser.add_argument("--alpha-lr", type=float, help="the learning rate of alpha", default=1e-5)
    parser.add_argument("--max-learn-act-ratio", type=float, help="the max ratio of learn step to act step", default=4)
    parser.add_argument("--load-path", type=str, help="the log file", default='/devdata1/lhdata/off+online/0130-20000-3/300k/300k.pt')

    args = parser.parse_args()
    print('------  ' + args.namespace + ' node connect to ray cluster  ------')
    # ray.init(address=args.address, namespace=args.namespace)
    ray.init(address='auto', namespace=args.namespace, _redis_password='5241590000000000')

    print('------  learner initialize  ------')
    learner = remote_trainer.options(name=args.name).remote(load_path=args.load_path,
                                                            action_dim=10,
                                                            encoder_lr=args.encoder_lr,
                                                            soft_q_lr=args.soft_q_lr,
                                                            policy_lr=args.policy_lr,
                                                            alpha_lr=args.alpha_lr,
                                                            soft_tau=args.soft_tau)

    time.sleep(5)

    print('------  connect to dataset  ------')
    dataset = get_remote_class(name=args.dataset_name, namespace=args.dataset_namespace)
    assert ray.get(dataset.type.remote()) == 'online'
    time.sleep(20)
    check_transition_num(dataset, args.batch_size)

    print('------  start training  ------')
    # 不能把训练写成一个函数，因为这样会导致整个训练过程是一个进程，阻碍了actor与learner的通信
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    log_path = os.path.join(args.save_path, args.log_path)
    with open(log_path, 'w') as f:
        f.write('step,critic_loss,actor_loss,alpha_loss,alpha_value\n')
    with open(os.path.join(args.save_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    model_path = os.path.join(args.save_path, 'model')
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    for step in range(int(args.step)):
        l_a_r = ray.get(dataset.learn_act_ratio.remote())
        # print(step, l_a_r)
        while l_a_r > args.max_learn_act_ratio:
            time.sleep(1)
            l_a_r = ray.get(dataset.learn_act_ratio.remote())
        last_obs, obs, last_action, action, reward, next_obs, done = ray.get(dataset.sample.remote(args.batch_size))
        critic_loss, actor_loss, alpha_loss, alpha_value = ray.get(
            learner.train_step.remote(last_obs, obs, last_action, action, reward, next_obs, done))
        # print(f'{step}  critic_loss: %.2f    actor_loss: %.2f    alpha_loss: %.2f    decoder_loss: %.2f' % (critic_loss, actor_loss, alpha_loss, decoder_loss))

        if step % 1000 == 0:
            with open(log_path, 'a') as f:
                f.write('%d,%.4f,%.4f,%.4f,%.4f\n' % (
                    step, critic_loss, actor_loss, alpha_loss, alpha_value))
            ray.get(learner.save_paras.remote(os.path.join(model_path, f'{int(step / 1000)}k_SAC.pkl')))

            print('node learner: ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) +
                  'step: %dk   critic_loss: %.2f    actor_loss: %.2f    alpha_loss: %.2f    alpha_value: %.2f' %
                  (step / 1000, critic_loss, actor_loss, alpha_loss, alpha_value))
