import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
import math


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


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


class MultiQNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim, num=2, init_w=3e-3):
        super(MultiQNetwork, self).__init__()
        # self.encoder = Encoder(state_dim)

        self.Qs = nn.ModuleList(
            [SoftQNetwork(feature_dim, action_dim, hidden_dim, init_w).to(device) for _ in range(num)])

    def forward(self, feature):
        # feature = self.encoder(state)
        feature = feature.reshape((feature.shape[0], -1))
        out = [q_net(feature) for q_net in self.Qs]
        return torch.stack(out, dim=0)

    def qLoss(self, target, feature, action, criterion):
        loss = 0
        action = action.view(-1).long()
        # feature = self.encoder(state)
        feature = feature.reshape((feature.shape[0], -1))
        assert action.shape[0] == feature.shape[0]
        for q_net in self.Qs:
            # for i in range(action.shape[0]):
            #     loss1 += criterion(q_net(feature[i])[action[i]], target[i])
            loss += criterion(q_net(feature)[torch.tensor(range(action.shape[0])), action], target.view(-1))
        return loss

    # def parameters(self, recurse: bool = True):
    #     p = []
    #     for q_net in self.Qs:
    #         p += q_net.parameters()
    #     return p

    def getValue(self, feature, dist):
        """

        :param state: shape = [256, 2, 130, 130]
        :param dist: shape = [256, 25]
        :return:
        """
        # feature = self.encoder(state)
        # feature = feature.reshape((feature.shape[0], -1))  # shape = [256, feature_dim]
        feature = feature.reshape((feature.shape[0], -1))
        q1 = self.Qs[0](feature)
        q2 = self.Qs[1](feature)
        q = torch.min(q1, q2)
        v = torch.tensor([q[i] * dist[i] for i in range(dist.shape[-1])]).to(device)
        print(v.shape)
        return v


class PolicyNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim, init_w=3e-3, log_std_min=-50, log_std_max=10):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # self.encoder = Encoder(state_dim)

        self.linear1 = nn.Linear(feature_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.linear3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, feature, detach_encoder=False):
        # if detach_encoder:
        #     with torch.no_grad():
        #         feature = self.encoder(obs)
        #         feature = feature.reshape((feature.shape[0], -1))
        # else:
        #     feature = self.encoder(obs)
        #     feature = feature.reshape((feature.shape[0], -1))
        # print(feature.mean())
        feature = feature.reshape((feature.shape[0], -1))
        x = F.relu(self.linear1(feature))
        x = F.relu(self.linear2(x))
        x = F.softmax(self.linear3(x), dim=-1)

        return x

    def evaluate(self, feature, detach_encoder=False, epsilon=1e-10):
        prob = self.forward(feature, detach_encoder)
        dist = Categorical(prob)
        action = dist.sample().view(-1, 1)
        z = (prob == 0.0).float() * epsilon
        log_prob = torch.log(prob + z)

        return action, prob, log_prob

    def get_action(self, feature):
        # if isinstance(obs, np.ndarray):
        #     obs = torch.FloatTensor(np.moveaxis(obs, -1, -3)).unsqueeze(0).to(device)
        prob = self.forward(feature)
        dist = Categorical(prob)
        action = dist.sample().view(-1, 1)
        action = action.detach().cpu().numpy()
        return action[0]


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
        # assert torch.isnan(prob).any() == False
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
        return action


class Encoder(nn.Module):
    '''Encoder with the same regularization applied after every layer, and with the
       regularization parameter tuned only with the final layer's feature gradients.'''

    def __init__(self, state_dim=(140, 140, 3)):
        nn.Module.__init__(self)
        assert len(state_dim) == 3

        # self.repr_dim = 32 * 35 * 35
        self.convnet = nn.Sequential(nn.Conv2d(state_dim[-1], 16, kernel_size=(5, 5), stride=(2, 2)),
                                     nn.ReLU(),
                                     # NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2)),
                                     nn.ReLU(),
                                     # NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2)),
                                     nn.ReLU(),
                                     # NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(64, 64, kernel_size=(5, 5), stride=(2, 2)),
                                     nn.ReLU())

        with torch.no_grad():
            input_tensor = torch.randn(1, state_dim[2], state_dim[0], state_dim[1])
            output_tensor = self.convnet(input_tensor)
        self.feature_shape = tuple(output_tensor.shape[1:])
        self.feature_dim = np.prod(self.feature_shape)

    def forward(self, state):
        """

        :param state: shape=(batch_size, 3, height, width)
        :return:
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(np.moveaxis(state, -1, -3)).unsqueeze(0).to(self.device)
        x = self.convnet(state)
        return x

class StackPolicyNetworkWithEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, init_w=3e-3, log_std_min=-50, log_std_max=10):
        super(StackPolicyNetworkWithEncoder, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.encoder = Encoder(state_dim)

        self.linear1 = nn.Linear(self.encoder.feature_dim + action_dim + 1, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.linear3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, last_action, detach_encoder=False):
        if detach_encoder:
            with torch.no_grad():
                feature = self.encoder(state)
        else:
            feature = self.encoder(state)
        # print(feature.mean())
        feature = feature.reshape((feature.shape[0], -1))
        last_action = nn.functional.one_hot(last_action.type(torch.int64).squeeze(-1), num_classes=11)
        q_input = torch.cat([feature, last_action], dim=1)
        x = F.relu(self.linear1(q_input))
        x = F.relu(self.linear2(x))
        x = F.softmax(self.linear3(x), dim=-1)

        return x

    def evaluate(self, state, last_action, epsilon=1e-10):
        prob = self.forward(state, last_action)
        # assert torch.isnan(prob).any() == False
        dist = Categorical(prob)
        action = dist.sample().view(-1, 1)
        z = (prob == 0.0).float() * epsilon
        log_prob = torch.log(prob + z)

        return action, prob, log_prob

    def get_action(self, state, last_action):
        # if isinstance(obs, np.ndarray):
        #     obs = torch.FloatTensor(np.moveaxis(obs, -1, -3)).unsqueeze(0).to(device)
        # if isinstance(last_obs, np.ndarray):
        #     last_obs = torch.FloatTensor(np.moveaxis(last_obs, -1, -3)).unsqueeze(0).to(device)
        if isinstance(last_action, np.ndarray):
            last_action = torch.FloatTensor(last_action).unsqueeze(0).to(device)
        prob = self.forward(state, last_action)
        dist = Categorical(prob)
        action = dist.sample().view(-1, 1)
        action = action.detach().cpu().numpy()
        return action


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


class SoftQNetworkWithEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, init_w=3e-3):
        """

        :param state_dim:
        :param action_dim:
        :param hidden_dim:
        :param init_w:
        """
        super(SoftQNetworkWithEncoder, self).__init__()

        self.encoder = Encoder(state_dim)

        self.linear1 = nn.Linear(self.encoder.feature_dim + action_dim + 1, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, last_action, detach_encoder=True):
        if detach_encoder:
            with torch.no_grad():
                feature = self.encoder(state)
        else:
            feature = self.encoder(state)
        # print(feature.mean())
        feature = feature.reshape((feature.shape[0], -1))
        last_action = nn.functional.one_hot(last_action.type(torch.int64).squeeze(-1), num_classes=11)
        q_input = torch.cat([feature, last_action], dim=1)
        x = F.relu(self.linear1(q_input))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class StackMultiQNetworkWithEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num=2, init_w=3e-3):
        super(StackMultiQNetworkWithEncoder, self).__init__()

        self.Qs = nn.ModuleList(
            [SoftQNetworkWithEncoder(state_dim, action_dim, hidden_dim, init_w).to(device) for _ in
             range(num)])

    def forward(self, state, last_action):
        out = [q_net(state, last_action) for q_net in self.Qs]
        return torch.stack(out, dim=0)

    def qLoss(self, target, state, last_action, action, criterion):
        loss = 0
        action = action.squeeze(-1).long()
        for q_net in self.Qs:
            # for i in range(action.shape[0]):

            #     loss1 += criterion(q_net(feature[i])[action[i]], target[i])
            loss += criterion(q_net(state, last_action)[torch.tensor(range(action.shape[0])), action], target.view(-1))
        return loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def l2_projection(constraint):
    @torch.no_grad()
    def fn(module):
        if hasattr(module, 'weight') and constraint > 0:
            w = module.weight
            norm = torch.norm(w)
            w.mul_(torch.clip(constraint / norm, max=1))

    return fn


# augment layers from A-LIX, https://github.com/Aladoro/Stabilizing-Off-Policy-RL

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


class ALIXEncoder(nn.Module):
    '''Encoder with the same regularization applied after every layer, and with the
       regularization parameter tuned only with the final layer's feature gradients.'''

    def __init__(self, state_dim, aug):
        nn.Module.__init__(self)
        self.aug = aug
        assert len(state_dim) == 3

        # self.repr_dim = 32 * 35 * 35
        self.convnet = nn.Sequential(nn.Conv2d(state_dim[-1], 32, 3, stride=2),
                                     nn.ReLU(),
                                     NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(32, 32, 3, stride=2),
                                     nn.ReLU(),
                                     NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(),
                                     NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(),
                                     self.aug)

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


class ALIXEncoder2(nn.Module):
    '''Encoder with the same regularization applied after every layer, and with the
       regularization parameter tuned only with the final layer's feature gradients.'''

    def __init__(self, state_dim, aug):
        nn.Module.__init__(self)
        self.aug = aug
        assert len(state_dim) == 3

        # self.repr_dim = 32 * 35 * 35
        self.convnet = nn.Sequential(nn.Conv2d(state_dim[-1], 32, 3, stride=1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2),
                                     self.aug,
                                     # NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2),
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
        self.feature_dim = 32 * 29 * 29

    def forward(self, state):
        """

        :param state: shape=(batch_size, 3, height, width)
        :return:
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(np.moveaxis(state, -1, -3)).unsqueeze(0).to(self.device)
        x = self.convnet(state)
        return x


class ALIXEncoder3(nn.Module):
    '''Encoder with the same regularization applied after every layer, and with the
       regularization parameter tuned only with the final layer's feature gradients.'''

    def __init__(self, state_dim, aug):
        nn.Module.__init__(self)
        self.aug = aug
        assert len(state_dim) == 3

        # self.repr_dim = 32 * 35 * 35
        self.convnet = nn.Sequential(nn.Conv2d(state_dim[-1], 32, 3, stride=2),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=2),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

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


class ALIXEncoder4(nn.Module):
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
                                     self.aug,
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


# pretrained autoencoder as feature exactor
class autoencoderExactor(nn.Module):
    def __init__(self, obs_shape, feature_dim, linear_hidden_dim, device):
        super(autoencoderExactor, self).__init__()
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.device = device
        aug = ParameterizedReg(aug=LocalSignalMixing(pad=2, fixed_batch=True), parameter_init=0.5,
                               param_grad_fn='alix_param_grad', param_grad_fn_args=[3, 0.535, 1e-20])
        self.encoder = ALIXEncoder1(obs_shape, aug=aug)
        self.linear1 = nn.Sequential(nn.Linear(self.encoder.feature_dim, linear_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(linear_hidden_dim, linear_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(linear_hidden_dim, feature_dim)
                                     )

        self.linear2 = nn.Sequential(nn.Linear(feature_dim, linear_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(linear_hidden_dim, linear_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(linear_hidden_dim, self.encoder.feature_dim)
                                     )
        self.decoder = nn.Sequential(nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(1, 1)),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(1, 1)),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2)),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(32, obs_shape[-1], kernel_size=(3, 3), stride=(2, 2)),
                                     nn.Sigmoid()
                                     )

        self.to(device)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(np.moveaxis(obs, -1, -3)).unsqueeze(0).to(self.device)
        encoder_out = self.encoder(obs)
        linear1_in = encoder_out.reshape((encoder_out.shape[0], -1))
        feature = self.linear1(linear1_in)
        linear2_out = self.linear2(feature)
        decoder_out = self.decoder(linear2_out.reshape(encoder_out.shape))
        return decoder_out

    def exact_feature(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(np.moveaxis(obs, -1, -3)).unsqueeze(0).to(self.device)
        encoder_out = self.encoder(obs)
        linear1_in = encoder_out.reshape((encoder_out.shape[0], -1))
        feature = self.linear1(linear1_in)
        return feature


class encoderExactor(nn.Module):
    def __init__(self, obs_shape, feature_dim, linear_hidden_dim):
        super(encoderExactor, self).__init__()
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        # self.device = device
        aug = ParameterizedReg(aug=LocalSignalMixing(pad=2, fixed_batch=True), parameter_init=0.5,
                               param_grad_fn='alix_param_grad', param_grad_fn_args=[3, 0.535, 1e-20])
        self.encoder = ALIXEncoder1(obs_shape, aug=aug)
        self.linear = nn.Sequential(nn.Linear(self.encoder.feature_dim, linear_hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(linear_hidden_dim, linear_hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(linear_hidden_dim, feature_dim)
                                    )
        assert self.encoder.feature_dim == 32 * 30 * 30, f'{self.encoder.feature_dim}'

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(np.moveaxis(obs, -1, -3)).to(self.device)
            if len(obs.shape) == 3:
                obs = obs.unsqueeze(0)
        encoder_out = self.encoder(obs)
        linear1_in = encoder_out.reshape((encoder_out.shape[0], -1))
        feature = self.linear(linear1_in)
        return feature


class decoderExactor(nn.Module):
    def __init__(self, obs_shape, feature_dim, linear_hidden_dim):
        super(decoderExactor, self).__init__()
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        # self.to(device)
        self.linear = nn.Sequential(nn.Linear(feature_dim, linear_hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(linear_hidden_dim, linear_hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(linear_hidden_dim, 32 * 30 * 30)
                                    )
        self.decoder = nn.Sequential(nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(1, 1)),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(1, 1)),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2)),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(32, obs_shape[-1], kernel_size=(3, 3), stride=(2, 2)),
                                     nn.Sigmoid()
                                     )

    def forward(self, feature):
        linear_out = self.linear(feature)
        decoder_out = self.decoder(linear_out.reshape((feature.shape[0], 32, 30, 30)))
        return decoder_out
