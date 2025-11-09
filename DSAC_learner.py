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

class encoder(nn.Module):
    def __init__(self, state_dim):
        super(encoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(state_dim[-1], 16, kernel_size=(5, 5), stride=(2, 2)),
                                   # nn.BatchNorm2d(16),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2)),
                                   # nn.BatchNorm2d(32),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2)),
                                   # nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(5, 5), stride=(2, 2)),
                                   # nn.BatchNorm2d(64),
                                   nn.ReLU())

    def forward(self, state):
        """

        :param state: shape=(batch_size, 3, height, width)
        :return:
        """
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class decoder(nn.Module):
    def __init__(self, state_dim):
        super(decoder, self).__init__()
        self.convT = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=(5, 5), stride=(2, 2)),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(64, 32, kernel_size=(5, 5), stride=(2, 2)),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(32, 16, kernel_size=(5, 5), stride=(2, 2)),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(16, 1, kernel_size=(5, 5), stride=(2, 2)),
                                    nn.Sigmoid()
                                    )

    def forward(self, state):
        x = self.convT(state)
        return x


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
    def __init__(self, state_dim, action_dim, hidden_dim, num=2, init_w=3e-3):
        super(MultiQNetwork, self).__init__()
        self.encoder = encoder(state_dim)

        height = state_dim[0]
        width = state_dim[1]
        for _ in range(4):
            height = int(((height - 4) + 1) / 2)
            width = int(((width - 4) + 1) / 2)
        self.feature_dim = height*width*64

        self.Qs = [SoftQNetwork(self.feature_dim, action_dim, hidden_dim, init_w).to(device) for _ in range(num)]

    def forward(self, state):
        feature = self.encoder(state)
        feature = feature.reshape((feature.shape[0], -1))
        out = [q_net(feature) for q_net in self.Qs]
        return torch.stack(out, dim=0)

    def qLoss(self, target, state, action, criterion):
        loss = 0
        action = action.view(-1).long()
        feature = self.encoder(state)
        feature = feature.reshape((feature.shape[0], -1))
        assert action.shape[0] == feature.shape[0]
        for q_net in self.Qs:
            # for i in range(action.shape[0]):
            #     loss1 += criterion(q_net(feature[i])[action[i]], target[i])
            loss += criterion(q_net(feature)[torch.tensor(range(action.shape[0])), action], target.view(-1))
        return loss

    def parameters(self, recurse: bool = True):
        p = []
        # p += self.encoder.parameters()
        for q_net in self.Qs:
            p += q_net.parameters()
        return p

    def getValue(self, state, dist):
        """

        :param state: shape = [256, 2, 130, 130]
        :param dist: shape = [256, 25]
        :return:
        """
        feature = self.encoder(state)
        feature = feature.reshape((feature.shape[0], -1))  # shape = [256, feature_dim]
        q1 = self.Qs[0](feature)
        q2 = self.Qs[1](feature)
        q = torch.min(q1, q2)
        v = torch.tensor([q[i] * dist[i] for i in range(dist.shape[-1])]).to(device)
        print(v.shape)
        return v


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, init_w=3e-3, log_std_min=-50, log_std_max=10):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.encoder = encoder(state_dim)

        height = state_dim[0]
        width = state_dim[1]
        for _ in range(4):
            height = int(((height - 4) + 1) / 2)
            width = int(((width - 4) + 1) / 2)
        self.feature_dim = height * width * 64

        self.linear1 = nn.Linear(self.feature_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.linear3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        feature = self.encoder(state)
        feature = feature.reshape((feature.shape[0], -1))
        # print(feature.mean())
        x = F.relu(self.linear1(feature))
        x = F.relu(self.linear2(x))
        x = F.softmax(self.linear3(x), dim=-1)

        return x

    def evaluate(self, state, epsilon=1e-6):
        prob = self.forward(state)
        dist = Categorical(prob)
        action = dist.sample().view(-1, 1)
        z = (prob == 0.0).float() * 1e-9
        log_prob = torch.log(prob + z)

        return action, prob, log_prob

    def get_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(np.moveaxis(state, -1, -3)).unsqueeze(0).to(device)
        prob = self.forward(state)
        dist = Categorical(prob)
        action = dist.sample().view(-1, 1)
        action = action.detach().cpu().numpy()
        return action[0]


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1, gamma: float = 2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs = inputs.view(-1, 1)
        # targets = targets.view

        focal_loss = torch.zeros_like(targets)
        mask = targets.bool()
        focal_loss[mask] = -(1-inputs[mask]) ** self.gamma * torch.log(inputs[mask]+1e-10)
        focal_loss[~mask] = -inputs[~mask] ** self.gamma * torch.log(1 - inputs[~mask]+1e-10)
        # focal_loss = -(1-inputs) ** self.gamma * torch.log(inputs) * targets \
        #              - inputs ** self.gamma * torch.log(1 - inputs) * (1 - targets)
        # print((inputs-targets).sum())
        focal_loss = focal_loss.mean()
        return focal_loss.mean()


class GHM_Loss(nn.Module):
    def __init__(self, bins=10, alpha=0.8):
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        g = torch.abs(self._custom_loss_grad(x, target)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = (x.size(0) * x.size(1) * x.size(2) * x.size(3))

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd

        return self._custom_loss(x, target, beta[bin_idx].to(device))


class GHMC_Loss(GHM_Loss):
    def __init__(self, bins=10, alpha=0.8):
        super(GHMC_Loss, self).__init__(bins, alpha)

    def _custom_loss(self, x, target, weight):
        return F.binary_cross_entropy_with_logits(x, target, weight=weight)

    def _custom_loss_grad(self, x, target):
        return torch.sigmoid(x).detach() - target


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


@retry(stop_max_attempt_number=250, wait_fixed=2000)
def check_transition_num(dataset, batch):
    if ray.get(dataset.__len__.remote()) < batch:
        raise Exception(f'There are less than {batch} transitions in the dataset after 500 seconds, please check the '
                        f'actor!')
    else:
        return


@ray.remote(num_cpus=2, num_gpus=1)
class remote_sac_trainer(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, gamma=0.99, alpha=1.0, soft_q_lr=1e-3, policy_lr=3e-4,
                 decoder_lr=1e-3, encoder_lr=1e-3, soft_tau=5e-3, load_path=None):
        super().__init__()
        self.soft_q_nets = MultiQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_nets = MultiQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.alpha = torch.tensor(alpha, requires_grad=True)
        self.decoder = decoder(state_dim).to(device)

        if load_path is not None:
            self.load_paras(load_path)
        else:
            for target_param, param in zip(self.target_soft_q_nets.parameters(), self.soft_q_nets.parameters()):
                target_param.data.copy_(param.data)

            for target_param, param in zip(self.actor.encoder.parameters(), self.soft_q_nets.encoder.parameters()):
                target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(self.soft_q_nets.parameters(), lr=soft_q_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.alpha], lr=policy_lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=decoder_lr)
        self.encoder_optimizer = optim.Adam(self.soft_q_nets.encoder.parameters(), lr=encoder_lr)

        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()
        self.decoder_loss = FocalLoss().to(device) #GHMC_Loss().to(device)

        self.target_entropy = 0.98*np.log(action_dim)
        self.step = 0
        self.gamma = gamma
        self.soft_tau = soft_tau

    def get_actor_paras(self):
        return self.step, list(para.cpu() for para in self.actor.parameters())

    def get_critic_paras(self):
        return self.step, list(para.cpu() for para in self.actor.parameters())

    def sac_update(self,
                   state, 
                   action, 
                   reward, 
                   next_state, 
                   done,
                   mean_lambda=1e-3,
                   std_lambda=1e-3,
                   z_lambda=0.0,
                   gradient_steps=1
                   ):
        # state 原始形状 [130, 130, 2]
        # 注意这里的维度设定！！！！卷积层的输入为(N,C_in,H,W) [256, 2, 130, 130]
        state = torch.FloatTensor(np.moveaxis(state.copy(), -1, -3)).to(device)
        next_state = torch.FloatTensor(np.moveaxis(next_state.copy(), -1, -3)).to(device)
        action = torch.FloatTensor(action.copy()).to(device)
        reward = torch.FloatTensor(reward.copy()).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done.copy())).unsqueeze(1).to(device)
        for _ in range(gradient_steps):
            # next_q_value = reward + (1 - done) * gamma * target_value
            with torch.no_grad():
                next_action, next_prob, next_log_prob = self.actor.evaluate(next_state)
                next_feature = self.target_soft_q_nets.encoder(next_state)
                next_feature = next_feature.reshape((next_feature.shape[0], -1))  # shape = [256, feature_dim]
                next_q1 = self.target_soft_q_nets.Qs[0](next_feature)
                next_q2 = self.target_soft_q_nets.Qs[1](next_feature)
                next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
                next_v_value = torch.tensor([next_q[i] @ next_prob[i] for i in range(next_prob.shape[0])]).to(device)
                # next_v_value = self.target_soft_q_nets.getValue(state, next_prob) - self.alpha * next_log_prob * next_prob
                target_q_value = reward + (1 - done) * self.gamma * next_v_value.unsqueeze(1)
            critic_loss = self.soft_q_nets.qLoss(target_q_value, state, action, self.soft_q_criterion)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            _, prob, log_prob = self.actor.evaluate(state)
            expected_q_value = torch.min(self.soft_q_nets(state), dim=0)[0]
            expected_q_value = self.alpha * log_prob - expected_q_value
            actor_loss = 0
            for i in range(prob.shape[0]):
                actor_loss += expected_q_value[i] @ prob[i]
            actor_loss /= prob.shape[0]

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            h = self.soft_q_nets.encoder(state)
            decoder_loss = self.decoder_loss(self.decoder(h), state[:, 1:, 7:-8, 7:-8])
            self.decoder_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            decoder_loss.backward()
            self.decoder_optimizer.step()
            self.encoder_optimizer.step()

            for target_param, param in zip(self.actor.encoder.parameters(), self.soft_q_nets.encoder.parameters()):
                target_param.data.copy_(param.data)

            for target_param, param in zip(self.target_soft_q_nets.parameters(), self.soft_q_nets.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
                )

            with torch.no_grad():
                new_action, prob, log_prob = self.actor.evaluate(state)
                log_prob = (log_prob + self.target_entropy)
                alpha_loss = torch.tensor([log_prob[i] @ prob[i] for i in range(prob.shape[0])]).to(device)
            alpha_loss = -self.alpha * alpha_loss.mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            return critic_loss.item(), actor_loss.item(), alpha_loss.item(), decoder_loss.item()

    def save_paras(self, path):
        torch.save({'actor': self.actor.state_dict(),
                    'critic': self.soft_q_nets.state_dict(),
                    'target_critic': self.target_soft_q_nets.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'alpha': self.alpha},
                   path)

        return

    def load_paras(self, path):
        payload = torch.load(path)
        self.actor.load_state_dict(payload['actor'])
        self.soft_q_nets.load_state_dict(payload['critic'])
        self.target_soft_q_nets.load_state_dict(payload['target_critic'])
        self.decoder.load_state_dict(payload['decoder'])
        self.alpha = payload['alpha']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ray arguments
    parser.add_argument("--address", type=str, help="the address of ray head node", default='ray://172.18.22.5:10001')
    parser.add_argument("--namespace", type=str, help="the name of node", default='learner')
    parser.add_argument("--name", type=str, help="the name of learner", default='learner-node')
    parser.add_argument("--dataset-namespace", type=str, help="the namespace of dataset node", default='dataset')
    parser.add_argument("--dataset-name", type=str, help="the name of replay buffer", default='dataset-node')

    # RL arguments
    parser.add_argument("--save-path", type=str, help="the dir to save log and model", default='SAC-experiment')
    parser.add_argument("--log-path", type=str, help="the log file", default='log.txt')
    parser.add_argument("--train-path", type=str, help="the training log file", default='train.txt')
    parser.add_argument("--step", type=float, help="steps of training", default=5e6)
    parser.add_argument("--hidden-size", type=int, help="number of hidden units per layer", default=512)
    parser.add_argument("--batch-size", type=int, help="number of samples per minibatch", default=256)
    parser.add_argument("--soft-tau", type=float, help="target smoothing coefficient", default=5e-3)
    parser.add_argument("--gamma", type=float, help="steps of training", default=0.99)
    parser.add_argument("--gradient-steps", type=int, help="the num of gradient step per SAC update", default=1)
    parser.add_argument("--soft-q-lr", type=float, help="the learning rate of soft Q function", default=3e-4)
    parser.add_argument("--policy-lr", type=float, help="the learning rate of policy", default=1e-4)
    parser.add_argument("--max-learn-act-ratio", type=float, help="the max ratio of learn step to act step", default=4)

    args = parser.parse_args()
    print('------  ' + args.namespace + ' node connect to ray cluster  ------')
    # ray.init(address=args.address, namespace=args.namespace)
    ray.init(address='auto', namespace=args.namespace, _redis_password='5241590000000000')

    print('------  learner initialize  ------')
    learner = remote_sac_trainer.options(name=args.name).remote(state_dim=(140, 140, 2),
                                                                action_dim=10,
                                                                hidden_dim=args.hidden_size,
                                                                gamma=args.gamma,
                                                                soft_q_lr=args.soft_q_lr,
                                                                policy_lr=args.policy_lr,
                                                                soft_tau=args.soft_tau)


    time.sleep(5)

    print('------  connect to dataset  ------')
    dataset = get_remote_class(name=args.dataset_name, namespace=args.dataset_namespace)
    time.sleep(20)
    check_transition_num(dataset, args.batch_size)

    print('------  start training  ------')
    # 不能把训练写成一个函数，因为这样会导致整个训练过程是一个进程，阻碍了actor与learner的通信
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    log_path = os.path.join(args.save_path, args.log_path)
    train_path = os.path.join(args.save_path, args.train_path)
    with open(log_path, 'w') as f:
        f.write('step,critic_loss,actor_loss,alpha_loss,decoder_loss\n')
    model_path = os.path.join(args.save_path, 'model')
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    for step in range(int(args.step)):
        l_a_r = ray.get(dataset.learn_act_ratio.remote())
        # print(step, l_a_r)
        while l_a_r > args.max_learn_act_ratio:
            time.sleep(1)
            l_a_r = ray.get(dataset.learn_act_ratio.remote())
        state, action, reward, next_state, done = ray.get(dataset.sample.remote(args.batch_size))
        critic_loss, actor_loss, alpha_loss, decoder_loss = ray.get(learner.sac_update.remote(state, action, reward, next_state, done))
        # print(f'{step}  critic_loss: %.2f    actor_loss: %.2f    alpha_loss: %.2f    decoder_loss: %.2f' % (critic_loss, actor_loss, alpha_loss, decoder_loss))
        with open(log_path, 'a') as f:
                f.write('%d,%.4f,%.4f,%.4f,%.4f\n' % (step, critic_loss, actor_loss, alpha_loss, decoder_loss))
        if step % 1000 == 0:
            ray.get(learner.save_paras.remote(os.path.join(model_path, f'{int(step / 1000)}k_SAC.pkl')))

            print('node learner: ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) +
                  'step: %dk   critic_loss: %.2f    actor_loss: %.2f    alpha_loss: %.2f    decoder_loss: %.2f' %
                  (step / 1000, critic_loss, actor_loss, alpha_loss, decoder_loss))
