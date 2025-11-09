import argparse
import os
import json

import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import str2bool, showInfo
import copy
from dataLoader import transitionSet, stackedTransitionSet, PrioritizedExperienceReplay
from network_architecture import PolicyNetwork, MultiQNetwork, \
    LocalSignalMixing, ParameterizedReg, custom_parameterized_aug_optimizer_builder, l2_projection, encoderExactor, \
    ALIXEncoder, ALIXEncoder1, ALIXEncoder2, ALIXEncoder3, StackPolicyNetwork, StackMultiQNetwork,\
    ALIXEncoder4



class StackTrainer:
    def __init__(self, obs_shape, action_dim, actor_hidden_dim, critic_hidden_dim, device, log_alpha, target_entropy,
                 soft_q_lr=1e-3, decoder_lr=1e-3, policy_lr=3e-4, alpha_lr=3e-5, encoder_lr=1e-3, gamma=0.99,
                 soft_tau=0.005, w=0.5, beta=1., PER_flag=True, priorities_coefficient=0.9, priorities_bias=1,
                 encoder='ALIXEncoder1'):
        self.obs_shape = list(obs_shape)
        self.obs_shape[-1] *= 2  # 与上一时刻的一起
        self.device = device
        aug = ParameterizedReg(aug=LocalSignalMixing(pad=2, fixed_batch=True), parameter_init=0.5,
                               param_grad_fn='alix_param_grad', param_grad_fn_args=[3, 0.535, 1e-20])
        encoder = eval(encoder)
        self.encoder = encoder(self.obs_shape, aug=aug).to(device)
        # self.encoder = AllFeatTiedRegularizedEncoder2(obs_shape, aug=aug).to(device)

        feature_dim = self.encoder.feature_dim
        self.actor = StackPolicyNetwork(feature_dim=feature_dim, action_dim=action_dim, hidden_dim=actor_hidden_dim).to(
            device)
        self.soft_q_nets = StackMultiQNetwork(feature_dim, action_dim, critic_hidden_dim).to(device)
        self.target_soft_q_nets = StackMultiQNetwork(feature_dim, action_dim, critic_hidden_dim).to(device)
        for target_param, param in zip(self.target_soft_q_nets.parameters(), self.soft_q_nets.parameters()):
            target_param.data.copy_(param.data)

        self.log_alpha = torch.tensor(log_alpha).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = 0.98 * np.log(action_dim) if target_entropy is None else target_entropy
        self.gamma = gamma
        self.soft_q_criterion = nn.MSELoss()
        self.soft_tau = soft_tau
        self.w = w
        self.beta = beta
        self.alpha_lr = alpha_lr

        self.encoder_optimizer = custom_parameterized_aug_optimizer_builder(encoder=self.encoder,
                                                                            encoder_lr=encoder_lr,
                                                                            lr=2e-3, betas=[0.5, 0.999])
        self.critic_optimizer = optim.Adam(self.soft_q_nets.parameters(), lr=soft_q_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.encoder_lr = encoder_lr
        self.soft_q_lr = soft_q_lr
        self.policy_lr = policy_lr
        self.alpha_lr = alpha_lr

        self.PER_flag = PER_flag
        if self.PER_flag:
            print('using Prioritized Experience Replay')
        else:
            print('not using Prioritized Experience Replay')
        self.priorities_coefficient = priorities_coefficient
        self.priorities_bias = priorities_bias

    @property
    def alpha(self):
        return self.log_alpha.exp()

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
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
        self.encoder.load_state_dict(payload['encoder'])

    def transfer2tensor(self, *paras):
        # out = []
        for para in paras:
            if not isinstance(para, torch.Tensor):
                if len(para.shape) == 4:  # adjust axis for the image as np.ndarray
                    para = np.moveaxis(para, -1, -3)
                elif len(para.shape) == 1:  # add axis for action, reward and done to make the shape as (batch_size, 1)
                    para = np.expand_dims(para, axis=-1)
                para = torch.FloatTensor(para).to(self.device)
            yield para
            # out.append(para)
        # return out

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

        with torch.no_grad():
            _, prob, _ = self.actor.evaluate(feature, last_action)
        q_pre = q.min(dim=0)[0]
        v = torch.einsum('ij,ij->i', q_pre, prob.detach())
        assert len(v.shape) == 1 and v.shape[0] == batch_size
        assert v.shape == q_pre[torch.tensor(range(action.shape[0])), action.view(-1).long()].shape
        gan_loss = v - q_pre[torch.tensor(range(action.shape[0])), action.view(-1).long()]
        # assert gan_loss.shape == batch_size
        # gan_loss = gan_loss.mean()

        return bellman_loss_sum, target_bellman_loss, bellman_loss, gan_loss, \
               q_pre[torch.tensor(range(action.shape[0])), action.view(-1).long()].mean().item(), v.mean().item()

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

    def load_imitation_paras(self, path):
        """
        Initialize actor, alpha, encoder, decoder from imitation learning.
        Q-nets and target Q-nets is not initialized.
        :param path:
        :return:
        """
        payload = torch.load(path)
        self.actor.load_state_dict(payload['actor'])
        self.encoder.load_state_dict(payload['encoder'])
        # self.log_alpha = payload['log_alpha']
        # self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)

    def imitation_loss(self, last_obs, obs, last_action, action):
        """
        模仿学习参数，注意梯度会回传给encoder
        :param last_obs:
        :param obs:
        :param last_action:
        :param action:
        :return:
        """
        feature = self.encoder(torch.cat([last_obs, obs], dim=1))
        _, prob, log_prob = self.actor.evaluate(feature, last_action)

        actor_loss = -torch.log(prob[torch.tensor(range(action.shape[0])), action.view(-1).long()])
        with torch.no_grad():
            # new_action, prob, log_prob = self.actor.evaluate(feature)
            entropy = -torch.einsum('ij,ij->i', log_prob, prob)

        alpha_loss = self.log_alpha.exp() * (entropy - self.target_entropy)
        return actor_loss, alpha_loss, entropy

    def train_step(self, train_set, batch_size, beta):
        out, indices, weights, priorities = train_set.sample(batch_size=batch_size)
        last_obs, obs, last_action, action, reward, next_obs, done, _ = map(np.stack, zip(*out))
        # done = reward > 20
        last_obs, obs, last_action, action, reward, next_obs, done = self.transfer2tensor(last_obs, obs,
                                                                                          last_action, action,
                                                                                          reward, next_obs, done)

        bellman_loss_sum, target_bellman_loss, bellman_loss, gan_loss, q, v = self.critic_loss(last_obs, obs,
                                                                                               last_action, action,
                                                                                               reward, next_obs,
                                                                                               done)
        critic_loss = (1 - beta) * bellman_loss_sum.mean() + beta * gan_loss.mean()
        actor_loss, alpha_loss, entropy = self.actor_loss(last_obs, obs, last_action)
        loss = critic_loss + actor_loss.mean() + alpha_loss.mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        self.encoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optimizer.step()
        self.soft_q_nets.apply(l2_projection(30.))
        self.actor_optimizer.step()
        self.alpha_optimizer.step()
        self.encoder_optimizer.step()
        self.encoder.apply(l2_projection(30.))

        if self.PER_flag:
            priorities = self.priorities_coefficient * priorities + (1 - self.priorities_coefficient) * (
                    np.clip(bellman_loss_sum.detach().cpu().numpy() ** 0.5, 0, 20) + self.priorities_bias)
            train_set.priority_update(indices, priorities)

        for target_param, param in zip(self.target_soft_q_nets.parameters(), self.soft_q_nets.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

        logger = {
            'bellman_loss': bellman_loss.mean().item(),
            'target_bellman_loss': target_bellman_loss.mean().item(),
            'bellman_loss_sum': bellman_loss_sum.mean().item(),
            'gan_loss': gan_loss.mean().item(),
            'critic_loss': critic_loss.mean().item(),
            'mean_q': q,
            'mean_v': v,
            'actor_loss': actor_loss.mean().item(),
            'alpha_loss': alpha_loss.mean().item(),
            'alpha': self.alpha.item(),
            'entropy': entropy.mean().item()
        }
        return logger

    def valid_step(self, valid_set, beta, step):
        self.train(False)
        with torch.no_grad():
            out, indices, weights, priorities = valid_set.sample(batch_size=len(valid_set))
            last_obs, obs, last_action, action, reward, next_obs, done = map(np.stack, zip(*out))
            last_obs, obs, last_action, action, reward, next_obs, done = self.transfer2tensor(last_obs, obs,
                                                                                              last_action, action,
                                                                                              reward, next_obs, done)
            bellman_loss_sum, target_bellman_loss, bellman_loss, gan_loss, q, v = self.critic_loss(last_obs, obs,
                                                                                                   last_action, action,
                                                                                                   reward, next_obs,
                                                                                                   done)
            critic_loss = (1 - beta) * bellman_loss_sum + beta * gan_loss
            if step < 0:  # initialize actor by imitation
                actor_loss, alpha_loss, entropy = self.imitation_loss(last_obs, obs, last_action, action)
            else:
                actor_loss, alpha_loss, entropy = self.actor_loss(last_obs, obs, last_action)

            logger = {
                'valid_bellman_loss': bellman_loss.mean().item(),
                'valid_target_bellman_loss': target_bellman_loss.mean().item(),
                'valid_bellman_loss_sum': bellman_loss_sum.mean().item(),
                'valid_gan_loss': gan_loss.mean().item(),
                'valid_critic_loss': critic_loss.mean().item(),
                'valid_mean_q': q,
                'valid_mean_v': v,
                'valid_actor_loss': actor_loss.mean().item(),
                'valid_alpha_loss': alpha_loss.mean().item(),
                'valid_alpha': self.alpha.item(),
                'valid_entropy': entropy.mean().item()
            }
        self.train(True)
        return logger

    def define_optimizer(self, encoder_lr, soft_q_lr, policy_lr, alpha_lr):
        self.encoder_optimizer = custom_parameterized_aug_optimizer_builder(encoder=self.encoder,
                                                                            encoder_lr=encoder_lr,
                                                                            lr=2e-3, betas=[0.5, 0.999])
        self.critic_optimizer = optim.Adam(self.soft_q_nets.parameters(), lr=soft_q_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        return

    def imitation_step(self, train_set, batch_size, beta):
        out, indices, weights, priorities = train_set.sample(batch_size=batch_size)
        last_obs, obs, last_action, action, reward, next_obs, done, _ = map(np.stack, zip(*out))
        last_obs, obs, last_action, action, reward, next_obs, done = self.transfer2tensor(last_obs, obs,
                                                                                          last_action, action,
                                                                                          reward, next_obs, done)
        bellman_loss_sum, target_bellman_loss, bellman_loss, gan_loss, q, v = self.critic_loss(last_obs, obs,
                                                                                               last_action, action,
                                                                                               reward, next_obs,
                                                                                               done)

        critic_loss = (1 - beta) * bellman_loss_sum.mean() + beta * gan_loss.mean()
        actor_loss, alpha_loss, entropy = self.imitation_loss(last_obs, obs, last_action, action)
        loss = critic_loss + actor_loss.mean()  # 此时actor的entropy与alpha无关，优化alpha loss没有意义
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.encoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optimizer.step()
        self.soft_q_nets.apply(l2_projection(30.))
        self.actor_optimizer.step()
        self.encoder_optimizer.step()
        self.encoder.apply(l2_projection(30.))

        for target_param, param in zip(self.target_soft_q_nets.parameters(), self.soft_q_nets.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

        logger = {
            'bellman_loss': bellman_loss.mean().item(),
            'target_bellman_loss': target_bellman_loss.mean().item(),
            'bellman_loss_sum': bellman_loss_sum.mean().item(),
            'gan_loss': gan_loss.mean().item(),
            'critic_loss': critic_loss.mean().item(),
            'mean_q': q,
            'mean_v': v,
            'actor_loss': actor_loss.mean().item(),
            'alpha_loss': alpha_loss.mean().item(),
            'alpha': self.alpha.item(),
            'entropy': entropy.mean().item()
        }
        return logger

    def train(self, mode: bool = True):
        self.encoder.train(mode)
        self.actor.train(mode)
        self.soft_q_nets.train(mode)
        self.target_soft_q_nets.train(mode)
        return


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=float, help="steps of training", default=5e4)
    parser.add_argument("--soft-q-lr", type=float, help="the learning rate of soft q-nets", default=3e-4)
    parser.add_argument("--encoder-lr", type=float, help="the learning rate of encoder and decoder", default=3e-4)
    parser.add_argument("--policy-lr", type=float, help="the learning rate of policy nets", default=1e-5)
    parser.add_argument("--alpha-lr", type=float, help="the learning rate of alpha", default=1e-6)
    parser.add_argument("--save-path", type=str, help="the dir to save log and model",
                        default='/devdata1/lhdata/off+online/test')
    parser.add_argument("--batch-size", type=int, help="number of samples per minibatch", default=128)
    parser.add_argument("--log-alpha", type=float, help="the init log alpha", default=0.)
    parser.add_argument("--target-entropy", type=float, help="the target entropy", default=1.0)
    parser.add_argument("--w", type=float, help="the coefficient of the double Q residual algorithm loss", default=0.5)
    parser.add_argument("--gamma", type=float, help="the gamma in RL", default=.9)
    parser.add_argument("--soft-tau", type=float, help="the tau for target Q-net update", default=.05)
    parser.add_argument("--beta", type=float, help="the coefficient of the gan loss", default=.3)
    parser.add_argument("--actor-hidden-dim", type=int, help="dimension of hidden layer in actor", default=256)
    parser.add_argument("--critic-hidden-dim", type=int, help="dimension of hidden layer in critic", default=256)
    parser.add_argument("--warm-step", type=int, help="the step to warm Q-net and target Q-net", default=40000)
    parser.add_argument("--PER-flag", type=str2bool, help="whether use Prioritized Experience Replay", default=True)
    parser.add_argument("--PER-coefficient", type=float, help="the coefficient of PER weight update", default=.9)
    parser.add_argument("--PER-bias", type=float, help="the bias of PER weight update", default=1.)
    parser.add_argument("--encoder", type=str, help="the encoder to use", default='ALIXEncoder1',
                        choices=['ALIXEncoder', 'ALIXEncoder1', 'ALIXEncoder2', 'ALIXEncoder3', 'ALIXEncoder4'])
    parser.add_argument("--seed", type=int, help="random seed", default=1)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_set = PrioritizedExperienceReplay(
        load_path='/devdata1/lhdata/off+online/rule-based data/20230109.pt')
    # train_set = PrioritizedExperienceReplay(
    #     load_path='/devdata/lhdata/demonstration/buffer/stackedTransitionSet_5_new_dense_all2833.pt')
    # train_set = PrioritizedExperienceReplay(
    #     load_path='/devdata/lhdata/demonstration/buffer/stackedTransitionSet_5_all2865.pt')
    # train_set = PrioritizedExperienceReplay(
    #     load_path='/devdata/lhdata/demonstration/buffer/stackedTransitionSet_5_train2436.pt')
    # valid_set = PrioritizedExperienceReplay(
    #     load_path='/devdata/lhdata/demonstration/buffer/stackedTransitionSet_5_valid429.pt')
    # train_set = PrioritizedExperienceReplay(
    #     load_path='/devdata/lhdata/demonstration/buffer/stackedTransitionSet_5_trajectory_train.pt')
    # valid_set = PrioritizedExperienceReplay(
    #     load_path='/devdata/lhdata/demonstration/buffer/stackedTransitionSet_5_trajectory_valid.pt')
    trainer = StackTrainer(obs_shape=(140, 140, 2),
                           action_dim=10,
                           actor_hidden_dim=args.actor_hidden_dim,
                           critic_hidden_dim=args.critic_hidden_dim,
                           gamma=args.gamma,
                           device=device,
                           log_alpha=args.log_alpha,
                           target_entropy=args.target_entropy,
                           soft_q_lr=args.soft_q_lr,
                           alpha_lr=args.alpha_lr,
                           policy_lr=args.policy_lr,
                           encoder_lr=args.encoder_lr,
                           soft_tau=args.soft_tau,
                           w=args.w,
                           beta=args.beta,
                           PER_flag=args.PER_flag,
                           priorities_coefficient=args.PER_coefficient,
                           priorities_bias=args.PER_bias,
                           encoder=args.encoder)

    if not os.path.exists(os.path.join(args.save_path, 'model')):
        os.makedirs(os.path.join(args.save_path, 'model'))
    with open(os.path.join(args.save_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    console_log_key = ['bellman_loss_sum', 'gan_loss', 'actor_loss', 'alpha_loss', 'mean_q', 'mean_v']
    log_key = ['bellman_loss', 'target_bellman_loss', 'bellman_loss_sum', 'gan_loss',
               'critic_loss', 'actor_loss', 'alpha_loss', 'alpha', 'entropy', 'mean_q', 'mean_v', ]
    # 'valid_bellman_loss', 'valid_target_bellman_loss', 'valid_bellman_loss_sum', 'valid_gan_loss',
    # 'valid_critic_loss', 'valid_actor_loss', 'valid_alpha_loss', 'valid_entropy', 'valid_mean_q',
    # 'valid_mean_v']
    step = - args.warm_step
    info = 'step'
    for key in log_key:
        info += ',' + key
    with open(os.path.join(args.save_path, 'log.csv'), 'w') as f:
        f.write(info + '\n')

    trainer.define_optimizer(encoder_lr=args.policy_lr,
                             soft_q_lr=args.soft_q_lr,
                             policy_lr=args.policy_lr,
                             alpha_lr=args.alpha_lr
                             )

    while step < int(args.step):
        # logger = trainer.policy_evaluation_with_valid(train_set, valid_set, args.batch_size, args.beta)
        # logger.update(trainer.policy_improvement_with_valid(train_set, valid_set, args.batch_size))
        if step < 0:
            logger = trainer.imitation_step(train_set, args.batch_size, args.beta)
        else:
            logger = trainer.train_step(train_set, args.batch_size, args.beta)
        step += 1
        if step % 5000 == 0:
            # logger.update(trainer.valid_step(valid_set, args.beta, step))
            showInfo(step, logger, console_log_key)
            info = str(step)
            for key in log_key:
                if key in logger.keys():
                    info += ',%.4f' % logger[key]
                else:
                    info += ',None'
            with open(os.path.join(args.save_path, 'log.csv'), 'a') as f:
                f.write(info + '\n')
            trainer.save_paras(os.path.join(args.save_path, 'model', '%dk.pt' % (step / 1000)))
        if step == 0:
            # 重置优化器的冲量等状态，避免之前的BC的冲量对RL的影响
            trainer.actor_optimizer.state = collections.defaultdict(dict)
            trainer.critic_optimizer.state = collections.defaultdict(dict)
            trainer.encoder_optimizer.state = collections.defaultdict(dict)
            # trainer.define_optimizer(encoder_lr=args.encoder_lr,
            #                          soft_q_lr=args.soft_q_lr,
            #                          policy_lr=args.policy_lr,
            #                          alpha_lr=args.alpha_lr
            #                          )
