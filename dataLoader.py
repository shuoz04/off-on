"""
load transitions
each transition is a tuple: origin figure(np.ndarray(560,560,3)), obs(np.ndarray(140,140,2)), action(int),
                            reward(tuple(sparse, dense, safe, sum)), next obs(np.ndarray(140,140,2)), done(bool),
                            current(int), remote
"""
import math
import torch
import os
import random
import numpy as np


class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def push(self, *para):
        self.buffer.append(para)
        if len(self.buffer) == 1:
            self.obs_shape = self.buffer[0][0].shape

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        data = map(np.stack, zip(*batch))
        return data

    def reset(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)


class ReplayBufferWithCapacity:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = int(capacity)
        self.position = 0

    def push(self, *para):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = para
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        data = map(np.stack, zip(*batch))
        return data

    def reset(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)


class obsSet(ReplayBuffer):
    def __init__(self, data_path='/devdata/lhdata/demonstration/'):
        super(obsSet, self).__init__()
        files = os.listdir(data_path)
        for file in files:
            if file[-3:] == '.pt':
                transitions = torch.load(os.path.join(data_path, file))
                for transition in transitions:
                    self.push(transition[1])
                self.push(transitions[-1][1])

        self.obs_shape = self.buffer[0].shape

    def push(self, obs):
        self.buffer.append(obs)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return np.array(batch)


class policySet(ReplayBuffer):
    def __init__(self, data_path='/devdata/lhdata/demonstration/', target=8, load_path=None):
        super(policySet, self).__init__()
        if load_path is not None:
            self.buffer = torch.load(load_path)
        else:
            files = os.listdir(data_path)
            i = 0
            j = 0
            back_sum = 0

            for file in files:
                if file[-3:] == '.pt':
                    if int(file[-5:-3]) == target:
                        j += 1
                        transitions = torch.load(os.path.join(data_path, file))
                        if transitions[-1][3][-1] > 390:
                            i += 1
                            for transition in transitions:
                                if transition[2] < 5:
                                    back_sum += 1
                                self.push(transition[1], transition[2])
                        print(file, transitions[-1][3][-1], transitions[-1][-2])
            assert self.__len__() > 0, 'Load no data!!!'
            print(f'Here are {j} demonstrations, {i} of which are successful.')
            print(back_sum / i)
        self.obs_shape = self.buffer[0][0].shape


class stackedPolicySet(ReplayBuffer):
    def __init__(self, data_path='/devdata/lhdata/demonstration/', target=8, load_path=None):
        super(stackedPolicySet, self).__init__()
        if load_path is not None:
            self.buffer = torch.load(load_path)
        else:
            files = os.listdir(data_path)
            i = 0
            j = 0
            back_sum = 0

            for file in files:
                if file[-3:] == '.pt':
                    if int(file[-5:-3]) == target:
                        j += 1
                        transitions = torch.load(os.path.join(data_path, file))
                        if transitions[-1][3][-1] > 390:
                            last_frame = transitions[0][1]
                            last_action = 10
                            i += 1
                            for transition in transitions:
                                if transition[2] < 5:
                                    back_sum += 1
                                self.push(last_frame, transition[1], last_action, transition[2])

                                last_frame = transition[1]
                                last_action = transition[2]
                        print(file, transitions[-1][3][-1], transitions[-1][-2])
            assert self.__len__() > 0, 'Load no data!!!'
            print(f'Here are {j} demonstrations, {i} of which are successful.')
            print(back_sum / i)
        self.obs_shape = self.buffer[0][0].shape


class transitionSet(ReplayBuffer):
    def __init__(self, data_path='/devdata/lhdata/demonstration/', target=5, load_path=None):
        super(transitionSet, self).__init__()
        if load_path is not None:
            self.buffer = torch.load(load_path)
        else:
            files = os.listdir(data_path)
            i = 0
            j = 0
            back_sum = 0

            for file in files:
                if file[-3:] == '.pt':
                    if int(file[-5:-3]) == target:
                        j += 1
                        transitions = torch.load(os.path.join(data_path, file))
                        if transitions[-1][3][-1] > 390:
                            i += 1
                            for transition in transitions:
                                if transition[2] < 5:
                                    back_sum += 1
                                obs = transition[1]
                                action = transition[2]
                                next_obs = transition[4]
                                reward = transition[3][0] / 20 + transition[3][1] + transition[3][2] / 5
                                done = transition[-2]
                                self.push(obs, action, reward, next_obs, done)
                        print(file, transitions[-1][3][-1], transitions[-1][-2])
            assert self.__len__() > 0, 'Load no data!!!'
            print(f'Here are {j} demonstrations, {i} of which are successful.')
            print(back_sum / i)
        self.obs_shape = self.buffer[0][0].shape


class stackedTransitionSet(ReplayBuffer):
    def __init__(self, load_path=None):
        super(stackedTransitionSet, self).__init__()
        if load_path is not None:
            self.buffer = torch.load(load_path)
            self.obs_shape = self.buffer[0][0].shape

    def split(self, valid_size=0.01):
        if valid_size < 1:
            valid_size = int(len(self) * valid_size)
        valid_index = random.sample(range(len(self)), int(valid_size))
        train_index = list(set(range(len(self))) - set(valid_index))
        train_set = stackedTransitionSet()
        train_set.buffer = [self.buffer[index] for index in train_index]
        valid_set = stackedTransitionSet()
        valid_set.buffer = [self.buffer[index] for index in valid_index]
        return train_set, valid_set

# SunTree和PrioritizedExperienceReplay参考https://github.com/takoika/PrioritizedExperienceReplay

class SumTree(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.tree_level = math.ceil(math.log(max_size + 1, 2)) + 1  # 树的层数，树的叶子节点个数不小于max_size
        self.tree_size = 2 ** self.tree_level - 1  # 树的总节点数
        self.tree = [0 for i in range(self.tree_size)]  # 记录每个节点的权重，未填充的叶子节点初始权重为0（即所有节点权重均为0）
        self.data = [None for i in range(self.max_size)]  # 记录每个叶子节点对应的数据。只有叶子节点对应于buffer中的数据
        self.size = 0  # 已填充的数据大小
        self.cursor = 0  # 下一个填充数据的位置

    def add(self, contents, value):
        index = self.cursor
        self.cursor = (self.cursor + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        self.data[index] = contents
        self.val_update(index, value)

    def get_val(self, index):
        tree_index = 2 ** (self.tree_level - 1) - 1 + index
        return self.tree[tree_index]

    def val_update(self, index, value):
        tree_index = 2 ** (self.tree_level - 1) - 1 + index  # 第index个叶子节点对应的节点编号
        diff = value - self.tree[tree_index]
        self.reconstruct(tree_index, diff)

    def reconstruct(self, tindex, diff):
        self.tree[tindex] += diff
        if not tindex == 0:
            tindex = int((tindex - 1) / 2)  # 更新父节点，使父节点的权重依然为当前节点与兄弟节点权重之和
            self.reconstruct(tindex, diff)

    def find(self, value, norm=True):
        """
        给定一个随机数，返回对应的数据、权重和位置
        :param value: 随机数
        :param norm: 如果为True，则表示输入的随机数value是正则化的，即（0,1）之间，需要将value乘以根节点的权重
        :return:数据，权重，数据位置
        """
        if norm:
            value *= self.tree[0]
        return self._find(value, 0)

    def _find(self, value, index):
        if 2 ** (self.tree_level - 1) - 1 <= index:  # index对应的节点已经是叶子节点，则直接返回数据，权重，数据位置
            return self.data[index - (2 ** (self.tree_level - 1) - 1)], self.tree[index], index - (
                    2 ** (self.tree_level - 1) - 1)

        left = self.tree[2 * index + 1]

        if value <= left:  # 如果value小于左孩子的值
            return self._find(value, 2 * index + 1)  # 则在左分支继续寻找
        else:  # 如果value大于左孩子的值
            return self._find(value - left, 2 * (index + 1))  # 则在右分支继续寻找（value要减去左孩子的值）

    def print_tree(self):
        """
        打印每个节点的权重
        :return:
        """
        for k in range(1, self.tree_level + 1):
            for j in range(2 ** (k - 1) - 1, 2 ** k - 1):
                print(self.tree[j], end=' ')
            print()

    def filled_size(self):
        return self.size


class PrioritizedExperienceReplay(object):
    def __init__(self, load_path, buffer=None, alpha=1, init_priority=20):
        if load_path != None:
            buffer = torch.load(load_path)
        memory_size = len(buffer)
        self.tree = SumTree(memory_size)
        self.memory_size = memory_size
        self.alpha = alpha
        for data in buffer:
            self.tree.add(data, init_priority ** alpha)

    def sample(self, batch_size, beta=0):
        assert self.tree.filled_size() >= batch_size, \
            f'The batch size {batch_size} is too large. The buffer has {self.tree.filled_size()} data.'

        out = []
        indices = []
        weights = []
        priorities = []
        for _ in range(batch_size):
            r = random.random()
            data, priority, index = self.tree.find(r)
            priorities.append(priority)
            weights.append(
                (1. / self.memory_size / priority) ** beta if priority > 1e-16 else 0)  # 用于importance sampling
            indices.append(index)
            out.append(data)
            self.priority_update([index], [0])  # To avoid duplicating 避免重复采样！！！

        self.priority_update(indices, priorities)  # Revert priorities

        indices = np.array(indices)
        weights = np.array(weights)
        priorities = np.array(priorities)
        weights /= max(weights)  # 只需要数据而不用重要性采样

        return out, indices, weights, priorities

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.

        Parameters
        ----------
        indices :
            list of sample indices
        """
        if isinstance(indices, list):
            for i, p in zip(indices, priorities):
                self.tree.val_update(i, p ** self.alpha)
        else:
            assert isinstance(indices, np.ndarray)
            for i in range(len(indices)):
                self.tree.val_update(indices[i], priorities[i] ** self.alpha)

    def reset_alpha(self, alpha):
        """ Reset a exponent alpha.

        Parameters
        ----------
        alpha : float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(i) ** -old_alpha for i in range(self.tree.filled_size())]
        self.priority_update(range(self.tree.filled_size()), priorities)

    def __len__(self):
        return self.tree.filled_size()


if __name__ == '__main__':
    data_path = '/devdata/lhdata/demonstration/data'
    # set = obsSet(data_path)
    # print(len(set))
    target = 5
    # set = policySet(data_path, 5)
    # print(len(set), set.obs_shape)
    # torch.save(set.buffer, f'/devdata/lhdata/demonstration/buffer/policySet_{target}.pt')
    set = stackedPolicySet(data_path, 5, load_path=f'/devdata/lhdata/demonstration/buffer/stackedPolicySet_{target}.pt')
    print(len(set))
    # print(len(set), set.obs_shape)
    # torch.save(set.buffer, f'/devdata/lhdata/demonstration/buffer/stackedPolicySet_{target}.pt')
