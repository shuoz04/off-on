import random
import time
import numpy as np
import ray
import argparse
import torch
from utils import str2bool


class ReplayBuffer:
    def __init__(self, capacity, stack=True):
        self.capacity = int(capacity)
        self.buffer = []
        self.act_step = 0
        self.learn_step = 0
        self.stack = stack
        self.obs = None
        if stack:
            self.last_obs = None
            self.last_action = None

    def type(self):
        return 'online'

    def push(self, *para):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        if self.obs is None:
            self.obs, action, reward, next_obs, done = para
        else:
            _, action, reward, next_obs, done = para
        if self.last_obs is None:
            self.last_obs = self.obs.copy()
            self.last_action = np.array([10])
        self.buffer[
            self.act_step % self.capacity] = self.last_obs, self.obs, self.last_action, action, reward, next_obs, done
        self.act_step += 1
        if done:
            self.last_obs = None
            self.last_action = None
            self.obs = None
        else:
            self.last_obs = self.obs.copy()
            self.last_action = action
            self.obs = next_obs
        # if self.act_step == 5000:
        #     payload = [self.buffer,
        #                self.act_step,
        #                self.learn_step]
        #     torch.save(payload, 'dataset.pt')

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self)))
        data = map(np.stack, zip(*batch))
        self.learn_step += 1
        return data

    def reset(self):
        self.buffer = []
        self.act_step = 0
        self.learn_step = 0
        # payload = torch.load('dataset.pt')
        # self.buffer = payload[0]
        # self.act_step = 5000
        # self.learn_step = 0

    def learn_act_ratio(self):
        return self.learn_step / self.act_step

    def __len__(self):
        return len(self.buffer)


class mixReplayBuffer:
    def __init__(self, offline_path, capacity, stack=True):
        self.online_buffer = ReplayBuffer(capacity, stack)
        offline_payload = torch.load(offline_path)
        self.offline_buffer = ReplayBuffer(len(offline_payload), stack)
        self.offline_buffer.buffer = offline_payload

    def type(self):
        return 'mix'

    def online_length(self):
        return len(self.online_buffer)

    def offline_length(self):
        return len(self.offline_buffer)

    def learn_act_ratio(self):
        return self.online_buffer.learn_act_ratio()

    def reset(self):
        self.online_buffer.reset()

    def online_offline_ratio(self):
        return len(self.online_buffer)/len(self.offline_buffer)

    def push(self, *para):
        self.online_buffer.push(*para)

    def sample(self, offline_batch_size, online_batch_size):
        offline_last_obs, offline_obs, offline_last_action, offline_action, offline_reward, offline_next_obs, offline_done = self.offline_buffer.sample(offline_batch_size)
        online_last_obs, online_obs, online_last_action, online_action, online_reward, online_next_obs, online_done = self.online_buffer.sample(
            online_batch_size)
        # 离线数据在前，在线数据在后
        last_obs = np.concatenate((offline_last_obs, online_last_obs), axis=0)
        obs = np.concatenate((offline_obs, online_obs), axis=0)
        last_action = np.concatenate((offline_last_action, online_last_action), axis=0)
        action = np.concatenate((offline_action, online_action), axis=0)
        reward = np.concatenate((offline_reward, online_reward), axis=0)
        next_obs = np.concatenate((offline_next_obs, online_next_obs), axis=0)
        done = np.concatenate((offline_done, online_done), axis=0)
        return last_obs, obs, last_action, action, reward, next_obs, done

    def online_len(self):
        return self.online_buffer.__len__()

    def offline_len(self):
        return self.offline_buffer.__len__()


class ReplayBufferWithInitial:
    def __init__(self, capacity, offline_path, stack=True):
        self.capacity = int(capacity)
        self.offline_path = offline_path
        self.buffer = torch.load(offline_path)
        assert len(self.buffer) <= self.capacity
        self.position = len(self.buffer) % self.capacity
        self.act_step = 0
        self.learn_step = 0
        self.stack = stack
        self.obs = None
        if self.stack:
            self.last_obs = None
            self.last_action = None

    def type(self):
        return 'initial'

    def push(self, *para):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        if self.obs is None:
            self.obs, action, reward, next_obs, done = para
        else:
            _, action, reward, next_obs, done = para
        if self.last_obs is None:
            self.last_obs = self.obs.copy()
            self.last_action = np.array([10])
        self.buffer[self.position] = self.last_obs, self.obs, self.last_action, action, reward, next_obs, done
        self.act_step += 1
        self.position = (self.position+1) % self.capacity
        if done:
            self.last_obs = None
            self.last_action = None
            self.obs = None
        else:
            self.last_obs = self.obs.copy()
            self.last_action = action
            self.obs = next_obs
        # if self.act_step == 5000:
        #     payload = [self.buffer,
        #                self.act_step,
        #                self.learn_step]
        #     torch.save(payload, 'dataset.pt')

    def sample(self, batch_size):
        batch = random.sample(self.buffer, max(batch_size, len(self)))
        data = map(np.stack, zip(*batch))
        self.learn_step += 1
        return data

    def reset(self):
        self.buffer = torch.load(self.offline_path)
        assert len(self.buffer) <= self.capacity
        self.position = len(self.buffer) % self.capacity
        self.act_step = 0
        self.learn_step = 0
        self.obs = None
        if self.stack:
            self.last_obs = None
            self.last_action = None

    def learn_act_ratio(self):
        return self.learn_step / self.act_step

    def online_ratio_and_step(self):
        """
        在所有数据中在线数据的比例
        :return:
        """
        if self.act_step >= self.capacity:
            return 1., self.capacity
        else:
            return self.act_step / len(self.buffer), self.act_step

    def online_len(self):
        if self.act_step >= self.capacity:
            return self.capacity
        else:
            return self.act_step

    def __len__(self):
        return len(self.buffer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", type=str, help="the address of ray head node", default='ray://172.18.22.9:10001')
    parser.add_argument("--namespace", type=str, help="the name of node", default='dataset')
    parser.add_argument("--name", type=str, help="the name of replay buffer", default='dataset-node')
    parser.add_argument("--capacity", type=int, help="the capacity of replay buffer", default=1e4)
    parser.add_argument("--type", type=str, help="type of replay buffer", default='online')
    parser.add_argument("--offline-path", type=str, help="the path of offline replay buffer", default='/devdata1/lhdata/off+online/rule-based data/20230130-20000.pt')
    args = parser.parse_args()

    print('------' + args.namespace + ' node connect to ray cluster------')
    # ray.init(address=args.address, namespace=args.namespace)
    ray.init(address='auto', namespace=args.namespace, _redis_password='5241590000000000')
    # ray.init(address='ray://172.18.22.5:10001', namespace=args.namespace)

    if args.type == 'mix':
        print('------initialize mix dataset------')
        RemoteMixReplayBuffer = ray.remote(mixReplayBuffer)
        buffer = RemoteMixReplayBuffer(offline_path=args.offline_path, capacity=args.capacity)
    elif args.type == 'online':
        print('------initialize online dataset------')
        RemoteReplayBuffer = ray.remote(ReplayBuffer)
        buffer = RemoteReplayBuffer.options(name=args.name).remote(args.capacity)
    elif args.tyep == 'initial':
        print('------initialize online dataset------')
        RemoteReplayBuffer = ray.remote(ReplayBuffer)
        buffer = RemoteReplayBuffer.options(name=args.name).remote(args.capacity)
    else:
        assert False, 'wrong buffer type!'
    buffer.reset.remote()
    while True:
        time.sleep(1)
    # model2 = ray.get_actor(name='buffer', namespace='dataset')
    # print(ray.get(model2.__len__.remote()))
