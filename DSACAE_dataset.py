import random
import time
import numpy as np
import ray
import argparse
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = []
        self.act_step = 0
        self.learn_step = 0
        
    def push(self, *para):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.act_step % self.capacity] = para
        self.act_step += 1
        if self.act_step == 10000:
            payload = [self.buffer,
                       self.act_step,
                       self.learn_step]
            torch.save(payload, 'dataset.pt')

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", type=str, help="the address of ray head node", default='ray://172.18.22.9:10001')
    parser.add_argument("--namespace", type=str, help="the name of node", default='dataset')
    parser.add_argument("--name", type=str, help="the name of replay buffer", default='dataset-node')
    parser.add_argument("--capacity", type=int, help="the capacity of replay buffer", default=1e4)
    args = parser.parse_args()

    print('------' + args.namespace + ' node connect to ray cluster------')
    # ray.init(address=args.address, namespace=args.namespace)
    ray.init(address='auto', namespace=args.namespace, _redis_password='5241590000000000')
    # ray.init(address='ray://172.18.22.5:10001', namespace=args.namespace)

    print('------initialize dataset------')
    RemoteReplayBuffer = ray.remote(ReplayBuffer)
    buffer = RemoteReplayBuffer.options(name=args.name).remote(args.capacity)
    buffer.reset.remote()
    while True:
        time.sleep(1)
    # model2 = ray.get_actor(name='buffer', namespace='dataset')
    # print(ray.get(model2.__len__.remote()))
