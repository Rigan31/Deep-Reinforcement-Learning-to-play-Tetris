import time
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn

from tetris import Tetris
from collections import deque
import csv


class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.deepNeural1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.deepNeural2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.deepNeural3 = nn.Sequential(nn.Linear(64, 1))

        self.createWeights()

    def createWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.deepNeural1(x)
        x = self.deepNeural2(x)
        x = self.deepNeural3(x)

        return x

scores = []

def train(learning_rate=1e-3, gamma=0.99):
    width = 10
    height = 20
    block_size = 30
    batch_size = 512
    learning_rate = learning_rate
    gamma = gamma
    initial_epsilon = 1
    final_epsilon = 1e-3
    num_decay_epochs = 2000
    num_epochs = 3000 # 3000
    save_interval = 500 # 1000
    replay_memory_size = 30000
    saved_path = "epoch_models"
    random_seed = 123
    
    
    
    torch.manual_seed(random_seed)

    env = Tetris(width=width, height=height, block_size=block_size)
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    state = env.reset()
    
    replay_memory = deque(maxlen=replay_memory_size)
    epoch = 0
    epoch_time = time.time()

    while epoch < num_epochs:

        next_steps = env.get_next_states()
        epsilon = final_epsilon + (max(num_decay_epochs - epoch, 0) * (
                initial_epsilon - final_epsilon) / num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)

        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        model.train()

        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :]
        action = next_actions[index]

        reward, done = env.step(action, render=False)

        replay_memory.append([state, reward, next_state, done])
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines

            state = env.reset()
            # print("score: ", final_score)
        else:
            state = next_state
            continue


        if len(replay_memory) < replay_memory_size / 10:

            continue

        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))


        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}, Epoch time: {}".format(
            epoch,
            num_epochs,
            action,
            final_score,
            final_tetrominoes,
            final_cleared_lines,
            time.time() - epoch_time))
        epoch_time = time.time()

        if epoch > 0 and epoch % save_interval == 0:
            scores.append({'learning_rate': learning_rate, 'gamma': gamma, 'epoch': epoch, 'score': final_score})
            torch.save(model, "{}/tetris_lr={}gm={}ep={}".format(saved_path, learning_rate, gamma, epoch))

    torch.save(model, "{}/tetris".format(saved_path))


if __name__ == "__main__":
    learning_rate = [0.0001, 0.0005, 0.001, 0.005]
    gamma = [0.96, 0.97, 0.98, 0.99]
    for lr in learning_rate:
        for gm in gamma:
            train(lr, gm)
    
    with open('scores.csv', 'w', newline='') as csvfile:
        fieldnames = ['learning_rate', 'gamma', 'epoch', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for score in scores:
            writer.writerow(score)