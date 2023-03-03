import torch
import cv2
from tetris import Tetris
import torch.nn as nn

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

width = 10
height = 20
block_size = 30
fps = 300
epoch_model_path = "epoch_models"
output = "final.mp4"
random_seed = 123

torch.manual_seed(random_seed)
model = torch.load("{}/tetris_lr={}gm={}ep={}".format(epoch_model_path, 0.001, 0.99 , 3000), map_location=lambda storage, loc: storage)
# model = torch.load("{}tetris".format(epoch_model_path), map_location=lambda storage, loc: storage)

model.eval()

env = Tetris(width=width, height=height, block_size=block_size)
env.reset()
out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"MJPG"), fps,
                        (int(1.5*width*block_size), height*block_size))

while True:
    next_steps = env.get_next_states()
    next_actions, next_states = zip(*next_steps.items())
    next_states = torch.stack(next_states)
    predictions = model(next_states)[:, 0]
    index = torch.argmax(predictions).item()
    action = next_actions[index]
    _, done = env.step(action, render=True, video=out)

    if done:
        out.release()
        break