
"""
在4x4的方框格子中，随机存放了1个奶酪和2瓶毒药，小老鼠在方格框的起始(0,0)位置。
小老鼠为了吃到奶酪需要寻找奶酪(在格子上移动)，小老鼠如果吃到了毒药会被毒死。
求小老鼠吃到奶酪需要移动格子步数的最优解。用深度学习网络来解答
"""
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# CUDA环境是否存在(如果不存在就使用CPU环境)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 环境数据生成器
class EnvDataGenerator:
    @classmethod
    def randomLoc(cls, grid):
        return [np.random.randint(0, grid), np.random.randint(0, grid)]

    @classmethod
    def generateData(cls, board_size):
        mousePos = (0, 0)

        while True:
            allPos = list()
            for row in range(board_size):
                for col in range(board_size):
                    allPos.append((row, col))
            allPos.remove(mousePos)

            # index = np.random.randint(0, len(allPos))
            index = random.randint(0, len(allPos) - 1)
            cheesePos = allPos[index]
            allPos.remove(cheesePos)

            poisonPos = []
            for i in range(2):
                # index = np.random.randint(0, len(allPos))
                index = random.randint(0, len(allPos) - 1)
                poisonPos.append(allPos[index])
                allPos.remove(allPos[index])

            maxRc = board_size - 1
            # 毒药位置不能包围老鼠位置
            boState0 = (mousePos == (0, 0)) \
                       and ((0, 1) in poisonPos) and ((1, 0) in poisonPos)
            # 毒药位置不能包围奶酪位置
            boState1 = (cheesePos == (0, maxRc)) \
                       and ((0, maxRc - 1) in poisonPos and (1, maxRc) in poisonPos)
            boState2 = (cheesePos == (maxRc, 0)) \
                       and ((maxRc - 1, 0) in poisonPos and (maxRc, 1) in poisonPos)
            boState3 = (cheesePos == (maxRc, maxRc)) \
                       and ((maxRc - 1, maxRc) in poisonPos and (maxRc, maxRc - 1) in poisonPos)
            deadPath = True
            if not (deadPath in [boState0, boState1, boState2, boState3]):
                return mousePos, cheesePos, poisonPos
            # print("dead path", [boState0, boState1, boState2, boState3])

# 定义环境
class MouseMazeEnv:
    # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
    Dir_Up = 0
    Dir_Left = 1
    Dir_Right = 2
    Dir_Down = 3
    def __init__(self):
        self.board_size = 4
        mouseLoc, cheeseLoc, poisonLoc = EnvDataGenerator.generateData(self.board_size)
        self.mousePos = mouseLoc
        self.cheesePos = cheeseLoc
        self.poisonPos = poisonLoc
        self.done = False
        self.stepCnt = 0
        self.deadCnt = 0
        self.successCnt = 0
        self.dir = self.Dir_Right
        self.stepRefValue = 0

    def reset(self):
        mouseLoc, cheeseLoc, poisonLoc = EnvDataGenerator.generateData(self.board_size)
        self.mousePos = mouseLoc
        self.cheesePos = cheeseLoc
        self.poisonPos = poisonLoc
        self.dir = self.Dir_Right
        self.done = False
        self.stepCnt = 0
        self.stepRefValue = 0
        return self.getEnvState()

    def getEnvState(self):
        dts = []

        mousePos = self.mousePos
        cheesePos = self.cheesePos
        poisonPos = self.poisonPos

        mouseY, mouseX = mousePos[0], mousePos[1]
        cheeseY, cheeseX = cheesePos[0], cheesePos[1]
        dy = 1.0 * (cheeseY - mouseY) / self.board_size
        dx = 1.0 * (cheeseX - mouseX) / self.board_size
        dts.append(abs(dy))
        dts.append(abs(dx))

        # set mouse hit the wall state value
        curY, curX = mousePos[0], mousePos[1]
        checkPoints = [(curY - 1, curX),  # UP
                       (curY + 1, curX),  # DOWN
                       (curY, curX - 1),  # LEFT
                       (curY, curX + 1)]  # Right
        # state value (1: hit the wall 0: not)
        for tmpP in checkPoints:
            tmpState = 0
            if tmpP[0] < 0 or tmpP[0] >= self.board_size or tmpP[1] < 0 or tmpP[1] >= self.board_size:
                tmpState = 0.5
            dts.append(tmpState)
        # poisoned state
        for tmpP in poisonPos:
            tmpState = 0
            if tmpP == mousePos:
                tmpState = 0.5
            dts.append(tmpState)

        return np.array(dts)

    def step(self, action):
        self.stepCnt += 1
        row, col = self.mousePos[0], self.mousePos[1]
        if action == self.Dir_Up:  # 上移
            row -= 1
        elif action == self.Dir_Down:  # 下移
            row += 1
        elif action == self.Dir_Left:  # 左移
            col -= 1
        elif action == self.Dir_Right:  # 右移
            col += 1
        self.mousePos = (row, col)

        # 奖励机制
        if self.mousePos == self.cheesePos:
            reward = 1.0
            # reward = 1.0 + 1.0 * self.stepRefValue / self.stepCnt
            self.done = True
            # print("Success, eated cheese!")
            self.successCnt += 1
        elif self.mousePos in self.poisonPos:
            reward = -1.0
            self.done = True
            self.deadCnt += 1
            # print("Failure, mouse has been poisoned!")
        else:
            boOut = self.mousePos[0] < 0 or self.mousePos[0] >= self.board_size \
                    or self.mousePos[1] < 0 or self.mousePos[1] >= self.board_size
            if boOut:
                reward = -1.0
                self.done = True
                self.deadCnt += 1
            else:
                reward = 0.0
        return self.getEnvState(), reward, self.done

    def setBoardSize(self, size):
        self.board_size = size

    def showInfo(self, tag):
        print(tag)
        print("mouseLoc:", self.mousePos)
        print("cheeseLoc:", self.cheesePos)
        print("poisonLoc:", self.poisonPos)

# 定义Q网络
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.fc1.to(device)
        self.fc2.to(device)
        self.fc3.to(device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义Deep Q-Learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.q_network = QNetwork(state_size, action_size)
        self.target_q_network = QNetwork(state_size, action_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
        self.loss_function = nn.MSELoss()

        self.q_network.to(device)
        self.target_q_network.to(device)
        self.loss_function.to(device)

        self.gamma = 0.99
        self.epsilon = 0.1
        self.lossValue = -1.0

    def select_action(self, state, env):
        if np.random.rand() < self.epsilon:
            return np.random.choice(4)  # 随机选择动作
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def train(self, state, action, next_state, reward, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = self.q_network(state)
        next_q_values = self.target_q_network(next_state)

        if done:
            q_values[0][action] = reward
        else:
            q_values[0][action] = reward + self.gamma * torch.max(next_q_values)

        self.optimizer.zero_grad()
        loss = self.loss_function(self.q_network(state), q_values)
        loss.backward()
        self.optimizer.step()
        self.lossValue = loss.item()

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())


model_path = "mouse_cheese.pth"


def saveModel(agent):
    torch.save(agent.q_network.state_dict(), model_path)


def loadModel(state_size, action_size):
    loaded_agent = DQNAgent(state_size, action_size)
    loaded_agent.q_network.load_state_dict(torch.load(model_path))
    loaded_agent.q_network.eval()  # 设置模型为评估模式，不再进行梯度更新
    return loaded_agent

# 训练处理
def train_process():
    # 训练
    state_size = 8
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    env = MouseMazeEnv()
    num_episodes = 80000
    total_reward = 0
    for episode in tqdm(range(num_episodes), ascii=True):
        state = env.reset()
        # total_reward = 0

        while not env.done:
            action = agent.select_action(state, env)
            next_state, reward, done = env.step(action)
            agent.train(state, action, next_state, reward, done)
            state = next_state
            total_reward += reward

        agent.update_target_network()

        if episode % 100 == 0:
            lossValue = "%.6f" % agent.lossValue
            # print(f"Episode: {episode}, Total Reward: {total_reward} lossValue:{lossValue} deadCnt:{env.deadCnt} successCnt:{env.successCnt}")
            ratio = "%.3f" % (1.0 * env.successCnt / (env.successCnt + env.deadCnt))
            print(f"epi: {episode}, rewards: {total_reward} loss:{lossValue} fail:{env.deadCnt} suc:{env.successCnt} ratio:{ratio}")
    saveModel(agent)

def testModel():
    state_size = 8
    action_size = 4
    env = MouseMazeEnv()
    # 分析在新格子框中的适应情况
    env.setBoardSize(6)
    agent = loadModel(state_size, action_size)
    total_cnt = 1000
    success_cnt = 0
    for i in range(total_cnt):
        # 测试
        state = env.reset()
        env.showInfo("-" * 30 + f" test first[{i}]:")
        stepCnt = 0
        total_reward = 0
        while not env.done:
            action = agent.select_action(state, env)
            next_state, reward, _ = env.step(action)
            print(f"step={stepCnt},next_state={env.mousePos}")
            stepCnt += 1
            state = next_state
            total_reward = reward
        if total_reward > 0:
            success_cnt += 1
        env.showInfo("-" * 30 + " test end:")
        print(f"Mouse's final location: {env.mousePos}")
    print("success ration=%.3f" % (1.0 * success_cnt / total_cnt))

# train_process()
testModel()