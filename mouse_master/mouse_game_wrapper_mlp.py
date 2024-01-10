
import gym
import numpy as np

from mouse_game import Mouse_Game

class MouseEnv(gym.Env):
    def __init__(self, seed=0, board_size=6, silent_mode=True, limit_step=True):
        super().__init__()
        self.game = Mouse_Game(seed, board_size, silent_mode)
        self.game.reset()

        self.board_size = board_size
        # [0,...,7] mouse,cheese,poison pos value
        # [8,...,13] mouse pos relative to the cheese and poison pos value
        # [14,...,17] mouse hit the wall state value
        self.max_state_len = 18

        self.action_space = gym.spaces.Discrete(4)  # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
        self.observation_space = gym.spaces.Box(
            low=-self.board_size, high=self.board_size,
            shape=(1,self.max_state_len),
            dtype=np.float32
        )

        self.done = False
        # More than enough steps to get the cheese.
        self.step_limit = self.board_size * 2
        self.reward_step_counter = 0

    def reset(self):
        self.game.reset()

        self.done = False
        self.reward_step_counter = 0

        obs = self._generate_observation()
        return obs

    def _generate_observation(self):
        obs = np.zeros((1, self.max_state_len), dtype=np.float32)

        mousePos = self.game.mousePos
        cheesePos = self.game.cheesePos
        poisonPos = self.game.poisonPos

        mouseY, mouseX = mousePos[0], mousePos[1]
        cheeseY, cheeseX = cheesePos[0], cheesePos[1]
        poisonY0, poisonX0 = poisonPos[0][0],poisonPos[0][1]
        poisonY1, poisonX1 = poisonPos[1][0], poisonPos[1][1]
        dts = [mouseY, mouseX, cheeseY, cheeseX, poisonY0, poisonX0, poisonY1, poisonX1]

        # set pos state value
        index = 0
        for tmpS in dts:
            obs[0,index] = tmpS
            index += 1

        # set relative state value
        mc_dy = mouseY - cheeseY
        mc_dx = mouseX - cheeseX
        mp_dy0 = mouseY - poisonY0
        mp_dx0 = mouseX - poisonX0
        mp_dy1 = mouseY - poisonY1
        mp_dx1 = mouseX - poisonX1
        dts = [mc_dy, mc_dx, mp_dy0, mp_dx0, mp_dy1, mp_dx1]
        for tmpS in dts:
            obs[0,index] = tmpS
            index += 1

        # set mouse hit the wall state value
        curY,curX = mousePos[0], mousePos[1]
        checkPoints = [(curY - 1, curX), # UP
                       (curY + 1, curX), # DOWN
                       (curY, curX - 1), # LEFT
                       (curY, curX + 1)] # Right
        # state value (1: hit the wall 0: not)
        hitWallState = self.board_size // 2
        for tmpP in checkPoints:
            if tmpP[0] < 0 or tmpP[0] >= self.board_size or tmpP[1] < 0 or tmpP[1] >= self.board_size:
                obs[0,index] = 1
            else:
                obs[0,index] = 0
            index += 1
        return obs

    # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
    def step(self, action):
        self.done, info = self.game.step(action)
        # print(self.action_space.n)
        # print("step:", action)

        obs = self._generate_observation()

        self.reward_step_counter += 1

        # Max reward on mouse obtain the cheese
        maxReward = 2.0
        if self.done:
            curValue = info["result"]

            cheesePos = self.game.cheesePos
            row, col = cheesePos[0], cheesePos[1]
            maxRelStep = (row + col) * 2
            # Mouse hit the wall
            if curValue == 1:
                reward = -maxReward
                return obs, reward, self.done, info

            # Mouse been poisoned
            if curValue == 2:
                reward = -maxReward
                return obs, reward, self.done, info

            # Mouse obtained cheese
            if curValue == 3:
                reward = maxReward
                return obs, reward, self.done, info

        # Common state
        reward = 0.0
        return obs, reward, self.done, info

    """
    # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
    def step(self, action):
        self.done, info = self.game.step(action)
        # print(self.action_space.n)
        # print("step:", action)

        obs = self._generate_observation()

        self.reward_step_counter += 1
        # Step limit reached, game over.
        if not self.done and (self.reward_step_counter > self.step_limit):
            reward = -0.1 * (self.reward_step_counter - self.board_size * 2)
            self.reward_step_counter = 0
            self.done = True
            return obs, reward, self.done, info

        # Max reward on mouse obtain the cheese
        maxReward = 2.0
        if self.done:
            curValue = info["result"]

            cheesePos = self.game.cheesePos
            row, col = cheesePos[0], cheesePos[1]
            maxRelStep = (row + col) * 2
            # Mouse hit the wall
            if curValue == 1:
                if self.reward_step_counter <= maxRelStep:
                    reward = -maxReward
                else:
                    reward = -max(maxReward * (maxRelStep / self.reward_step_counter), maxReward / 2)
                return obs, reward, self.done, info

            # Mouse been poisoned
            if curValue == 2:
                if self.reward_step_counter <= maxRelStep:
                    reward = -maxReward
                else:
                    reward = -max(maxReward * (maxRelStep / self.reward_step_counter), maxReward / 2)
                return obs, reward, self.done, info

            # Mouse obtained cheese
            if curValue == 3:
                if self.reward_step_counter <= maxRelStep:
                    reward = maxReward
                else:
                    reward = max(maxReward * (maxRelStep / self.reward_step_counter), maxReward / 2)
                return obs, reward, self.done, info

        # Common state
        curMousePos = np.array(info["mouse"])
        preMousePos = np.array(info["mousePre"])
        cheesePos = np.array(self.game.cheesePos)
        commonReward = maxReward * 0.1

        # Mouse near the cheese should give the reward
        # The sum of the squares of all element values and then the square root
        if np.linalg.norm(curMousePos - cheesePos) < np.linalg.norm(preMousePos - cheesePos):
            return obs, commonReward, self.done, info
        else:
            return obs, -commonReward, self.done, info
    """

    def render(self):
        self.game.render()

    def get_action_mask(self):
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])

    # Check action if mouse collided with the wall or poisoned
    def _check_action_validity(self, action):
        # current_direction = self.game.direction
        row, col = self.game.mousePos[0], self.game.mousePos[1]
        if action == 0:  # UP
            row -= 1
        elif action == 1:  # LEFT
            col -= 1
        elif action == 2:  # RIGHT
            col += 1
        elif action == 3:  # DOWN
            row += 1
        # Check if mouse collided with the wall or poisoned
        newPos = (row, col)
        game_over = (newPos in self.game.poisonPos) or row < 0 or row >= self.board_size or col < 0 or col >= self.board_size
        if game_over:
            return False
        else:
            return True
