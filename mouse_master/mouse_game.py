
import random
import numpy as np

class Mouse_Game():
    Dir_Left = "LEFT"
    Dir_Right = "RIGHT"
    Dir_Up = "UP"
    Dir_Down = "DOWN"

    def __init__(self, seed=0, board_size=6,  silent_mode=True):
        self.seed = seed
        self.board_size = board_size
        self.silent_mode = silent_mode
        self.mousePos = None
        self.cheesePos = None
        self.poisonPos = None
        self.direction = None

        self.stepCnt = 0
        random.seed(seed)

    def reset(self):
        self._generateData()
        self.stepCnt = 0
        self.direction = self.Dir_Right

    # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
    def step(self, action):
        self._update_direction(action)  # Update direction based on action.

        # Move mouse based on current action.
        row, col = self.mousePos[0], self.mousePos[1]
        if self.direction == self.Dir_Up:
            row -= 1
        elif self.direction == self.Dir_Down:
            row += 1
        elif self.direction == self.Dir_Left:
            col -= 1
        elif self.direction == self.Dir_Right:
            col += 1
        self.stepCnt += 1
        mousePrePos = self.mousePos

        slt_info = ["Common", "Hit wall", "Poisoned", "Cheese obtained"]
        slt_value = [0, 1, 2, 3]
        curValue = slt_value[0]

        # check if mouse obtained cheese
        done = False
        if (row, col) == self.cheesePos:
            done = True
            curValue = slt_value[3]

        # check if mouse been poisoned
        if not done and ((row, col) in self.poisonPos):
            done = True
            curValue = slt_value[2]

        # check if mouse collided with wall
        if not done and (row < 0 or row >= self.board_size or col < 0 or col >= self.board_size):
            done = True
            curValue = slt_value[1]

        # Set the mouse's new pos for updating the obs
        self.mousePos = (row, col)

        if done and not self.silent_mode:
            print("done:", slt_info[curValue])

        info = {"mouse": self.mousePos,
                "mousePre": mousePrePos,
                "result": curValue}

        return done, info

    def _generateData(self):
        self.mousePos = (0, 0)

        while True:
            allPos = list()
            for row in range(self.board_size):
                for col in range(self.board_size):
                    allPos.append((row, col))
            allPos.remove(self.mousePos)

            # index = np.random.randint(0, len(allPos))
            index = random.randint(0, len(allPos) - 1)
            self.cheesePos = allPos[index]
            allPos.remove(self.cheesePos)

            self.poisonPos = []
            for i in range(2):
                # index = np.random.randint(0, len(allPos))
                index = random.randint(0, len(allPos) - 1)
                self.poisonPos.append(allPos[index])
                allPos.remove(allPos[index])

            maxRc = self.board_size - 1
            # poison pos cannot wrapper mouse pos
            boState0 = (self.mousePos == (0, 0)) \
                       and ((0,1) in self.poisonPos) and ((1,0) in self.poisonPos)
            #  poison pos cannot wrapper cheese pos
            boState1 = (self.cheesePos == (0, maxRc)) \
                       and ((0,maxRc - 1) in self.poisonPos and (1,maxRc) in self.poisonPos)
            boState2 = (self.cheesePos == (maxRc, 0)) \
                       and ((maxRc - 1, 0) in self.poisonPos and (maxRc, 1) in self.poisonPos)
            boState3 = (self.cheesePos == (maxRc, maxRc)) \
                       and ((maxRc - 1, maxRc) in self.poisonPos and (maxRc, maxRc - 1) in self.poisonPos)
            deadPath = True
            if not (deadPath in [boState0, boState1, boState2, boState3]):
                break
            print("dead path", [boState0, boState1, boState2, boState3])

    # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
    def _update_direction(self, action):
        if action == 0:
            self.direction = self.Dir_Up
        elif action == 1:
            self.direction = self.Dir_Left
        elif action == 2:
            self.direction = self.Dir_Right
        elif action == 3:
            self.direction = self.Dir_Down

    def render(self):
        print("render")

    def showInfo(self, tag):
        print(tag)
        print("cheese pos:", self.cheesePos)
        print("poison pos:", self.poisonPos)
        print("mouse pos:", self.mousePos)

if __name__ == '__main__':
    mgame = Mouse_Game()
    mgame.reset()