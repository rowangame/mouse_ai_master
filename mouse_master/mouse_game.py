
import random
import os

import cv2
import numpy as np
import imageio

from pym.mouse_ai_master.res import sprite_convert

from pygame import mixer

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
        self.sprites = None
        self.stepCnt = 0
        random.seed(seed)

        if not silent_mode:
            mixer.init()
            self.snd_victory = mixer.Sound("../sound/victory.wav")
            self.snd_fail = mixer.Sound("../sound/fail.wav")
            self.frames = []

    def reset(self):
        self._generateData()
        self.stepCnt = 0
        self.direction = self.Dir_Right
        if self.frames is not None:
            self.frames.clear()

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
        # print("render...")
        if self.silent_mode:
            return

        # 画背景图
        # bgColor = (255,255,255)
        # imgMap = np.zeros((self.board_size, self.board_size, 3), dtype=np.uint8)
        cellSize = 70
        bgValue = 255
        width, height = cellSize * self.board_size + 1, cellSize * self.board_size + 1
        channel = 3
        imgMap = np.full((width, height, channel), fill_value=bgValue, dtype=np.uint8)

        # 画边框(水平)
        borderColor = (140, 140, 140)
        for row in range(self.board_size):
            tmpY = row * cellSize
            cv2.line(imgMap, (0, tmpY), (width-1, tmpY), borderColor, 1)
        # 画边框(垂直)
        for col in range(self.board_size):
            tmpX = col * cellSize
            cv2.line(imgMap, (tmpX, 0), (tmpX, height-1), borderColor, 1)

        # 加载精灵(边距设置为1)
        sprSize = cellSize - 2
        if self.sprites == None:
            self.sprites = sprite_convert.convertSprites(sprSize)

        # 画精灵
        sprMouse = self.sprites[0]
        row, col = self.mousePos[0], self.mousePos[1]
        tmpX = col * cellSize + 1
        tmpY = row * cellSize + 1
        imgMap[tmpY:tmpY+sprSize,tmpX:tmpX+sprSize] = sprMouse[0:sprSize,0:sprSize]

        if self.mousePos == self.cheesePos:
            sprPoison = self.sprites[2]
            for tmpPos in self.poisonPos:
                row, col = tmpPos[0], tmpPos[1]
                tmpX = col * cellSize + 1
                tmpY = row * cellSize + 1
                imgMap[tmpY:tmpY + sprSize, tmpX:tmpX + sprSize] = sprPoison[0:sprSize, 0:sprSize]

                if not self.silent_mode:
                    self.snd_victory.play()
        elif self.mousePos in self.poisonPos:
            sprCheese = self.sprites[1]
            row, col = self.cheesePos[0], self.cheesePos[1]
            tmpX = col * cellSize + 1
            tmpY = row * cellSize + 1
            imgMap[tmpY:tmpY + sprSize, tmpX:tmpX + sprSize] = sprCheese[0:cellSize, 0:sprSize]

            sprPoison = self.sprites[2]
            for tmpPos in self.poisonPos:
                if tmpPos == self.mousePos:
                    if not self.silent_mode:
                        self.snd_fail.play()
                    continue
                row, col = tmpPos[0], tmpPos[1]
                tmpX = col * cellSize + 1
                tmpY = row * cellSize + 1
                imgMap[tmpY:tmpY + sprSize, tmpX:tmpX + sprSize] = sprPoison[0:sprSize, 0:sprSize]
        else:
            sprCheese = self.sprites[1]
            row, col = self.cheesePos[0], self.cheesePos[1]
            tmpX = col * cellSize + 1
            tmpY = row * cellSize + 1
            imgMap[tmpY:tmpY+sprSize,tmpX:tmpX+sprSize] = sprCheese[0:cellSize,0:sprSize]

            sprPoison = self.sprites[2]
            for tmpPos in self.poisonPos:
                row, col = tmpPos[0], tmpPos[1]
                tmpX = col * cellSize + 1
                tmpY = row * cellSize + 1
                imgMap[tmpY:tmpY + sprSize, tmpX:tmpX + sprSize] = sprPoison[0:sprSize, 0:sprSize]
        # imageio图片数据是bgr格式,需要转化一下
        self.frames.append(cv2.cvtColor(imgMap, cv2.COLOR_BGR2RGB))
        cv2.imshow("mouse-ai", imgMap)
        cv2.waitKey(350)

    def finalize(self, episode):
        if not self.silent_mode:
            if episode < 20:
                # 保存gif图片
                path = "./../out"
                if not os.path.exists(path):
                    os.mkdir(path)
                gifName = os.path.join(path, "episode-%d.gif" % episode)
                imageio.mimsave(gifName, self.frames, 'GIF', duration=350)

            # 关闭cv2窗口
            cv2.destroyAllWindows()

    def showInfo(self, tag):
        print(tag)
        print("cheese pos:", self.cheesePos)
        print("poison pos:", self.poisonPos)
        print("mouse pos:", self.mousePos)

if __name__ == '__main__':
    mgame = Mouse_Game(board_size=8, silent_mode=False)
    for i in range(20):
        mgame.reset()
        mgame.render()
    cv2.destroyAllWindows()