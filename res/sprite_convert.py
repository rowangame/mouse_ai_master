import os.path

import cv2
import numpy as np

"""
缩放处理
@:return 图片数据
"""
def scaleSprite(img: np.ndarray, size: int):
    return cv2.resize(img, (size, size))

def convertSprites(size: int = 80):
    results = []
    imgNames = ("mouse.png", "cheese.jpg", "poison.png")
    for name in imgNames:
        absPath = os.path.abspath(__file__)
        # 取上一级目录
        imgPath = os.path.join(os.path.dirname(absPath), name)
        # print(imgPath)
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        sImg = scaleSprite(img, size)

        # savePath = "s_" + name.replace(".png", ".jpg")
        # cv2.imwrite(savePath, sImg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        results.append(sImg)
    return results

convertSprites(80)