
from rembg import remove
import cv2

def regbg_mouse():
    img = cv2.imread("mouse.jpg")
    imgRlt = remove(img)
    cv2.imwrite("./mouse-out.png", imgRlt)
    print("end...")


regbg_mouse()