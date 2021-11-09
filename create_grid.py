import cv2
import glob
import numpy as np
from utils import make_folder
color_masks = 'predictionimages-color/'
original = 'test_img/'

length = len(glob.glob('./predictionimages-color/*'))
make_folder('Final_grids')

i = 0
count = 0
flag = 1
while i < length:
    Vertical = []
    for j in range(2):
        Horizontal = []
        for k in range(3):
            print(count)
            if count < length:
                image = cv2.imread(original + str(count) + '.jpg')
                image2 = cv2.imread(color_masks + str(count) + '.png')    
                image=cv2.resize(image,(100,100))
                image2=cv2.resize(image2,(100,100))
                count = count + 1
                Horizontal.append(np.hstack([image,image2]))
            else:
                flag = 0
                break
        i = count
        if flag == 1:
            Vertical.append(np.vstack([Horizontal[0],Horizontal[1], Horizontal[2]]))
        else:
            break
    try:
        vertical2 = np.hstack([Vertical[0], Vertical[1]])
        cv2.imwrite('Final_grids/' + str(i)+'.png',vertical2)
    except:
        cv2.imwrite('Final_grids/' + str(i)+'.png',Vertical[0])




