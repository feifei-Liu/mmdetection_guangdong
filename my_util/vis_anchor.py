import cv2
import numpy as np
import math
# bottom = (np.zeros((288,512,3)))
# anchors = [[15,19],
#            [18,61],
#            [20,160],
#            [24,29],
#            [30,51],
#            [37,978],
#            [40,22],
#            [290,70],
#            [2284,232]]
# x0 = 250
# y0 = 150
# for anchor in anchors:
#     w = anchor[0]
#     h = anchor[1]
#     cv2.rectangle(bottom, (int(x0 - w / 2), int(y0 - h / 2)), (int(x0 + w / 2), int(y0 + h / 2)), [0, 0, 255],1)
# cv2.imshow("img", bottom.astype(np.uint8))
# cv2.waitKey(-1)

def show_anchor_pixelsize(anchor_boxsizes,anchor_aspect_ratios,base_size):
    bottom = (np.zeros((800,1333,3)))
    x0 = 1333//2
    y0 = 800//2
    ws = []
    hs = []
    for i in range(len(anchor_aspect_ratios)):
        for j in range(len(anchor_boxsizes)):
            h = base_size * anchor_boxsizes[j] * np.sqrt(anchor_aspect_ratios[i])
            w = base_size * anchor_boxsizes[j] * np.sqrt(1. / anchor_aspect_ratios[i])
            ws.append(w)
            hs.append(h)
            print('\n')

            print('ratio ', anchor_aspect_ratios[i], '   anchor_boxsizes: ' , anchor_boxsizes[j])
            print('w: ', w, ' h: ',h ,'\n')
            cv2.rectangle(bottom, (int(x0-w/2), int(y0-h/2)), (int(x0+w/2), int(y0+h/2)), [0,0,255], 1)
            # cv2.rectangle(bottom, (int(x0+16-w/2), int(y0-h/2)), (int(x0+16+w/2), int(y0+h/2)), [0,255,0], 1)
    bottom = bottom.astype(np.uint8)
    bottom = cv2.resize(bottom,(1333,800))
    cv2.imshow("img",bottom.astype(np.uint8))
    cv2.waitKey(-1)

anchor_boxsizes = [4, 8, 16, 32, 64]
# anchor_boxsizes = [4]
anchor_aspect_ratios = [0.05,0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0,20]
# anchor_aspect_ratios = [0.5,1,2]
base_size = 4
show_anchor_pixelsize(anchor_boxsizes,anchor_aspect_ratios,base_size)

