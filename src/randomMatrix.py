import json

import numpy as np


def crop_image(piece_obj,img):
        for piece in piece_obj['pieces']:
                print(piece)

#
# json_data = json.load(open("../output/7/7_11/hard3_0_left/data.json"))
# crop_image(json_data,None)


arr = np.array([[0,1,2],[3,4,5],[6,7,8]])
# arr.fill((0,"empty"))
print(arr)
for iy, ix in np.ndindex(arr.shape):
        print(f"{ix} - {iy} - {arr[iy,ix]} - {arr[ix,iy]}")
