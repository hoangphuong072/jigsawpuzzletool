import glob
import json
import os
import shutil
from json import JSONEncoder
from pathlib import Path

import numpy as np
from PIL import ImageDraw
from PIL import Image


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def get_list_file(directory):
    arr = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            arr.append((path, os.path.dirname(path), Path(path).stem))
            print(path)
    return arr


def get_list_image(directory, type="png"):
    files = glob.glob(directory + '/**/*.' + type, recursive=True)
    arr = []
    for filename in files:
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            arr.append((path, os.path.dirname(path), Path(path).stem))
            # print(path)
    return arr


def get_type_of_piece(img):
    sum(x.count(0) for x in img)
    for row in img:
        for col in row:
            if col.alpha == 0:
                return True
    return False


base_cell = 55
max_piece = 70
list_grid_file = get_list_file("../grid")
list_input_file = get_list_image("../input")

list_grid = []
for grid_file, _, name in list_grid_file:
    arr = np.loadtxt(grid_file,
                     delimiter=",", dtype=int)

    list_grid.append((f"{name}_0_normal", np.rot90(arr,k=0)))
    list_grid.append((f"{name}_0_left", np.fliplr(np.rot90(arr,k=0))))
    list_grid.append((f"{name}_0_top", np.flipud(np.rot90(arr,k=0))))
    list_grid.append((f"{name}_0_both", np.flip(np.rot90(arr,k=0))))

    list_grid.append((f"{name}_90_normal", np.rot90(arr,k=1)))
    list_grid.append((f"{name}_90_left", np.fliplr(np.rot90(arr,k=1))))
    list_grid.append((f"{name}_90_top", np.flipud(np.rot90(arr,k=1))))
    list_grid.append((f"{name}_90_both", np.flip(np.rot90(arr,k=1))))

for input_file, input_dir, input_name in list_input_file:
    # print(input_dir)
    # print(input_dir.replace("input","output"))
    output_image_dir = os.path.join(input_dir.replace("input", "output"), input_name)
    print("------------------------------------")
    print(output_image_dir)
    print(input_file)
    # continue
    if os.path.exists(output_image_dir):
        shutil.rmtree(output_image_dir)
    os.makedirs(output_image_dir)
    json_data = {}
    json_data['name'] = input_name

    with Image.open(input_file) as input:

        for grid_name, arr in list_grid:
            print(grid_name)
            output_grid_dir = os.path.join(output_image_dir, grid_name)
            os.mkdir(output_grid_dir)
            json_data['level'] = grid_name
            json_data['pieces'] = []

            for piece_index in range(1, max_piece):
                print(f"index : {piece_index}")
                # with Drawing() as draw:
                with Image.new(mode="RGBA",size=input.size) as output:

                    draw = ImageDraw.Draw(output)
                    top, left, right, bottom = 15, 15, 0, 0
                    has_shape = False
                    type_shape_flag = 0
                    type_note = []
                    for iy, ix in np.ndindex(arr.shape):
                        if int(arr[iy, ix]) == piece_index:
                            has_shape = True
                            top = min(top, iy)
                            bottom = max(bottom, iy)
                            left = min(left, ix)
                            right = max(right, ix)
                            # with input.copy() as clone:
                            clone = input.crop((base_cell * ix, base_cell * iy, base_cell*(ix+1), base_cell*(iy+1)))
                            color_atr  =   clone.getcolors(1000000)[-1]
                            if color_atr[1]==(0,0,0,0):
                                if color_atr[0]==base_cell*base_cell:
                                    type_shape_flag += 1
                                else:
                                    type_shape_flag += 10000
                            else:
                                type_shape_flag += 100
                            # print(clone.size)
                            # clone.show()
                            output.paste(clone,(base_cell * ix,base_cell * iy))
                                # draw.bitmap(base_cell * ix,base_cell * iy,
                                #
                                #            clone.convert("BMP"))
                    print(( left, top,right,
                           bottom))
                    if not has_shape:
                        print("no shape")
                        continue
                    piece = {}
                    piece['number'] = piece_index
                    piece['type_shape_flag'] = type_shape_flag
                    piece['type_note'] = type_note
                    if type_shape_flag < 100:
                        piece['type'] = "empty"
                    elif type_shape_flag < 10000:
                        piece['type'] = "dynamic"
                    else:
                        piece['type'] = "block"
                    piece['position'] = [left, top]
                    piece_matrix = arr.copy()[top:bottom + 1:1, left:right + 1:1]
                    piece_matrix[piece_matrix != piece_index] = 0
                    piece_matrix[piece_matrix == piece_index] = 1
                    piece['matrix'] = piece_matrix
                    json_data['pieces'].append(piece)

                    if piece['type'] != "empty":
                            # draw.draw(output)
                            print((base_cell * left, base_cell * top, base_cell * (right+1),
                                   base_cell * (bottom+1)))
                            output = output.crop((base_cell * left, base_cell * top, base_cell * (right+1),
                                         base_cell * (bottom+1)))
                            output.save(os.path.join(output_grid_dir, f'{piece_index}.png'))

            with open(os.path.join(output_grid_dir, "data.json"), "w") as file:
                json.dump(json_data, file, cls=NumpyArrayEncoder, indent=2)
