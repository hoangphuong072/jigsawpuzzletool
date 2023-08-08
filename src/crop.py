import concurrent.futures
import glob
import json
import os
import shutil
from json import JSONEncoder
from pathlib import Path

import numpy as np
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
max_piece = 60
max_grid_length = 15

list_grid_file = get_list_file("../grid")
list_input_file = get_list_image("../input")

list_grid = []
for grid_file, _, name in list_grid_file:
    arr = np.loadtxt(grid_file,
                     delimiter=",", dtype=int)

    list_grid.append((f"{name}_0_normal", np.rot90(arr, k=0)))
    list_grid.append((f"{name}_0_left", np.fliplr(np.rot90(arr, k=0))))
    list_grid.append((f"{name}_0_top", np.flipud(np.rot90(arr, k=0))))
    list_grid.append((f"{name}_0_both", np.flip(np.rot90(arr, k=0))))

    list_grid.append((f"{name}_90_normal", np.rot90(arr, k=1)))
    list_grid.append((f"{name}_90_left", np.fliplr(np.rot90(arr, k=1))))
    list_grid.append((f"{name}_90_top", np.flipud(np.rot90(arr, k=1))))
    list_grid.append((f"{name}_90_both", np.flip(np.rot90(arr, k=1))))
# print(list_grid)
list_grid = sorted(list_grid, key=lambda grid: grid[0])


# print(list_grid)
# exit()
def worker(input_file, input_dir, input_name):
    output_image_dir = os.path.join(input_dir.replace("input", "output"), input_name)
    print("------------------------------------")
    print(output_image_dir)
    print(input_file)
    # continue
    # if os.path.exists(output_image_dir):
    shutil.rmtree(output_image_dir, ignore_errors=True)
    os.makedirs(output_image_dir)

    with Image.open(input_file) as input:
        input_arr = np.zeros((15, 15), dtype=tuple)
        for iy, ix in np.ndindex(input_arr.shape):
            clone = input.crop((base_cell * ix, base_cell * iy, base_cell * (ix + 1), base_cell * (iy + 1)))
            color_atr = clone.getcolors(1000000)[-1]
            plus = 0
            if color_atr[1] == (0, 0, 0, 0):
                if color_atr[0] == base_cell * base_cell:
                    plus = 1
                else:
                    plus = 10000
            else:
                plus = 100
            input_arr[iy][ix] = (plus, clone)
        for grid_name, arr in list_grid:
            print(grid_name)
            output_grid_dir = os.path.join(output_image_dir, grid_name)
            os.makedirs(output_grid_dir)
            json_data = {'invalid': False, 'name': input_name, 'level': grid_name, 'pieces': []}

            for piece_index in range(1, max_piece):
                # print(f"index : {piece_index}")
                with Image.new(mode="RGBA", color=4, size=input.size) as output:

                    top, left, right, bottom = max_grid_length, max_grid_length, 0, 0
                    piece = {'number': piece_index, 'type_shape_flag': 0, 'not_empty_count': 0}
                    for iy, ix in np.ndindex(arr.shape):
                        if int(arr[iy, ix]) == piece_index:
                            top = min(top, iy)
                            bottom = max(bottom, iy)
                            left = min(left, ix)
                            right = max(right, ix)
                            if input_arr[iy][ix][0] >= 100:
                                piece['not_empty_count'] += 1
                            piece['type_shape_flag'] += input_arr[iy][ix][0]
                            output.paste(input_arr[iy][ix][1], (base_cell * ix, base_cell * iy))
                    if piece['not_empty_count'] == 1:
                        json_data['invalid'] = True
                        break
                    if piece['type_shape_flag'] < 100:
                        piece['type'] = "empty"
                    elif piece['type_shape_flag'] < 10000:
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
                        output = output.crop((base_cell * left, base_cell * top, base_cell * (right + 1),
                                              base_cell * (bottom + 1)))
                        output = output.convert("P", palette=Image.Palette.ADAPTIVE, colors=24)
                        # output = output.convert("P", palette=Image.WEB, colors=24)
                        # output = output.convert("RGBA", colors=24)
                        output.save(os.path.join(output_grid_dir, f'{piece_index}.png'), format="PNG", optimize=True)

            with open(os.path.join(output_grid_dir, "data.json"), "w") as file:
                json.dump(json_data, file, cls=NumpyArrayEncoder, indent=2)
            print("done")
            if json_data['invalid'] == True:
                shutil.rmtree(output_grid_dir, ignore_errors=True)
                print("remove invalid grid")

    # print(input_dir)
    # print(input_dir.replace("input","output"))


with concurrent.futures.ThreadPoolExecutor(max_workers=500) as executor:
    futures = []
    for input_file, input_dir, input_name in list_input_file:
        # future = executor.submit(worker, input_file, input_dir, input_name)
        worker(input_file, input_dir, input_name)
        break
    concurrent.futures.wait(futures)
    print("finish")
