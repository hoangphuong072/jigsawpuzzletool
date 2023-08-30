import concurrent.futures
import glob
import inspect
import json
import os
import random
import shutil
import sys
import time
import traceback
from concurrent.futures import ALL_COMPLETED
from json import JSONEncoder
from pathlib import Path
from LinePrint import line_print as print

import PIL.ImageDraw
import numpy as np
from PIL import Image, ImageFont
from fontTools.ttLib import TTFont


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
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


def get_list_image(type="png"):
    # files = glob.glob(directory + '/**/*.' + type, recursive=True)
    arr = []
    for group in os.listdir(input_dir):
        group_path = os.path.join(input_dir, group)
        if os.path.isdir(group_path):
            for image in os.listdir(group_path):
                img_path = os.path.join(group_path, image)
                if Path(img_path).suffix == ".png":
                    img_name = Path(img_path).stem
                    output_path = os.path.join(output_dir, group, img_name)
                    output_test_image_path = os.path.join(output_test_dir, group)
                    stage_name = f"{group}_{img_name}"
                    arr.append((img_path, img_name, output_path, output_test_image_path, stage_name))
    # for filename in files:
    #     path = os.path.join(directory, filename)
    #     if os.path.isfile(path):
    #         arr.append((path, os.path.dirname(path), Path(path).stem))
    # print(path)
    return arr


def crop_img(input, piece):
    # print(piece)
    matrix = np.array(piece['matrix'])
    # print(piece['box'])

    box = tuple(x * base_cell for x in piece['box'])
    # print(box)

    output = input.crop(box)
    for iy, ix in search(matrix, 0):
        output.paste(transparent_cell, (ix * base_cell, iy * base_cell))
    output = output.convert("P", palette=Image.Palette.ADAPTIVE, colors=24)

    return output


def test_grid(pieces, image_dir, grid_name):
    scale_time = 5
    img_size_real = int(825/scale_time)
    cell_size = int(55/scale_time)
    num_frame = len(pieces)
    with Image.new(mode="RGBA", size=(img_size_real * num_frame, img_size_real + 7 * cell_size)) as result:
        draw = PIL.ImageDraw.Draw(result)
        font = ImageFont.truetype("../arial_narrow_7.ttf", size=250/scale_time)
        with Image.new(mode="RGBA", size=(img_size_real, img_size_real + 7 * cell_size)) as img:
            i = 0
            for piece in pieces:
                if piece['type'] in ["block", "dynamic"]:
                    # print(f"number is {piece['number']}")
                    with Image.open(os.path.join(image_dir, grid_name, str(piece["number"]) + ".png")) as piece_img:
                        piece_img = piece_img.resize((int(piece_img.size[0]/scale_time),int(piece_img.size[1]/scale_time)))
                        piece_img = piece_img.convert(mode="RGBA")
                        img.paste(piece_img, (cell_size * piece["position"][0], cell_size * piece["position"][1]),
                                  piece_img)

                        result.paste(img, (img_size_real * i, 7 * cell_size), img)
                        result.paste(piece_img, (img_size_real * i, 0), piece_img)
                        draw = PIL.ImageDraw.Draw(result)
                        draw.line((img_size_real * i, 0, img_size_real * i, img_size_real + 7 * cell_size), fill="black",
                                  width=2)
                        draw.text((img_size_real * (float(i) + 0.5), 0), str(piece['number']), fill="red", font=font)
                        i += 1

        draw.line((0, 7 * cell_size, img_size_real * num_frame, 7 * cell_size), fill="black", width=10)
        # result = result.resize((int(img_size_real * num_frame / 5), int((img_size_real + 7 * cell_size) / 5)))
        result.save(os.path.join(output_test_dir, f"{stage_name}_{grid_name}.png"), format="PNG", optimize=True)


def crop_arr(grid, left, top, right, bottom):
    return grid.copy()[top:bottom:1, left:right:1]


def create_cell_arr(img: Image.Image):
    arr = np.empty((15, 15), dtype=tuple)
    arr.fill((0, "empty"))

    for iy, ix in np.ndindex(arr.shape):
        clone: Image.Image = img.crop((base_cell * ix, base_cell * iy, base_cell * (ix + 1), base_cell * (iy + 1)))
        color_atr = clone.getcolors(1000000)[-1]
        if color_atr[1][0] == 0:
            not_transparent_count = base_cell * base_cell - color_atr[0]
            if not_transparent_count == 0:
                cell_type = "empty"
            else:
                cell_type = "half"
                if not_transparent_count >= 3021:
                    not_transparent_count = base_cell * base_cell
                    cell_type = "full"
        else:
            not_transparent_count = base_cell * base_cell
            cell_type = "full"

        top = sum(clone.getpixel((x, 0))[0] > 0 for x in range(0, 55))
        bottom = sum(clone.getpixel((x, 54))[0] > 0 for x in range(0, 55))
        left = sum(clone.getpixel((0, x))[0] > 0 for x in range(0, 55))
        right = sum(clone.getpixel((54, x))[0] > 0 for x in range(0, 55))
        arr[iy][ix] = (
            not_transparent_count, cell_type, {"top": top, "bottom": bottom, "left": left, "right": right}, (iy, ix))
    return arr


def merge_piece(arr, img_cell_arr, grid_name):
    has_change = False
    for piece_index in np.unique(arr):
        if piece_index <= 0: continue
        list_arr = search(arr, piece_index)
        if len(list_arr) == 0: continue
        piece = create_piece_json(piece_index, arr, img_cell_arr, grid_name)
        if piece['invalid']:
            for y, x in list_arr:
                arr[y, x] = 99
            has_change = True
    return has_change


def merge_cell(arr, cell_arr):
    has_change = False
    list_99 = search(arr, 99)
    for iy, ix in list_99:
        for key_max in ["top", "bottom", "left", "right"]:
            _iy, _ix = iy, ix

            if key_max == "top":
                _iy -= 1
            if key_max == "bottom":
                _iy += 1
            if key_max == "left":
                _ix -= 1
            if key_max == "right":
                _ix += 1

            if arr[iy, ix] != arr[_iy, _ix] and arr[_iy, _ix]!= -1:
                print(f"{key_max} in {cell_arr[iy, ix]} position {(iy, ix)} value {arr[iy, ix]} to {arr[_iy, _ix]}")
                arr[iy, ix] = arr[_iy, _ix]
                has_change = True
            if arr[iy, ix] != 99: break
    # print(list_99)
    list_half = list(filter(lambda x: cell_arr[x[0], x[1]][1] == "half", np.ndindex(arr.shape)))
    # print(list_half)
    # for x in list_half:
    #     if x not in list_99: list_99.append(x)
    for iy, ix in list_half:
        border = cell_arr[iy, ix][2]
        key_max = max(border, key=lambda x: border[x])
        # if cell_arr[iy, ix][1] == "full" and arr[iy, ix] == 99:
        #     key_max = random.choice(["top", "bottom", "left", "right"])
        # for _key_max in ["top", "bottom", "left", "right"]:
        #     if arr[iy, ix] == 99: key_max = _key_max
        _iy, _ix = iy, ix

        if key_max == "top":
            _iy -= 1
        if key_max == "bottom":
            _iy += 1
        if key_max == "left":
            _ix -= 1
        if key_max == "right":
            _ix += 1

        if arr[iy, ix] != arr[_iy, _ix]:
            print(f"{key_max} in {cell_arr[iy, ix]} position {(iy, ix)} value {arr[iy, ix]} to {arr[_iy, _ix]} ")
            arr[iy, ix] = arr[_iy, _ix]
            has_change = True
    return has_change


def split_piece(arr, cell_arr):
    has_change = False
    for piece_index in np.unique(arr):
        if piece_index <= 0 or piece_index != 99: continue
        list_arr = search(arr, piece_index)
        if len(list_arr) == 0: continue
        top = min(list_arr, key=lambda tup: tup[0])[0]
        bottom = max(list_arr, key=lambda tup: tup[0])[0]

        left = min(list_arr, key=lambda tup: tup[1])[1]
        right = max(list_arr, key=lambda tup: tup[1])[1]
        if right - left + 1 > 5:
            for y, x in list_arr:
                if x >= left + 3:
                    arr[y, x] += 100
                    print(f"change w {piece_index} {(x, y)}")
                    has_change = True
        if bottom - top + 1 > 7:
            for y, x in list_arr:
                if y >= top + 4:
                    arr[y, x] += 10000
                    print(f"change h {piece_index} {(x, y)}")
                    has_change = True

    return has_change


def save_result(string):
    with open(os.path.join(output_dir, "result.csv"), "a") as file:
        file.writelines(string + "\n")
        file.close()


def search(arr, value):
    result = []
    for iy, ix in np.ndindex(arr.shape):
        if arr[iy, ix] == value:
            result.append((iy, ix))
    return result


def create_piece_json(piece_index, arr, img_cell_arr, grid_name):
    piece = {'number': piece_index, 'type_shape_flag': 0, 'not_empty_count': 0,
             'not_transparent_pixel_count': 0, "invalid": False}
    list_arr = search(arr, piece_index)
    top = min(list_arr, key=lambda tup: tup[0])[0]
    bottom = max(list_arr, key=lambda tup: tup[0])[0]

    left = min(list_arr, key=lambda tup: tup[1])[1]
    right = max(list_arr, key=lambda tup: tup[1])[1]
    list_img_cell = [img_cell_arr.item(p) for p in list_arr]
    # print(list_img_cell)
    piece['box'] = (left, top, right + 1, bottom + 1)
    piece['not_transparent_pixel_count'] = sum(a[0] for a in list_img_cell)

    print(f"-------- piece {piece_index}---------")
    if 0 < piece['not_transparent_pixel_count'] < base_cell * base_cell * 3 and piece_index != -1:
        piece['invalid'] = True
        # print(
        # f"invalid piece {piece_index} pixel {piece['not_transparent_pixel_count']} image {input_dir}/{input_name} at {grid_name}")
    if piece['not_transparent_pixel_count'] == base_cell * base_cell * len(list_arr):
        piece['type'] = "dynamic"
    elif piece['not_transparent_pixel_count'] == 0:
        piece['type'] = "empty"
    else:
        piece['type'] = "block"
    piece['position'] = (left, top)
    piece_matrix = crop_arr(arr, *piece['box'])
    piece_matrix = np.where(piece_matrix == piece_index, 1, 0)
    piece['matrix'] = piece_matrix
    return piece


time_log = {}


def start_log(name):
    time_log[name] = time.time()


def end_log(name):
    print(f"{name} running : {time.time() - time_log[name]}")


base_cell = 55
transparent_cell = Image.new("RGBA", (base_cell, base_cell), color=(0, 0, 0, 0))
input_dir = "../input"
output_dir = "../output"
output_test_dir = "../test_output"
grid_dir = "../grid"

list_grid_file = get_list_file(grid_dir)
list_input_file = get_list_image()
# print(list_input_file)

list_grid = []
for grid_file, _, name in list_grid_file:
    arr = np.loadtxt(grid_file,
                     delimiter=",", dtype=int)
    name = name.lower()
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


def worker(img_path, img_name, output_path, output_test_image_path, stage_name):
    start_log("WORKER")
    print("------------------------------------")
    print(output_path)
    print(img_path)
    print(output_test_image_path)
    os.makedirs(output_path)

    with Image.open(img_path) as input:
        start_log("create_cell_arr")
        img_cell_arr = create_cell_arr(input)
        end_log("create_cell_arr")
        for grid_name, _arr in list_grid:
            start_log("GRID_TIME")

            arr = np.copy(_arr)
            print(grid_name)
            output_grid_dir = os.path.join(output_path, grid_name)
            os.makedirs(output_grid_dir)
            start_log("OPTIMAZE_GRID")

            for iy, ix in np.ndindex(arr.shape):
                if img_cell_arr[iy, ix][1] == "empty":
                    arr[iy, ix] = -1
            for _ in range(50):
                has_change = False
                for __ in range(30):
                    has_change |= merge_cell(arr, img_cell_arr)
                has_change |= merge_piece(arr, img_cell_arr, grid_name)
                has_change |= split_piece(arr, img_cell_arr)

                if not has_change:
                    print(f"optimaze grid {grid_name} success {_} ")
                    break
            else:
                print(f"optimaze grid {grid_name} fail")
            # for i in range(30):
            #     merge_piece(arr, img_cell_arr, grid_name)
            #     if not merge_cell(arr, img_cell_arr): break
            # if not split_piece(arr, img_cell_arr):
            #     break
            end_log("OPTIMAZE_GRID")

            start_log("create_piece_json")
            json_data = {'invalid': False, 'name': stage_name, 'level': grid_name, 'pieces': [], "grid": arr}
            for piece_index in np.unique(arr):
                piece = create_piece_json(piece_index, arr, img_cell_arr, grid_name)
                json_data['pieces'].append(piece)
                if piece['invalid']:
                    print(f" invalid grid {grid_name} in {stage_name} piece {piece_index}")
                    break

            end_log("create_piece_json")
            start_log("CROP_IMAGE")
            for piece in json_data['pieces']:
                # print(piece)
                if piece['invalid'] and False:
                    shutil.rmtree(output_grid_dir, ignore_errors=True)
                    print("remove invalid grid dir")
                    break
                if piece['type'] == "empty": continue
                output = crop_img(input, piece)
                output.save(os.path.join(output_grid_dir, f'{piece["number"]}.png'), format="PNG", optimize=True)
            else:
                with open(os.path.join(output_grid_dir, "data.json"), "w") as file:
                    json.dump(json_data, file, cls=NumpyArrayEncoder, indent=2)
                start_log("TEST")
                test_grid(json_data['pieces'], output_path, grid_name)
                end_log("TEST")
                save_result(f"{img_path},{grid_name}")
            end_log("CROP_IMAGE")

            end_log("GRID_TIME")

            # break
    end_log("WORKER")


shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir)
shutil.rmtree(output_test_dir, ignore_errors=True)
os.makedirs(output_test_dir)
start_log("TOTAL")
with concurrent.futures.ThreadPoolExecutor(max_workers=48) as executor:
    futures = []
    count = 0
    for img_path, img_name, output_path, output_test_image_path, stage_name in list_input_file:
        future = executor.submit(worker, img_path, img_name, output_path, output_test_image_path, stage_name)
        futures.append(future)
        # worker(img_path, img_name, output_path, output_test_image_path, stage_name)

        count += 1
        # if count >= 10000:
        if count >= 0:
            break

    concurrent.futures.wait(futures, return_when=ALL_COMPLETED)
    end_log("TOTAL")
    print("***finish")
