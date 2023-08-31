import concurrent.futures
import json
import os
import random
import shutil
from json import JSONEncoder
from pathlib import Path

import PIL.ImageDraw
import numpy as np
from PIL import Image, ImageFont, ImageOps

from LinePrint import line_print as print, end_log_by_pid, start_log_by_pid
from src.TimeTracking import start_tracking_time, print_tracking_time, end_tracking_time


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
    arr = []
    for group in os.listdir(input_dir):
        group_path = os.path.join(input_dir, group)
        if os.path.isdir(group_path):
            for image in os.listdir(group_path):
                img_path = os.path.join(group_path, image)
                if Path(img_path).suffix == "." + type:
                    img_name = Path(img_path).stem
                    output_path = os.path.join(output_dir, group, img_name)
                    output_test_image_path = os.path.join(output_test_dir, group)
                    stage_name = f"{group}_{img_name}"
                    arr.append((img_path, img_name, output_path, output_test_image_path, stage_name))
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
    img_size_real = int(825 / scale_time)
    cell_size = int(55 / scale_time)
    num_frame = len(pieces)
    with Image.new(mode="RGBA", size=(img_size_real * num_frame, img_size_real + 7 * cell_size)) as result:
        draw = PIL.ImageDraw.Draw(result)
        font = ImageFont.truetype("../arial_narrow_7.ttf", size=int(250 / scale_time))
        font_small = ImageFont.truetype("../arial_narrow_7.ttf", size=int(100 / scale_time))
        with Image.new(mode="RGBA", size=(img_size_real, img_size_real + 7 * cell_size)) as img:
            i = 0
            for piece in pieces:
                if piece['type'] in ["block", "dynamic"]:
                    # print(f"number is {piece['number']}")
                    with Image.open(os.path.join(image_dir, grid_name, str(piece["number"]) + ".png")) as piece_img:
                        piece_img = piece_img.resize(
                            (int(piece_img.size[0] / scale_time), int(piece_img.size[1] / scale_time)))
                        piece_img = piece_img.convert(mode="RGBA")
                        img.paste(piece_img, (cell_size * piece["position"][0], cell_size * piece["position"][1]),
                                  piece_img)

                        result.paste(img, (img_size_real * i, 7 * cell_size), img)
                        piece_img  = ImageOps.expand(piece_img,border=2,fill="red")
                        result.paste(piece_img, (img_size_real * i+5, 0), piece_img)
                        draw = PIL.ImageDraw.Draw(result)
                        draw.line((img_size_real * i, 0, img_size_real * i, img_size_real + 7 * cell_size),
                                  fill="black",
                                  width=2)
                        draw.text((img_size_real * (float(i) + 0.5), 0), str(piece['number']), fill="red", font=font)
                        draw.text((img_size_real * (float(i) + 0.9), 0), str(i + 1), fill="green", font=font_small)
                        i += 1

        draw.line((0, 7 * cell_size, img_size_real * num_frame, 7 * cell_size), fill="black", width=5)
        # result = result.resize((int(img_size_real * num_frame / 5), int((img_size_real + 7 * cell_size) / 5)))
        result.save(os.path.join(output_test_dir, f"{stage_name}_{grid_name}.png"), format="PNG", optimize=True)
        return os.path.join(output_test_dir, f"{stage_name}_{grid_name}.png")


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
        key_arr = ["top", "bottom", "left", "right"]
        random.shuffle(key_arr)
        for key_max in key_arr:
            _iy, _ix = iy, ix

            if key_max == "top":
                _iy -= 1
            if key_max == "bottom":
                _iy += 1
            if key_max == "left":
                _ix -= 1
            if key_max == "right":
                _ix += 1
            if 0 < _ix > 14 or 0 < _iy > 14: continue

            if arr[iy, ix] != arr[_iy, _ix] and arr[_iy, _ix] not in [-1, 99]:
                print(f"{key_max} in {cell_arr[iy, ix]} position {(iy, ix)} value {arr[iy, ix]} to {arr[_iy, _ix]}")
                arr[iy, ix] = arr[_iy, _ix]
                has_change = True
            if arr[iy, ix] != 99: break
    list_half = list(filter(lambda x: cell_arr[x[0], x[1]][1] == "half", np.ndindex(arr.shape)))
    for iy, ix in list_half:
        border = cell_arr[iy, ix][2]
        key_max = max(border, key=lambda x: border[x])
        _iy, _ix = iy, ix

        if key_max == "top":
            _iy -= 1
        if key_max == "bottom":
            _iy += 1
        if key_max == "left":
            _ix -= 1
        if key_max == "right":
            _ix += 1
        if 0 < _ix > 14 or 0 < _iy > 14: continue

        if arr[iy, ix] != arr[_iy, _ix] != 99:
            print(f"{key_max} in {cell_arr[iy, ix]} position {(iy, ix)} value {arr[iy, ix]} to {arr[_iy, _ix]} ")
            arr[iy, ix] = arr[_iy, _ix]
            has_change = True
    return has_change


def split_piece(arr, cell_arr):
    has_change = False
    for piece_index in np.unique(arr):
        if piece_index <= 0 or piece_index == 99: continue
        list_arr = search(arr, piece_index)
        if len(list_arr) == 0: continue
        top = min(list_arr, key=lambda tup: tup[0])[0]
        bottom = max(list_arr, key=lambda tup: tup[0])[0]

        left = min(list_arr, key=lambda tup: tup[1])[1]
        right = max(list_arr, key=lambda tup: tup[1])[1]
        width = right - left + 1
        height = bottom - top + 1
        if width > max_piece_width:
            for y, x in list_arr:
                if x >= left + random.randrange(2, width-2):
                    arr[y, x] += 100
                    print(f"change w {piece_index} ->{arr[y, x]} {(x, y)}")
                    has_change = True
        if height > max_piece_height:
            for y, x in list_arr:
                if y >= top + random.randrange(2, height-2):
                    arr[y, x] += 10000
                    print(f"change h {piece_index} ->{arr[y, x]} {(x, y)}")
                    has_change = True

    return has_change


def save_result(string):
    with open(os.path.join(output_dir, "result.csv"), "a") as file:
        file.writelines(string + "\n")
        file.close()
def save_html(string):
    with open(os.path.join(output_test_dir, "result.html"), "a") as file:
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


base_cell = 55
max_piece_width = 5
max_piece_height = 7
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
# list_grid = sorted(list_grid, key=lambda grid: grid[0])
# random.shuffle(list_grid)


# print(list_grid)
# exit()

def worker(img_path, img_name, output_path, output_test_image_path, stage_name):
    start_log_by_pid()
    start_tracking_time("WORKER")
    print("------------------------------------")
    print(output_path)
    print(img_path)
    print(output_test_image_path)
    os.makedirs(output_path)

    with Image.open(img_path) as input:
        start_tracking_time("create_cell_arr")
        img_cell_arr = create_cell_arr(input)
        print_tracking_time("create_cell_arr")
        used = []
        _list_grid = list_grid.copy()
        random.shuffle(_list_grid)
        for grid_name, _arr in _list_grid:

            if any([grid_name[:4] == name[0][:4] for name in used]): continue
            start_tracking_time("GRID_TIME")

            arr = np.copy(_arr)
            print(grid_name)
            output_grid_dir = os.path.join(output_path, grid_name)
            os.makedirs(output_grid_dir)
            start_tracking_time("OPTIMAZE_GRID")

            for iy, ix in np.ndindex(arr.shape):
                if img_cell_arr[iy, ix][1] == "empty":
                    arr[iy, ix] = -1
            for _ in range(50):
                print(f"optimaze grid {grid_name} time {_} ")
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
            print_tracking_time("OPTIMAZE_GRID")

            start_tracking_time("create_piece_json")
            json_data = {'invalid': False, 'name': stage_name, 'level': grid_name, 'pieces': [], "grid": arr}
            for piece_index in np.unique(arr):
                piece = create_piece_json(piece_index, arr, img_cell_arr, grid_name)
                json_data['pieces'].append(piece)
                json_data['invalid'] |= piece['invalid']
                if piece['invalid']:
                    print(f" invalid grid {grid_name} in {stage_name} piece {piece_index}")

                    break

            print_tracking_time("create_piece_json")
            start_tracking_time("CROP_IMAGE")
            if not json_data['invalid']:
                for piece in filter(lambda p: p['type'] != "empty", json_data['pieces']):
                    # print(piece)
                    # if piece['type'] == "empty": continue
                    output = crop_img(input, piece)
                    output.save(os.path.join(output_grid_dir, f'{piece["number"]}.png'), format="PNG", optimize=True)
                with open(os.path.join(output_grid_dir, "data.json"), "w") as file:
                    json.dump(json_data, file, cls=NumpyArrayEncoder, indent=2)
                start_tracking_time("TEST")
                test_img = test_grid(json_data['pieces'], output_path, grid_name)
                print_tracking_time("TEST")
                # save_result(f"{img_path},{grid_name}")
                used.append((grid_name,stage_name,test_img))
            else:
                shutil.rmtree(output_grid_dir, ignore_errors=True)
                print("remove invalid grid dir")
            print_tracking_time("CROP_IMAGE")

            print_tracking_time("GRID_TIME")

            # break
    print_tracking_time("WORKER", False)
    end_log_by_pid()
    # sorted(used)
    used.sort()
    return img_path, end_tracking_time('WORKER'), os.getpid(), used


shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir)
shutil.rmtree(output_test_dir, ignore_errors=True)
os.makedirs(output_test_dir)
start_tracking_time("TOTAL")

with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
    futures = []
    count = 0
    for img_path, img_name, output_path, output_test_image_path, stage_name in list_input_file:
        future = executor.submit(worker, img_path, img_name, output_path, output_test_image_path, stage_name)
        futures.append(future)
        # worker(img_path, img_name, output_path, output_test_image_path, stage_name)

        count += 1
        # if count >= 10000:
        if count >= 2:
            break

    concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
    total_time = 0
    result = {}
    for future in futures:
        try:
            img_path, time_proccess, pid, used = future.result()
            total_time += time_proccess
            print(f"{img_path} time proccess {time_proccess} , pid:{pid} ->{used}")
            result[img_path] = used

            # print(used)
        except Exception as e:
            print(e)
    result = dict(sorted(result.items()))
    for img_path in result.keys():
        for (grid_name,stage_name,test_img) in result[img_path]:
            save_result(f"{img_path},{grid_name}")
            save_html(f'<a href="{test_img}">{stage_name}_{grid_name}</a></br>')
    print_tracking_time("TOTAL")
    print(f"SUM TIME : {total_time}")
    print("***finish")

    executor.shutdown()
