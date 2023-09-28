import concurrent.futures
import json
import math
import os
import random
import shutil
import sys
from json import JSONEncoder
from pathlib import Path

import PIL.ImageDraw
import numpy as np
from PIL import Image, ImageFont, ImageOps

from LinePrint import line_print as print, end_log_by_pid, start_log_by_pid
sys.path.insert(0,"../src")
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
    num_col = 8
    img_size_real = int(825 / scale_time)
    cell_size = int(55 / scale_time)
    list_pieces = list(filter(lambda p: p['type'] in ["block", "dynamic"], pieces))
    font = ImageFont.truetype("../arial_narrow_7.ttf", size=int(100 / scale_time))
    font_small = ImageFont.truetype("../arial_narrow_7.ttf", size=int(80 / scale_time))
    with Image.new(mode="RGBA", size=(
            img_size_real * num_col,
            (img_size_real + 7 * cell_size) * math.ceil(len(list_pieces) / num_col))) as result:
        with Image.new(mode="RGBA", size=(img_size_real, img_size_real)) as img:
            for i, piece in enumerate(list_pieces):
                ofset_x = i % num_col * img_size_real
                ofset_y = int(i / num_col) * (img_size_real + 7 * cell_size)
                with Image.new(mode="RGBA", size=(img_size_real, 7 * cell_size)) as header:
                    with Image.open(os.path.join(image_dir, grid_name, str(piece["number"]) + ".png")) as piece_img:
                        piece_img = piece_img.resize(
                            (int(piece_img.size[0] / scale_time), int(piece_img.size[1] / scale_time)))
                        piece_img = piece_img.convert(mode="RGBA")
                        img.paste(piece_img, (cell_size * piece["position"][0], cell_size * piece["position"][1]),
                                  piece_img)

                        draw = PIL.ImageDraw.Draw(header)
                        draw.text((img_size_real * 0.5, 0), str(piece['number']), fill="red", font=font)
                        draw.text((img_size_real * 0.9, 0), str(i + 1), fill="green", font=font_small)

                        piece_img2 = ImageOps.expand(piece_img, border=1, fill="red")
                        header.paste(piece_img2, (5, 5), piece_img2)

                        img2 = ImageOps.expand(img, border=1, fill="black")
                        header2 = ImageOps.expand(header, border=1, fill="black")
                        result.paste(header2, (ofset_x, ofset_y), header2)
                        result.paste(img2, (ofset_x, ofset_y + header2.height - 2), img2)

        ImageOps.expand(result, border=1, fill="black").save(
            os.path.join(output_test_dir, f"{stage_name}_{grid_name}.png"), format="PNG", optimize=True)
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
        if piece_index in [-1, 99]: continue
        list_arr = search(arr, piece_index)
        if len(list_arr) == 0: continue
        piece = create_piece_json(piece_index, arr, img_cell_arr, grid_name)
        if piece['invalid_pixel']:
            print(piece)
            for y, x in list_arr:
                print(f"merge_piece {img_cell_arr[y, x][0]} {(y, x)} from {arr[y, x]} to 99")
                if arr[y, x] != -1 and img_cell_arr[y, x][0] != 0:
                    arr[y, x] = 99
            has_change = True
            break
    return has_change


def merge_cell_99(arr, cell_arr):
    has_change = False
    last_list = -1
    for _ in range(50):
        list_99 = search(arr, 99)
        print(f"list 99 size {list_99.__len__()} -> {list_99}")
        if len(list_99) == 0 or len(list_99) == last_list: break
        last_list = len(list_99)
        list_99.sort(
            key=lambda x: cell_arr[x[0], x[1]][0],
            reverse=True)
        for iy, ix in list_99:
            # top = min(list_99, key=lambda tup: tup[0])[0]
            # bottom = max(list_99, key=lambda tup: tup[0])[0]
            # left = min(list_99, key=lambda tup: tup[1])[1]
            # right = max(list_99, key=lambda tup: tup[1])[1]
            if (iy * 100 + ix) not in change_arr_history:
                change_arr_history[iy * 100 + ix] = []
            border = cell_arr[iy, ix][2].copy()
            print(border)
            border = [
                (border["top"], 0, iy - 1, ix, "top"),
                (border["bottom"], 0, iy + 1, ix, "bottom"),
                (border["left"], 0, iy, ix - 1, "left"),
                (border["right"], 0, iy, ix + 1, "right")]
            print(border)
            for b in border:
                if 0 <= b[2] < 15 and 0 <= b[3] < 15:
                    print(f"{b} -> {arr[b[2], b[3]]}")
                else:
                    print(f"{b} -> {(b[2], b[3])}")
            border = list(
                filter(
                    lambda d: 0 <= d[2] < 15
                              and 0 <= d[3] < 15
                              # and arr[d[2], d[3]] not in change_arr_history[iy * 100 + ix][-3:]
                              and arr[d[2], d[3]] != arr[iy, ix]
                              and arr[d[2], d[3]] not in [-1, 99],
                    border))
            border.sort(key=lambda d: d[0], reverse=True)
            print(
                f"border  size {border.__len__()} -> {border} ->{change_arr_history[iy * 100 + ix][-3:]} ->{arr[iy, ix]}")

            if len(border) > 0:
                print(change_arr_history[iy * 100 + ix][-3:])
                print(
                    f" merge 99 {border[0][4]} in {cell_arr[iy, ix]} position {(iy, ix)} value {arr[iy, ix]} to {arr[border[0][2], border[0][3]]} ")
                arr[iy, ix] = arr[border[0][2], border[0][3]]
                change_arr_history[iy * 100 + ix].append(arr[iy, ix])
                has_change = True
    return has_change

def merge_cell_98(arr, cell_arr):
    has_change = False
    last_list = -1
    for _ in range(100):
        list_99 = search(arr, 98)
        print(f"list 98 size {list_99.__len__()} -> {list_99}")
        if len(list_99) == 0 or len(list_99) == last_list: break
        last_list = len(list_99)
        list_99.sort(
            key=lambda x: cell_arr[x[0], x[1]][0],
            reverse=True)
        for iy, ix in list_99:
            if (iy * 100 + ix) not in change_arr_history:
                change_arr_history[iy * 100 + ix] = []
            border = cell_arr[iy, ix][2].copy()
            print(border)
            border = [
                (border["top"], 0, iy - 1, ix, "top"),
                (border["bottom"], 0, iy + 1, ix, "bottom"),
                (border["left"], 0, iy, ix - 1, "left"),
                (border["right"], 0, iy, ix + 1, "right")]
            print(border)
            for b in border:
                if 0 <= b[2] < 15 and 0 <= b[3] < 15:
                    print(f"{b} -> {arr[b[2], b[3]]}")
                else:
                    print(f"{b} -> {(b[2], b[3])}")
            border = list(
                filter(
                    lambda d: 0 <= d[2] < 15
                              and 0 <= d[3] < 15
                              # and arr[d[2], d[3]] not in change_arr_history[iy * 100 + ix][-3:]
                              and arr[d[2], d[3]] != arr[iy, ix]
                              and arr[d[2], d[3]] not in [-1],
                    border))
            border.sort(key=lambda d: d[0], reverse=True)
            print(
                f"border  size {border.__len__()} -> {border} ->{change_arr_history[iy * 100 + ix][-3:]} ->{arr[iy, ix]}")

            if len(border) > 0:
                print(change_arr_history[iy * 100 + ix][-3:])
                print(
                    f" merge 98 {border[0][4]} in {cell_arr[iy, ix]} position {(iy, ix)} value {arr[iy, ix]} to {arr[border[0][2], border[0][3]]} ")
                arr[iy, ix] = arr[border[0][2], border[0][3]]
                # change_arr_history[iy * 100 + ix].append(arr[iy, ix])
                has_change = True
    return has_change


def merge_cell(arr, cell_arr):
    has_change = False

    list_half = list(filter(lambda x: cell_arr[x[0], x[1]][1] == "half", np.ndindex(arr.shape)))
    print("Dsadsadsa")
    # print([cell_arr[y, x] for (y, x) in list_half])
    # print([arr[y, x] for (y, x) in list_half])
    for iy, ix in list_half:
        if arr[iy,ix]==99:continue
        if (iy * 100 + ix) not in change_arr_history:
            change_arr_history[iy * 100 + ix] = []
        border = cell_arr[iy, ix][2].copy()
        border = [
            (border["top"], 0, iy - 1, ix, "top"),
            (border["bottom"], 0, iy + 1, ix, "bottom"),
            (border["left"], 0, iy, ix - 1, "left"),
            (border["right"], 0, iy, ix + 1, "right")]
        border = list(
            filter(
                lambda d: 0 <= d[2] < 15
                          and 0 <= d[3] < 15
                          and arr[d[2], d[3]] not in change_arr_history[iy * 100 + ix][-3:]
                          and arr[d[2], d[3]] != arr[iy, ix]
                          and arr[d[2], d[3]] not in [-1, 99],
                border))
        border.sort(key=lambda d: d[0], reverse=True)
        if len(border) > 0:
            print(f" merge {border[0][4]} in {cell_arr[iy, ix]} position {(iy, ix)} value {arr[iy, ix]} to {arr[border[0][2], border[0][3]]} ")
            # arr[iy, ix] = arr[border[0][2], border[0][3]]
            arr[iy, ix] = 99
            change_arr_history[iy * 100 + ix].append(arr[iy, ix])
            has_change = True
    return has_change


change_arr_history = {}


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
            ofset = random.randrange(3, max_piece_width)
            increase = random.randrange(100, 9999)
            for y, x in list_arr:
                if x >= left + ofset:
                    arr[y, x] += increase
                    print(f"change w {piece_index} ->{arr[y, x]} {(y, x)}")
                    has_change = True
        elif height > max_piece_height:
            ofset = random.randrange(3, max_piece_height)
            increase = random.randrange(10000, 9000000)
            for y, x in list_arr:
                if y >= top + ofset:
                    arr[y, x] += increase
                    print(f"change h {piece_index} ->{arr[y, x]} {(y, x)}")
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
             'not_transparent_pixel_count': 0, "invalid_w_h": False, "invalid_pixel": False}
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
    if piece_index != -1:
        if 0 < piece['not_transparent_pixel_count'] < base_cell * base_cell * 3:
            piece['invalid_pixel'] = True
        if bottom - top + 1 > max_piece_height or right - left + 1 > max_piece_width:
            piece['invalid_w_h'] = True

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
        for _grid_name, _arr in _list_grid:
            used_count = len(list(filter(lambda name: _grid_name[:4] == name[0][:4], used)))
            grid_name = f"{_grid_name[:4]}{used_count + 1}"
            if used_count >= 3: continue
            start_tracking_time("GRID_TIME")

            arr = np.copy(_arr)
            print(grid_name)
            output_grid_dir = os.path.join(output_path, grid_name)
            os.makedirs(output_grid_dir)
            start_tracking_time("OPTIMAZE_GRID")

            for iy, ix in np.ndindex(arr.shape):
                if img_cell_arr[iy, ix][1] == "empty":
                    arr[iy, ix] = -1
                elif img_cell_arr[iy, ix][1] == "half":
                    arr[iy, ix] = 98
            merge_cell_98(arr, img_cell_arr)
            for _ in range(10):
                print(f"optimaze grid {grid_name} time {_} ")
                has_change = False
                # for __ in range(100):
                # has_change |= merge_cell(arr, img_cell_arr)
                has_change |= merge_piece(arr, img_cell_arr, grid_name)
                has_change |= merge_cell_99(arr, img_cell_arr)
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
                json_data['invalid'] |= piece['invalid_pixel'] | piece['invalid_w_h']
                if json_data['invalid']:
                    print(f" invalid grid {grid_name} in {stage_name} piece {piece_index}")
                    print(piece)
                    break

            print_tracking_time("create_piece_json")
            start_tracking_time("CROP_IMAGE")
            if not json_data['invalid']:
                for piece in filter(lambda p: p['type'] != "empty", json_data['pieces']):
                    output = crop_img(input, piece)
                    output.save(os.path.join(output_grid_dir, f'{piece["number"]}.png'), format="PNG",
                                optimize=True)
                with open(os.path.join(output_grid_dir, "data.json"), "w") as file:
                    json.dump(json_data, file, cls=NumpyArrayEncoder, indent=2)
                start_tracking_time("TEST")
                test_img = test_grid(json_data['pieces'], output_path, grid_name)
                print_tracking_time("TEST")
                used.append((grid_name, stage_name, test_img))
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

with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
    futures = []
    count = 0
    for img_path, img_name, output_path, output_test_image_path, stage_name in list_input_file:
        future = executor.submit(worker, img_path, img_name, output_path, output_test_image_path, stage_name)
        futures.append(future)
        # worker(img_path, img_name, output_path, output_test_image_path, stage_name)

        count += 1
        # if count >= 10000:
        if count >= 700:
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
        save_html(f'<h1>{stage_name}</h1></br>')
        for (grid_name, stage_name, test_img) in result[img_path]:
            save_result(f"{img_path},{grid_name}")
            save_html(f'<a href="{test_img}">{stage_name}_{grid_name}</a></br></br>')
            save_html(f'<img src="{test_img}"/></br>')
    print_tracking_time("TOTAL")
    print(f"SUM TIME : {total_time}")
    print("***finish")

    executor.shutdown()
