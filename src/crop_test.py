import glob
import json
import os
from pathlib import Path

from PIL import Image


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

# list_image = get_list_image("../out/1/1")
# print(list_image)
base_cell = 55
image_dir = "../output/7/7_11/easy2_0_both/"
with Image.new(mode="RGBA",size=(825,825)) as img:
    json_data = json.load(open(image_dir+"data.json"))
    for piece in json_data['pieces']:
        if piece['type'] in ["block","dynamic"]:
            print(f"number is {piece['number']}")
            with Image.open(image_dir+str(piece["number"])+".png") as piece_img:
                piece_img = piece_img.convert(mode="RGBA")
                img.paste(piece_img,(base_cell * piece["position"][0],base_cell * piece["position"][1]))

            img.show(f"piece {piece['number']}")
