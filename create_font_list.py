import os


FONT_DIR = "./fonts"
OUT_PATH = "./fonts/font_list.txt"

font_list = os.listdir(FONT_DIR)
font_list = [f for f in font_list if ".ttf" in f]
print("Got", len(font_list), "fonts")

with open(OUT_PATH, "w") as f:
    for i, font in enumerate(font_list):
        f.write(f"{i}|{font}\n")

