from PIL import Image, ImageDraw, ImageFont
import numpy as np

text = "chào"

canvas_width = 308
canvas_height = 128
shape = (canvas_width, canvas_height)
padding = 0.1
border = int(min(shape) * padding)
target_shape = tuple(np.array(shape) - 2 * border)

fontsize = 12
pre_remain = None
while True:
    # get text bbox
    img_center = (canvas_width//2, canvas_height//2)
    img = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    myFont = ImageFont.truetype('./fonts/arial.ttf', fontsize)
    draw.text(img_center, text, font=myFont, fill=(255, 255, 255), anchor="mm")
    rect = img.getbbox()

    res_shape = (int(rect[2] - rect[0]), int(rect[3] - rect[1]))
    remain = np.min(np.array(target_shape) - np.array(res_shape))
    if pre_remain is not None:
        m = pre_remain * remain
        if m <= 0:
            if m < 0 and remain < 0:
                fontsize -= 1
            if m == 0 and remain != 0:
                if remain < 0:
                    fontsize -= 1
                elif remain > 0:
                    fontsize += 1
            break
    if remain < 0:
        if fontsize == 2:
            break
        fontsize -= 1
    else:
        fontsize += 1
    pre_remain = remain

print(fontsize)

img_center = (canvas_width//2, canvas_height//2)
img = Image.new('RGB', (canvas_width, canvas_height), (127, 127, 127))
draw = ImageDraw.Draw(img)

# Custom font style and font size
myFont = ImageFont.truetype('./fonts/arial.ttf', fontsize)
draw.text(img_center, text, font=myFont, fill=0, anchor="mm")
img.save("./examples/test.png")
