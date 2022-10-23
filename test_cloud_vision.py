import os
import io
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./keys/text-replace-366410-b1df0306b203.json"
from google.cloud import vision


client = vision.ImageAnnotatorClient()
# The name of the image file to annotate
file_name = os.path.abspath('./custom_feed/poster3.png')

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)

response = client.text_detection(image=image)
texts = response.text_annotations
print('Texts:')

for text in texts:
    print('\n"{}"'.format(text.description))

    vertices = (['({},{})'.format(vertex.x, vertex.y)
                for vertex in text.bounding_poly.vertices])

    print('bounds: {}'.format(','.join(vertices)))

if response.error.message:
    raise Exception(
        '{}\nFor more info on error messages, check: '
        'https://cloud.google.com/apis/design/errors'.format(
            response.error.message))
