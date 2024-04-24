# # SVG to 3-point conversion

from svg.path import parse_path
from svg.path.path import Line
from xml.dom import minidom
import numpy as np
import scipy.ndimage
from bresenham import bresenham
from rdp import rdp


def mydrawPNG(vector_image, Side=500):
    raster_image = np.zeros((int(Side), int(Side)), dtype=np.float32)
    initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
    pixel_length = 0

    for i in range(0, len(vector_image)):
        if i > 0:
            if vector_image[i - 1, 2] == 1:
                initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

        cordList = list(bresenham(initX, initY, int(vector_image[i, 0]), int(vector_image[i, 1])))
        pixel_length += len(cordList)

        for cord in cordList:
            if (cord[0] > 0 and cord[1] > 0) and (cord[0] < Side and cord[1] < Side):
                raster_image[cord[1], cord[0]] = 255.0
        initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

    raster_image = scipy.ndimage.binary_dilation(raster_image) * 255.0
    print ('pixel length: ', pixel_length)
    return raster_image
def simplify_strokes(sketch, eps=2.0):
        simplified_stroke3 = []
        where_arr = np.where(sketch[:, 2] == 1)[0]
        for en,stroke in enumerate(np.split(sketch, (where_arr)[:-1])):
            points = np.array([(x, y) for x, y, _ in stroke[1:]])
            simplified_points = rdp(points, epsilon=eps)
            for i, (x, y) in enumerate(simplified_points):
                _, _, pen = stroke[1+i]
                simplified_stroke3.append((x, y, pen))
            simplified_stroke3.append((sketch[where_arr[en]][0], sketch[where_arr[en]][1], 1))
        return np.array(simplified_stroke3)
def to_stroke3(svg, eps=2.0):
    doc = minidom.parse(svg)
    path_strings = [path.getAttribute('d') for path
                in doc.getElementsByTagName('path')]
    doc.unlink()

    sketch = []

    for path_string in path_strings:
        path = parse_path(path_string)
        stroke = [] # list of 3-point lines
        for e in path:
            if isinstance(e, Line) or True:
                x0 = e.start.real
                y0 = e.start.imag
                x1 = e.end.real
                y1 = e.end.imag
                cordList = list(bresenham(int(x0), int(y0), int(x1), int(y1)))
                cordList = [[x, y, 0] for x, y in cordList]
                stroke.extend(cordList)
        stroke.append([x1, y1, 1])
        sketch.extend(stroke)


    sketch = np.array(sketch)
    sketch = simplify_strokes(sketch, eps=eps)
    
    sketch[:, 0] = sketch[:, 0] - np.min(sketch[:, 0])
    sketch[:, 1] = sketch[:, 1] - np.min(sketch[:, 1])

    sketch[:, [0]] = (sketch[:, [0]]/640)*256
    sketch[:, [1]] = (sketch[:, [1]]/480)*256
    return sketch

    

if __name__ == '__main__':
    sketch = to_stroke3('/home/sketchx/Datasets/sketches/airplane/n02691156_58-2.svg')
    print(sketch.shape)
    print(np.min(sketch[:, 0]), np.max(sketch[:, 0]))
    print(np.min(sketch[:, 1]), np.max(sketch[:, 1]))
    raster_image = mydrawPNG(sketch, Side=600)
    print ((raster_image==255).sum())
    import matplotlib.pyplot as plt
    plt.imshow(255.0 - raster_image, cmap='gray')
    plt.show()
    plt.imsave("test.png", 255.0 - raster_image, cmap='gray')

    np.where(sketch[:, 2] == 1)[0].shape