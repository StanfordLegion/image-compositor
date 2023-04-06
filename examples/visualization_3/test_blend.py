import cv2
import numpy as np
import sys


def blend_binary(foreground, background, af, ab):
    # Convert uint8 to float
    foreground = np.copy(foreground.astype(float))
    background = np.copy(background.astype(float))

    # Normalize the alpha mask to keep intensity between 0 and 1
    af = af.astype(float)/255
    ab = ab.astype(float)/255
    ao = cv2.add(af, cv2.multiply(1.0 - af, ab))

    a3f = cv2.merge([af, af, af])
    a3b = cv2.merge([ab, ab, ab])
    a3o = cv2.merge([ao, ao, ao])

    # Alpha blending
    foreground = cv2.multiply(a3f, foreground)
    background = cv2.multiply(1.0 - a3f, cv2.multiply(a3b, background))
    output = np.divide(cv2.add(foreground, background), a3o, out=np.zeros_like(a3o), where=(a3o!=0))

    # output = cv2.add(foreground, cv2.multiply(1.0 - a3f, background))

    return output, ao*255


def load_dat(filename):
    with open(filename, 'rb') as file:
        _ = file.readline()
        header = file.readline().rstrip()
        _ = file.readline()
        w, h = header.split()
        w = int(w)
        h = int(h)
        color = np.ndarray((h, w, 3), dtype=np.uint8)
        alpha = np.ndarray((h, w, 1), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                r, g, b, a = np.frombuffer(file.read(4), dtype=np.uint8)
                color[h - 1 - y, x, :] = [b, g, r]
                alpha[h - 1 - y, x, :] = [a]
        _ = file.readline()
        _ = file.readline()
        d = float(file.readline().rstrip())
    return color.astype(float), alpha.astype(float), d, w, h


img, a, _, _, _ = load_dat(sys.argv[1])
for i in range(2, len(sys.argv)):    
    res = load_dat(sys.argv[i])
    img, a = blend_binary(img, res[0], a, res[1])

    # b, g, r = cv2.split(img)
    # cv2.imwrite('test' + str(i) + '.png', cv2.merge([b, g, r, a]))

b, g, r = cv2.split(img)
cv2.imwrite('test.png', cv2.merge([b, g, r, a]))
