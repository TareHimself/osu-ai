

from os import listdir, path

import cv2


for img_path in listdir('data/raw/osu'):
    try:
        cv2.imwrite(path.join(
            'data/raw/osu-color', img_path), cv2.cvtColor(cv2.imread(path.join(
                'data/raw/osu', img_path), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB))

        print('Fixed', img_path)
    except Exception as e:
        print('ERROR WHILE LOADING', img_path, e)
