import re
from os import getcwd, listdir, path
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from windows import derive_capture_params

trans = transforms.ToTensor()

KEY_STATES = {
    "00": 0,
    "01": 1,
    "10": 2,
}

BUTTON_CLICKED_COLOR = np.array([254, 254, 220])
PLAY_AREA_CAPTURE_PARAMS = derive_capture_params()
FINAL_RESIZE_PERCENT = 0.3


def get_button_cell_color(img):
    return img[0][0]


def get_button_press_dist(button):
    return np.linalg.norm(get_button_cell_color(
        button) - BUTTON_CLICKED_COLOR)


# stores the key data for the last processed frame (assumes all input is sequential)
delta_storage = {}
# 0 = no change, -1 = decreasing, 1 = increasing


def get_press_state(unique_id, button):
    global delta_storage
    current_distance = get_button_press_dist(button)

    if delta_storage.get(unique_id, None) is None:
        delta_storage[unique_id] = [get_button_press_dist(button), 0]

    old_distance, old_dir = delta_storage.get(unique_id)

    diff = current_distance - old_distance
    current_dir = 1 if diff > 0 else (-1 if diff < 0 else 0)

    delta_storage[unique_id] = [current_distance, current_dir]

    if current_dir == 0 and current_distance < 80:
        return 1

    if current_dir > 0:  # and old_dir <= 0:
        return 0

    if current_dir < 0:  # and old_dir > 0:
        return 1

    return 0


def extract_data_from_image(image_path):
    BUTTON_CAPTURE_HEIGHT = 75
    BUTTON_CAPTURE_WIDTH = 46
    global delta_storage

    osu_screenshot = cv2.imread(image_path, cv2.IMREAD_COLOR)

    osu_buttons = osu_screenshot[960:960 +
                                 BUTTON_CAPTURE_HEIGHT, 1760:1760+(BUTTON_CAPTURE_WIDTH * 2)].copy()

    half_b_h = int((BUTTON_CAPTURE_HEIGHT / 2))
    half_b_w = int((BUTTON_CAPTURE_WIDTH / 2))

    capture_area_l = 10

    osu_left_button = osu_buttons[half_b_h - capture_area_l:half_b_h +
                                  capture_area_l, half_b_w - capture_area_l:half_b_w + capture_area_l]
    osu_right_button = osu_buttons[half_b_h - capture_area_l:half_b_h + capture_area_l, half_b_w +
                                   BUTTON_CAPTURE_WIDTH - capture_area_l:half_b_w + BUTTON_CAPTURE_WIDTH + capture_area_l]

    osu_play_area = osu_screenshot[PLAY_AREA_CAPTURE_PARAMS[3]:PLAY_AREA_CAPTURE_PARAMS[3] + PLAY_AREA_CAPTURE_PARAMS[1],
                                   PLAY_AREA_CAPTURE_PARAMS[2]:PLAY_AREA_CAPTURE_PARAMS[2]+PLAY_AREA_CAPTURE_PARAMS[0]].copy()

    left_press_state, right_press_state = int(get_press_state('left',
                                                              osu_left_button)), int(get_press_state('right', osu_right_button))

    osu_play_area = cv2.resize(osu_play_area, (int(PLAY_AREA_CAPTURE_PARAMS[0] * FINAL_RESIZE_PERCENT), int(
        PLAY_AREA_CAPTURE_PARAMS[1] * FINAL_RESIZE_PERCENT)), interpolation=cv2.INTER_LINEAR)

    key_state = KEY_STATES.get(
        f"{left_press_state}{right_press_state}", "invalid")

    # print(get_button_cell_color(osu_left_button),
    #       get_button_cell_color(osu_right_button), f"{left_press_state}{right_press_state}", delta_storage_snapshot, delta_storage)
    # cv2.imshow(
    #     f"debug", osu_screenshot)
    # cv2.waitKey(1)

    if key_state == 'invalid':
        delta_storage = {}
        return None

    return [osu_play_area, key_state]


class OsuDataset(torch.utils.data.Dataset):

    def __init__(self, project_name: str, use_clicks=True, force_rebuild=False) -> None:
        self.project_path = path.normpath(
            path.join(getcwd(), 'data', 'raw', project_name))
        self.processed_data_path = path.normpath(path.join(
            getcwd(), 'data', 'processed', f"{project_name}.npy"))
        self.images = []
        self.results = []

        if not force_rebuild and path.exists(self.processed_data_path):
            images, results = np.load(
                self.processed_data_path, allow_pickle=True)

            self.images = images
            self.results = results

        else:
            self.make_training_data()

    def make_training_data(self):
        global delta_storage

        print('Generating data')

        delta_storage = {}

        images = listdir(self.project_path)

        images.sort(key=lambda x: int(
            re.search(r"(?:[a-zA-Z]?)([0-9]+).png", x).groups()[0]))

        for img_path in tqdm(images):
            try:

                result = extract_data_from_image(
                    path.normpath(path.join(self.project_path, img_path)))

                if result is None:
                    continue

                image, action = result

                self.results.append(action)
                self.images.append(trans(image).numpy())

                # cv2.imshow("Test", image)
                # cv2.waitKey(0)

            except Exception as e:
                print('ERROR WHILE LOADING', img_path, e)

        # frame offset to account for replay/auto keyoverlay delay, should convert to parameter later
        frame_latency = 3
        for i in range(frame_latency):
            self.results.pop(0)
            self.images.pop()

        # old balancing code
        # classes_count = {}

        # for img_path in tqdm(images):
        #     try:

        #         result = extract_data_from_image(
        #             path.normpath(path.join(self.project_path, img_path)))

        #         if result is None:
        #             continue

        #         image, action = result

        #         if action not in classes_count.keys():
        #             classes_count[action] = []
        #         classes_count[action].append(image)

        #     except Exception as e:
        #         print('ERROR WHILE LOADING', img_path, e)

        # largest = max(list(map(lambda x: len(x), classes_count.values())))

        # for label, data in classes_count.items():
        #     total = len(data)
        #     if total < largest:
        #         for i in tqdm(range(largest - total), f'Balancing Class {label}'):
        #             classes_count[label].append(data[i % total])

        # for label, data in classes_count.items():
        #     for image in data:
        #         self.results.append(label)
        #         self.images.append(trans(image).numpy())

        np.save(self.processed_data_path, [self.images, self.results])

    def __getitem__(self, idx):
        return self.images[idx], self.results[idx]

    def __len__(self):
        return len(self.images)
