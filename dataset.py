import re
from os import getcwd, listdir, path
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from constants import PLAY_AREA_CAPTURE_PARAMS, GAME_CURSOR, BUTTON_CLICKED_COLOR, BUTTON_CAPTURE_WIDTH, BUTTON_CAPTURE_HEIGHT, FINAL_RESIZE_PERCENT, PLAY_AREA_WIDTH_HEIGHT
from windows import derive_capture_params

trans = transforms.ToTensor()

KEY_STATES = {
    "00": 0,
    "01": 1,
    "10": 2,
}


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


def get_cursor_position(play_field: np.array):
    result = cv2.matchTemplate(
        play_field, GAME_CURSOR, cv2.TM_SQDIFF)  # , None, GAME_CURSOR)

    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)

    _min_val, _max_val, min_loc, _max_loc = cv2.minMaxLoc(result, None)

    return min_loc + np.array([int(len(GAME_CURSOR[0]) / 2), int(len(GAME_CURSOR[1]) / 2)])


def get_buttons_state(image):
    osu_buttons = image[960:960 +
                        BUTTON_CAPTURE_HEIGHT, 1760:1760 + (BUTTON_CAPTURE_WIDTH * 2)].copy()

    half_b_h = int((BUTTON_CAPTURE_HEIGHT / 2))
    half_b_w = int((BUTTON_CAPTURE_WIDTH / 2))

    capture_area_l = 10

    osu_left_button = osu_buttons[half_b_h - capture_area_l:half_b_h +
                                  capture_area_l,
                                  half_b_w - capture_area_l:half_b_w + capture_area_l]
    osu_right_button = osu_buttons[half_b_h - capture_area_l:half_b_h + capture_area_l, half_b_w +
                                   BUTTON_CAPTURE_WIDTH - capture_area_l:half_b_w + BUTTON_CAPTURE_WIDTH + capture_area_l]

    left_press_state, right_press_state = int(get_press_state('left',
                                                              osu_left_button)), int(
        get_press_state('right', osu_right_button))

    return KEY_STATES.get(
        f"{left_press_state}{right_press_state}", "invalid")


def extract_data_from_image(image_path):
    global delta_storage

    osu_screenshot = cv2.imread(image_path, cv2.IMREAD_COLOR)

    key_state = get_buttons_state(osu_screenshot)

    osu_play_area = osu_screenshot[
        PLAY_AREA_CAPTURE_PARAMS[3]:PLAY_AREA_CAPTURE_PARAMS[3] + PLAY_AREA_CAPTURE_PARAMS[1],
        PLAY_AREA_CAPTURE_PARAMS[2]:PLAY_AREA_CAPTURE_PARAMS[2] + PLAY_AREA_CAPTURE_PARAMS[0]].copy()

    resized_play_area = cv2.resize(osu_play_area, (int(PLAY_AREA_CAPTURE_PARAMS[0] * FINAL_RESIZE_PERCENT), int(
        PLAY_AREA_CAPTURE_PARAMS[1] * FINAL_RESIZE_PERCENT)), interpolation=cv2.INTER_LINEAR)

    resized_play_area_grey = cv2.cvtColor(
        resized_play_area, cv2.COLOR_BGR2GRAY)

    # print("Key State:", key_state)
    # cv2.imshow(
    #     f"debug", resized_play_area_grey)
    # cv2.waitKey(0)

    # print(PLAY_AREA_WIDTH_HEIGHT)
    if key_state == 'invalid':
        delta_storage = {}
        return None

    return [np.stack([resized_play_area_grey, resized_play_area_grey, resized_play_area_grey], axis=-1), key_state, get_cursor_position(osu_play_area) / PLAY_AREA_WIDTH_HEIGHT]


class OsuDataset(torch.utils.data.Dataset):

    def __init__(self, project_name: str, frame_latency=3, train_actions=True, force_rebuild=False) -> None:
        self.project_path = path.normpath(
            path.join(getcwd(), 'data', 'raw', project_name))
        self.processed_data_path = path.normpath(path.join(
            getcwd(), 'data', 'processed', f"{project_name}.npy"))
        self.images = []
        self.labels = []
        self.is_actions = train_actions
        self.frame_latency = frame_latency

        if not force_rebuild and path.exists(self.processed_data_path):
            image_data, cursor_data, action_data = np.load(
                self.processed_data_path, allow_pickle=True)

            self.images = list(image_data)
            self.labels = list(action_data if self.is_actions else cursor_data)
        else:
            self.make_training_data()

        self.apply_frame_latency()

    def apply_frame_latency(self):
        for i in range(abs(self.frame_latency)):
            if(self.frame_latency > 0):
                self.labels.pop(0)
                self.images.pop()
            else:
                self.labels.pop()
                self.images.pop(0)

    def make_training_data(self):
        global delta_storage

        print('Generating data')

        delta_storage = {}

        images = listdir(self.project_path)

        images.sort(key=lambda x: int(
            re.search(r"(?:[a-zA-Z]?)([0-9]+).png", x).groups()[0]))

        image_data = []
        cursor_data = []
        action_data = []
        for img_path in tqdm(images):
            try:

                result = extract_data_from_image(
                    path.normpath(path.join(self.project_path, img_path)))

                if result is None:
                    continue

                image, key_state, cursor = result

                action_data.append(key_state)
                cursor_data.append(cursor)
                image_data.append(trans(image).numpy())

                # cv2.imshow("Test", image)
                # cv2.waitKey(0)

            except Exception as e:
                print('ERROR WHILE LOADING', img_path, e)

        np.save(self.processed_data_path, [
                image_data, cursor_data, action_data])
        self.images = image_data
        self.labels = action_data if self.is_actions else cursor_data

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)


# np.save('test_data.npy', extract_data_from_image(
#     "D:\Github\osu-ai\data\\raw\meaning-of-love-4.62\\755.png"))
