import re
from os import getcwd, listdir, path
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from queue import Queue
from threading import Thread
from tqdm import tqdm
from ai.constants import PLAY_AREA_CAPTURE_PARAMS, GAME_CURSOR, BUTTON_CLICKED_COLOR, BUTTON_CAPTURE_WIDTH, BUTTON_CAPTURE_HEIGHT, FINAL_RESIZE_PERCENT, PLAY_AREA_WIDTH_HEIGHT

image_to_pytorch_image = transforms.ToTensor()

INVALID_KEY_STATE = "An Invalid State"

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

    return (min_loc + np.array([int(len(GAME_CURSOR[0]) / 2), int(len(GAME_CURSOR[1]) / 2)])) / np.array([len(play_field[0]), len(play_field[1])])


def get_buttons_state(image):
    global delta_storage
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

    final_state = KEY_STATES.get(
        f"{left_press_state}{right_press_state}", INVALID_KEY_STATE)

    if final_state == INVALID_KEY_STATE:
        delta_storage = {}
        final_state = '00'

    return KEY_STATES
    return


def get_resized_play_area(screenshot):
    play_area = screenshot[
        PLAY_AREA_CAPTURE_PARAMS[3]:PLAY_AREA_CAPTURE_PARAMS[3] + PLAY_AREA_CAPTURE_PARAMS[1],
        PLAY_AREA_CAPTURE_PARAMS[2]:PLAY_AREA_CAPTURE_PARAMS[2] + PLAY_AREA_CAPTURE_PARAMS[0]].copy()

    return cv2.resize(play_area, (int(PLAY_AREA_CAPTURE_PARAMS[0] * FINAL_RESIZE_PERCENT), int(
        PLAY_AREA_CAPTURE_PARAMS[1] * FINAL_RESIZE_PERCENT)), interpolation=cv2.INTER_LINEAR)


def transform_resized(image):
    grayed = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY)

    normalized = np.stack(
        [grayed, grayed, grayed], axis=-1) / 255

    return image_to_pytorch_image(normalized).numpy()


def extract_actions_from_image(osu_screenshot):
    # print("Key State:", key_state)

    # cv2.imshow(
    #     f"debug", cv2.circle(resized_play_area_grey, get_cursor_position(
    #         resized_play_area), 20, (0, 0, 255), 20))
    # cv2.waitKey(0)

    # print(PLAY_AREA_WIDTH_HEIGHT)
    return [transform_resized(get_resized_play_area(osu_screenshot)), get_buttons_state(osu_screenshot)]


def extract_aim_from_image(osu_screenshot):
    resized_play_area = get_resized_play_area(osu_screenshot)

    return [transform_resized(resized_play_area), get_cursor_position(resized_play_area)]


class ImageProcessor:
    def __init__(self, project_path, images) -> None:
        self.processed = []
        self.project_path = project_path
        self.images = images
        self.buff = Queue()
        self.loader = tqdm(total=len(images),
                           desc=f"Processing Screenshots :")
        self.disk_thread = Thread(
            target=self.load_images, daemon=True, group=None)

    def load_images(self):
        for image_path in self.images:
            self.buff.put(cv2.imread(
                path.join(self.project_path, image_path), cv2.IMREAD_COLOR))
        self.buff.put(None)

    def process_images(self, extract_actions=True):
        self.disk_thread.start()
        data = self.buff.get()
        while data is not None:
            self.processed.append(extract_actions_from_image(
                data) if extract_actions else extract_aim_from_image(data))
            self.loader.update()
            data = self.buff.get()
        return np.array(self.processed, dtype=object)


class OsuDataset(torch.utils.data.Dataset):

    def __init__(self, project_name: str, frame_latency=3, is_actions=True, force_rebuild=False) -> None:
        self.project = project_name

        self.project_path = path.join(getcwd(), 'data', 'raw', project_name)
        self.processed_data_path = path.join(getcwd(
        ), 'data', 'processed', f"{'actions_' if is_actions else 'aim_'}{project_name}.npy")
        self.images = []
        self.labels = []
        self.frame_latency = frame_latency
        self.is_actions = is_actions

        if not force_rebuild and path.exists(self.processed_data_path):
            loaded_data = np.load(
                self.processed_data_path, allow_pickle=True)
            self.images = list(loaded_data[:, 0])
            self.labels = list(loaded_data[:, 1])
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

        delta_storage = {}

        images = listdir(self.project_path)

        if self.is_actions:
            images.sort(key=lambda x: int(
                re.search(r"(?:[a-zA-Z]?)([0-9]+).png", x).groups()[0]))

        processor = ImageProcessor(self.project_path, images)

        processed_data = processor.process_images(self.is_actions)

        np.save(self.processed_data_path, processed_data)

        self.images = list(processed_data[:, 0])

        self.labels = list(processed_data[:, 1])

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)


# np.save('test_data.npy', extract_data_from_image(
#     "D:\Github\osu-ai\data\\raw\meaning-of-love-4.62\\755.png"))
