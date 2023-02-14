import re
from os import getcwd, listdir, path
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from queue import Queue
from threading import Thread
from tqdm import tqdm
from constants import PLAY_AREA_CAPTURE_PARAMS, GAME_CURSOR, BUTTON_CLICKED_COLOR, BUTTON_CAPTURE_WIDTH, BUTTON_CAPTURE_HEIGHT, FINAL_RESIZE_PERCENT, PLAY_AREA_WIDTH_HEIGHT

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
        play_field, GAME_CURSOR, cv2.TM_SQDIFF)

    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)

    _min_val, _max_val, min_loc, _max_loc = cv2.minMaxLoc(result, None)

    return (min_loc + np.array([int(len(GAME_CURSOR[0]) / 2), int(len(GAME_CURSOR[1]) / 2)]))


def get_buttons_from_screenshot(screenshot):
    return screenshot[960:960 +
                      BUTTON_CAPTURE_HEIGHT, 1760:1760 + (BUTTON_CAPTURE_WIDTH * 2)].copy()


def get_buttons_state(osu_buttons):
    global delta_storage

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
        final_state = 0

    return final_state


def get_resized_play_area(screenshot):
    play_area = screenshot[
        PLAY_AREA_CAPTURE_PARAMS[3]:PLAY_AREA_CAPTURE_PARAMS[3] + PLAY_AREA_CAPTURE_PARAMS[1],
        PLAY_AREA_CAPTURE_PARAMS[2]:PLAY_AREA_CAPTURE_PARAMS[2] + PLAY_AREA_CAPTURE_PARAMS[0]].copy()

    return cv2.resize(play_area, (int(PLAY_AREA_CAPTURE_PARAMS[0] * FINAL_RESIZE_PERCENT), int(
        PLAY_AREA_CAPTURE_PARAMS[1] * FINAL_RESIZE_PERCENT)), interpolation=cv2.INTER_LINEAR)


def transform_resized(image):
    # grayed = cv2.cvtColor(
    #     image, cv2.COLOR_BGR2GRAY)

    # normalized = np.stack(
    #     [grayed, grayed, grayed], axis=-1) / 255

    normalized = image / 255

    return image_to_pytorch_image(normalized).numpy()


def extract_actions_from_image(area, buttons):

    # print(PLAY_AREA_WIDTH_HEIGHT)
    return [transform_resized(area), get_buttons_state(buttons)]


def extract_aim_from_image(area, buttons):

    buttons_state = get_buttons_state(buttons)

    if buttons_state == 0:
        return None

    # cv2.imshow(
    #     f"debug", cv2.circle(area, get_cursor_position(
    #         area), 5, (0, 0, 255), 5))
    # cv2.waitKey(0)

    return [transform_resized(area), get_cursor_position(area) / np.array([len(area[0]), len(area[1])])]


class ImageProcessor:
    def __init__(self) -> None:
        self.buff = Queue()

    def load_images(self, dataset_path, screenshot_ids, load_buttons=False):
        for screenshot_id in screenshot_ids:
            area_image = cv2.imread(
                path.join(dataset_path, 'area', screenshot_id), cv2.IMREAD_COLOR)
            button_image = cv2.imread(
                path.join(dataset_path, 'buttons', screenshot_id), cv2.IMREAD_COLOR)
            self.buff.put([area_image, button_image])
        self.buff.put(None)

    def process_images(self, loader_description, dataset, extract_actions=True):
        dataset_path = path.join(getcwd(), 'data', 'raw', dataset)

        screenshot_ids = listdir(path.join(dataset_path, 'area'))

        if extract_actions:  # sort the play area so we can process the input properly
            screenshot_ids.sort(key=lambda x: int(
                re.search(r"([0-9]+).png", x).groups()[0]))

        processed = []

        loading_bar = tqdm(total=len(screenshot_ids),
                           desc=loader_description)

        Thread(target=self.load_images, daemon=True, group=None,
               kwargs={"load_buttons": extract_actions, "dataset_path": dataset_path, "screenshot_ids": screenshot_ids}).start()  # load the images in a seperate thread

        data = self.buff.get()  # get an image or none if we are done
        while data is not None:
            area, buttons = data

            result = None

            # Extract the required data from the image
            if extract_actions:
                result = extract_actions_from_image(area, buttons)
            else:
                result = extract_aim_from_image(area, buttons)

            if result is not None:
                processed.append(result)

            # update the progress bar
            loading_bar.update()

            data = self.buff.get()
        loading_bar.close()
        return np.array(processed, dtype=object)


class OsuDataset(torch.utils.data.Dataset):

    def __init__(self, datasets: list[str], frame_latency=3, is_actions=True, force_rebuild=False) -> None:
        self.datasets = datasets

        self.processed_path = path.join(getcwd(
        ), 'data', 'processed', f"{'actions_' if is_actions else 'aim_'}{'-'.join(datasets)}.npy")
        self.images = []
        self.labels = []
        self.frame_latency = frame_latency
        self.is_actions = is_actions

        if not force_rebuild and path.exists(self.processed_path):
            loaded_data = np.load(
                self.processed_path, allow_pickle=True)
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

        processor = ImageProcessor()

        processed_data = np.empty((0, 2))

        for i in range(len(self.datasets)):
            dataset = self.datasets[i]
            processed_data = np.concatenate(
                [processed_data, processor.process_images(f"Processing Dataset {dataset} ({i + 1}/{len(self.datasets)})", dataset, self.is_actions)])

        np.save(self.processed_path, processed_data)

        self.images = list(processed_data[:, 0])

        self.labels = list(processed_data[:, 1])

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)


# np.save('test_data.npy', extract_data_from_image(
#     "D:\Github\osu-ai\data\\raw\meaning-of-love-4.62\\755.png"))
