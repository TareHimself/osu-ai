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


def transform_resized(image):
    # grayed = cv2.cvtColor(
    #     image, cv2.COLOR_BGR2GRAY)

    # normalized = np.stack(
    #     [grayed, grayed, grayed], axis=-1) / 255

    # normalized = image / 255

    # return image_to_pytorch_image(normalized).numpy()
    return image / 255


def extract_data(area, state: str):

    # cv2.imshow(
    #     f"debug", area)
    # cv2.waitKey(0)

    key_1, key_2, x, y = state.split(",")

    # cv2.imshow(
    #     f"debug", cv2.circle(area, (int((float(x.strip()) -
    #                                      PLAY_AREA_CAPTURE_PARAMS[2])), int((float(y.strip()) -
    #                                                                         PLAY_AREA_CAPTURE_PARAMS[3]))), 5, (0, 0, 255), 5))
    # cv2.waitKey(0)

    x = (float(x.strip()) -
         PLAY_AREA_CAPTURE_PARAMS[2]) / PLAY_AREA_CAPTURE_PARAMS[0]

    y = (float(y.strip()) -
         PLAY_AREA_CAPTURE_PARAMS[3]) / PLAY_AREA_CAPTURE_PARAMS[1]

    return [transform_resized(area), KEY_STATES.get(f"{key_1}{key_2}".strip(), 0), np.array([x, y])]


class ImageProcessor:
    def __init__(self) -> None:
        self.buff = Queue()

    def load_images(self, dataset_path, screenshot_ids, load_buttons=False):
        for screenshot_id in screenshot_ids:

            sct = np.load(path.join(dataset_path, 'display',
                                    screenshot_id), allow_pickle=True)

            f = open(path.join(dataset_path, 'state',
                               screenshot_id[:-3] + "txt"))
            state = f.read()
            self.buff.put([sct, state])
        self.buff.put(None)

    def process_images(self, loader_description, dataset, extract_actions=True):
        dataset_path = path.join(getcwd(), 'data', 'raw', dataset)

        screenshot_ids = listdir(path.join(dataset_path, 'display'))

        if extract_actions:  # sort the play area so we can process the input properly
            screenshot_ids.sort(key=lambda x: int(
                re.search(r"([0-9]+).npy", x).groups()[0]))

        processed = []

        loading_bar = tqdm(total=len(screenshot_ids),
                           desc=loader_description)

        Thread(target=self.load_images, daemon=True, group=None,
               kwargs={"dataset_path": dataset_path, "screenshot_ids": screenshot_ids}).start()  # load the images in a seperate thread

        data = self.buff.get()  # get an image or none if we are done
        while data is not None:
            sct, state = data

            result = None

            result = extract_data(sct, state)

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
        ), 'data', 'processed', f"{'-'.join(datasets)}.npy")
        self.images = []
        self.labels = []
        self.frame_latency = frame_latency
        self.is_actions = is_actions

        if not force_rebuild and path.exists(self.processed_path):
            loaded_data = np.load(
                self.processed_path, allow_pickle=True)
            self.images = list(loaded_data[:, 0])
            self.labels = list(loaded_data[:, 1]) if is_actions else list(
                loaded_data[:, 2])
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

        processed_data = np.empty((0, 3))

        for i in range(len(self.datasets)):
            dataset = self.datasets[i]
            processed_data = np.concatenate(
                [processed_data, processor.process_images(f"Processing Dataset {dataset} ({i + 1}/{len(self.datasets)})", dataset, self.is_actions)])

        np.save(self.processed_path, processed_data)

        self.images = list(processed_data[:, 0])

        self.labels = list(processed_data[:, 1]) if self.is_actions else list(
            processed_data[:, 2])

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)


# np.save('test_data.npy', extract_data_from_image(
#     "D:\Github\osu-ai\data\\raw\meaning-of-love-4.62\\755.png"))
