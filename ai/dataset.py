import re
from os import getcwd, listdir, path
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from queue import Queue
from threading import Thread
from tqdm import tqdm
from constants import CURRENT_STACK_NUM, FINAL_PLAY_AREA_SIZE, PLAY_AREA_CAPTURE_PARAMS
from collections import deque
image_to_pytorch_image = transforms.ToTensor()

INVALID_KEY_STATE = "An Invalid State"

KEY_STATES = {
    "00": 0,
    "01": 1,
    "10": 2,
}


def transform_frame(frame):
    # crop to play area
    frame = frame[PLAY_AREA_CAPTURE_PARAMS[3]:PLAY_AREA_CAPTURE_PARAMS[3] + PLAY_AREA_CAPTURE_PARAMS[1],
                  PLAY_AREA_CAPTURE_PARAMS[2]:PLAY_AREA_CAPTURE_PARAMS[2] + PLAY_AREA_CAPTURE_PARAMS[0]]

    # resize
    frame = cv2.resize(frame, FINAL_PLAY_AREA_SIZE,
                       interpolation=cv2.INTER_CUBIC)

    # greyscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # normalize
    frame = frame / 255

    # grayed = cv2.cvtColor(
    #     image, cv2.COLOR_BGR2GRAY)

    # normalized = np.stack(
    #     [grayed, grayed, grayed], axis=-1) / 255

    # normalized = image / 255

    # return image_to_pytorch_image(normalized).numpy()
    return frame


def extract_data(area, state: str):

    # cv2.imshow(
    #     f"debug", area)
    # cv2.waitKey(0)

    _, k1, k2, x, y = state.split("-")[1].split(',')

    # cv2.imshow(
    #     f"debug", cv2.circle(area, (int((float(x.strip()) -
    #                                      PLAY_AREA_CAPTURE_PARAMS[2])), int((float(y.strip()) -
    #                                                                         PLAY_AREA_CAPTURE_PARAMS[3]))), 5, (0, 0, 255), 5))
    # cv2.waitKey(0)

    x = (float(x.strip()) -
         PLAY_AREA_CAPTURE_PARAMS[2]) / PLAY_AREA_CAPTURE_PARAMS[0]

    y = (float(y.strip()) -
         PLAY_AREA_CAPTURE_PARAMS[3]) / PLAY_AREA_CAPTURE_PARAMS[1]

    return (transform_frame(area), KEY_STATES.get(f"{k1}{k2}".strip(), 0), np.array([x, y]))


class OsuDatasetCreator:
    def __init__(self) -> None:
        self.work_queue = Queue()
        self.frame_buffer = deque(maxlen=CURRENT_STACK_NUM)

    def get_stacked(self, frame):
        prev_frames = list(self.frame_buffer)
        prev_count = len(prev_frames)
        needed_count = CURRENT_STACK_NUM - prev_count
        final_frames = []

        if needed_count > 1:
            final_frames = prev_frames + [frame for _ in range(needed_count)]
        else:
            final_frames = prev_frames[prev_count -
                                       (CURRENT_STACK_NUM - 1):prev_count] + [frame]
        self.frame_buffer.append(frame)

        return np.stack(final_frames)

    def load_images_in_background(self, dataset_path, screenshot_ids, load_buttons=False):
        try:
            for screenshot_id in screenshot_ids:
                self.work_queue.put((cv2.imread(
                    path.join(dataset_path, screenshot_id), cv2.IMREAD_COLOR), screenshot_id[:-4]))

            self.work_queue.put(None)
        except Exception as e:
            self.work_queue.put(None)
            raise e

    def to_dataset(self, loader_description, dataset, extract_actions=True):
        dataset_path = path.join(getcwd(), 'data', 'raw', dataset)

        screenshot_ids = listdir(dataset_path)
        REXEXP = r"[a-z]+-([0-9]+),[0-1],[0-1],[0-9]+,[0-9]+.png"
        screenshot_ids.sort(key=lambda x: int(
            re.search(REXEXP, x).groups()[0]))
        processed = []

        loading_bar = tqdm(total=len(screenshot_ids),
                           desc=loader_description)

        Thread(target=self.load_images_in_background, daemon=True, group=None,
               kwargs={"dataset_path": dataset_path, "screenshot_ids": screenshot_ids}).start()  # load the images in a seperate thread

        data = self.work_queue.get()  # get an image or none if we are done
        while data is not None:
            sct, state = data

            frame, key_dat, aim_dat = extract_data(sct, state)

            stacked = self.get_stacked(frame)

            # cv2.imshow(f"debug", stacked.transpose((1, 2, 0)))
            # cv2.waitKey(10)

            processed.append(
                np.array([stacked, key_dat, aim_dat], dtype=object))

            # update the progress bar
            loading_bar.update()

            data = self.work_queue.get()
        loading_bar.close()
        self.frame_buffer.clear()
        return np.stack(processed)


class OsuDataset(torch.utils.data.Dataset):

    """
    label_type = 1 | 2, 1 = actions, 2 = aim
    """    
    def __init__(self, datasets: list[str], frame_latency=3, label_type=1, force_rebuild=False) -> None:
        self.datasets = datasets

        self.processed_path = path.join(getcwd(
        ), 'data', 'processed', f"{'-'.join(datasets)}.npy")
        self.images = []
        self.labels = []
        self.frame_latency = frame_latency
        self.label_index = label_type

        if not force_rebuild and path.exists(self.processed_path):
            loaded_data = np.load(
                self.processed_path, allow_pickle=True)
            self.images = list(loaded_data[:, 0])
            self.labels = list(loaded_data[:, self.label_index])
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

        processor = OsuDatasetCreator()

        processed_data = None

        for i in range(len(self.datasets)):
            dataset = self.datasets[i]
            new_data = processor.to_dataset(
                f"Processing Dataset {dataset} ({i + 1}/{len(self.datasets)})", dataset, self.label_index)
            if processed_data is not None:
                processed_data = np.concatenate([processed_data, new_data])
            else:
                processed_data = new_data

        if processed_data is None:
            raise Exception("No data was processed")

        np.save(self.processed_path, processed_data)

        self.images = list(processed_data[:, 0])

        self.labels = list(processed_data[:, 1]) if self.label_index else list(
            processed_data[:, 2])

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)


# np.save('test_data.npy', extract_data_from_image(
#     "D:\Github\osu-ai\data\\raw\meaning-of-love-4.62\\755.png"))
