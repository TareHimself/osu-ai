import re
import uuid
from os import getcwd, listdir, path, makedirs
import cv2
import numpy as np
import torch
import traceback
from tempfile import TemporaryDirectory
import zipfile
import torchvision.transforms as transforms
from queue import Queue
from threading import Thread
from tqdm import tqdm
from constants import CURRENT_STACK_NUM, FINAL_PLAY_AREA_SIZE, PLAY_AREA_CAPTURE_PARAMS, FINAL_RESIZE_PERCENT
from collections import deque


image_to_pytorch_image = transforms.ToTensor()

INVALID_KEY_STATE = "An Invalid State"

KEY_STATES = {
    "00": 0,
    "01": 1,
    "10": 2,
}


class OsuDataset(torch.utils.data.Dataset):

    """

    """

    PROCESSED_DATA_PATH = path.join(getcwd(), 'data', 'processed')
    RAW_DATA_PATH = path.join(getcwd(), 'data', 'raw')
    LABEL_TYPE_ACTIONS = 1
    LABEL_TYPE_AIM = 2
    FILE_REXEXP = r"[a-zA-Z0-9\(\)\s]+-([0-9]+),[0-1],[0-1],[0-9]+,[0-9]+.png"

    def __init__(self, datasets: list[str], frame_latency=3, label_type=LABEL_TYPE_ACTIONS, force_rebuild=False) -> None:
        self.datasets = datasets
        self.images = []
        self.labels = []
        self.frame_latency = frame_latency
        self.label_index = label_type
        self.data_to_process = Queue()
        self.force_rebuild = force_rebuild

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

    def extract_info(self, frame, state):
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

        # cv2.imshow(
        #     f"debug", area)
        # cv2.waitKey(0)

        _, k1, k2, x, y = state.split(',')

        # cv2.imshow(
        #     f"debug", cv2.circle(area, (int((float(x.strip()) -
        #                                      PLAY_AREA_CAPTURE_PARAMS[2])), int((float(y.strip()) -
        #                                                                         PLAY_AREA_CAPTURE_PARAMS[3]))), 5, (0, 0, 255), 5))
        # cv2.waitKey(0)

        x = (float(x.strip()) -
             PLAY_AREA_CAPTURE_PARAMS[2]) / PLAY_AREA_CAPTURE_PARAMS[0]

        y = (float(y.strip()) -
             PLAY_AREA_CAPTURE_PARAMS[3]) / PLAY_AREA_CAPTURE_PARAMS[1]

        return (frame, KEY_STATES.get(f"{k1}{k2}".strip(), 0), np.array([x, y]))

    def stack_frames(self, previous_frames: deque, frame):
        prev_frames = list(previous_frames)
        prev_count = len(prev_frames)
        needed_count = CURRENT_STACK_NUM - prev_count
        final_frames = []
        if needed_count > 1:
            final_frames = prev_frames + [frame for _ in range(needed_count)]
        else:
            final_frames = prev_frames[prev_count -
                                       (CURRENT_STACK_NUM - 1):prev_count] + [frame]
        previous_frames.append(frame)

        return np.stack(final_frames)

    def background_loader(self, dir, files_to_load):
        try:
            for item in files_to_load:
                image_file = cv2.imread(
                    path.join(dir, item), cv2.IMREAD_COLOR)
                self.data_to_process.put((image_file, item[:-4]))

            self.data_to_process.put(None)
        except Exception as e:
            print(e, traceback.format_exc())

    def get_or_create_dataset(self, temp_dir, dataset: str):

        try:

            processed_data_path = path.join(
                OsuDataset.PROCESSED_DATA_PATH, f"{CURRENT_STACK_NUM}-{FINAL_RESIZE_PERCENT}-{dataset}.npy")
            raw_data_path = path.join(
                OsuDataset.RAW_DATA_PATH, f'{dataset}')

            if not self.force_rebuild and path.exists(processed_data_path):
                loaded_data = np.load(processed_data_path, allow_pickle=True)
                return list(loaded_data[:, 0]), list(loaded_data[:, self.label_index])

            files = []

            with zipfile.ZipFile(file=raw_data_path) as zip_file:
                # Loop over each file
                for data in tqdm(zip_file.namelist(), desc=f"Extracting Dataset [{dataset[:-4]}]"):
                    # Extract each file to another directory
                    # If you want to extract to current working directory, don't specify path
                    zip_file.extract(member=data, path=self.temp_dir)
                    files.append(data)

            files.sort(key=lambda x: int(
                re.search(OsuDataset.FILE_REXEXP, x).groups()[0]))

            frame_queue = deque(maxlen=CURRENT_STACK_NUM - 1)

            processed = []

            Thread(target=self.background_loader,
                   daemon=True, group=None, args=[temp_dir, files]).start()

            loader = tqdm(total=len(files),
                          desc=f"Processing Dataset [{dataset[:-4]}]")

            data = self.data_to_process.get()

            while data is not None:

                frame, state = data

                frame, key_state, mouse_state = self.extract_info(
                    frame, state)

                stacked = self.stack_frames(frame_queue, frame)
                cv2.imshow("Debug", stacked.transpose(1, 2, 0))
                cv2.waitKey(2)

                processed.append(
                    np.array([stacked, key_state, mouse_state], dtype=object))

                loader.update()
                data = self.data_to_process.get()
            loader.close()

            processed = np.stack(processed)

            np.save(processed_data_path, processed)

            return list(processed[:, 0]), list(processed[:, self.label_index])
        except Exception as e:

            self.data_to_process.put(None)

            print(e, traceback.format_exc())

            return [], []

    def make_training_data(self):
        try:
            self.labels = []
            self.images = []
            with TemporaryDirectory() as temp_dir:
                self.temp_dir = temp_dir
                for dataset in self.datasets:
                    images, labels = self.get_or_create_dataset(
                        temp_dir, dataset)
                    self.images.extend(images)
                    self.labels.extend(labels)

        except Exception as e:
            print(e, traceback.format_exc())

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)


# np.save('test_data.npy', extract_data_from_image(
#     "D:\Github\osu-ai\data\\raw\meaning-of-love-4.62\\755.png"))
