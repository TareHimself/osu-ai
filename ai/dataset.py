import re
from os import path
import cv2
import numpy as np
import torch
import traceback
from tempfile import TemporaryDirectory
import zipfile
import torchvision.transforms as transforms
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from constants import CURRENT_STACK_NUM, FINAL_PLAY_AREA_SIZE, PLAY_AREA_CAPTURE_PARAMS, FINAL_RESIZE_PERCENT, PROCESSED_DATA_DIR, RAW_DATA_DIR, MAX_THREADS_FOR_RESIZING
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

    LABEL_TYPE_ACTIONS = 1
    LABEL_TYPE_AIM = 2
    FILE_REXEXP = r"[a-zA-Z0-9\(\)\s]+-([0-9]+),[0-1],[0-1],[0-9]+,[0-9]+.png"

    def __init__(self, datasets: list[str], label_type=LABEL_TYPE_ACTIONS, force_rebuild=False) -> None:
        self.datasets = datasets
        self.labels = []
        self.images = []
        self.label_index = label_type
        self.data_to_process = Queue()
        self.force_rebuild = force_rebuild

        self.make_training_data()

    def extract_info(self, frame, state):
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

    def extract_and_resize(self, dataset, source_path):
        files = []
        temp_path = self.temp_dir
        source_zip = zipfile.ZipFile(file=source_path)
        files_to_load = source_zip.namelist()
        loading_bar = tqdm(
            desc=f"Resizing Dataset [{dataset[:-4]}]", total=len(files_to_load))

        def resize_image(filename):
            nonlocal source_zip
            nonlocal temp_path
            try:
                source_zip.extract(member=filename, path=temp_path)

                current_item_path = path.join(temp_path, filename)

                frame = cv2.imread(current_item_path, cv2.IMREAD_COLOR)

                # crop to play area
                frame = frame[PLAY_AREA_CAPTURE_PARAMS[3]:PLAY_AREA_CAPTURE_PARAMS[3] + PLAY_AREA_CAPTURE_PARAMS[1],
                              PLAY_AREA_CAPTURE_PARAMS[2]:PLAY_AREA_CAPTURE_PARAMS[2] + PLAY_AREA_CAPTURE_PARAMS[0]]

                # resize
                frame = cv2.resize(
                    frame, FINAL_PLAY_AREA_SIZE, interpolation=cv2.INTER_LINEAR)

                cv2.imwrite(current_item_path, frame)
                files.append(filename)
                loading_bar.update()

            except Exception as e:
                print(e, traceback.format_exc())

        try:
            with ThreadPoolExecutor(MAX_THREADS_FOR_RESIZING) as exec:
                for file in files_to_load:
                    exec.submit(resize_image, file)
                exec.shutdown()

        except KeyboardInterrupt as e:
            pass

        return files

    def get_or_create_dataset(self, temp_dir, dataset: str):

        try:

            processed_data_path = path.join(
                PROCESSED_DATA_DIR, f"{CURRENT_STACK_NUM}-{FINAL_RESIZE_PERCENT}-{dataset[:-4]}.npy")
            raw_data_path = path.join(
                RAW_DATA_DIR, f'{dataset}')

            if not self.force_rebuild and path.exists(processed_data_path):
                loaded_data = np.load(processed_data_path, allow_pickle=True)
                return list(loaded_data[:, 0]),list(loaded_data[:, 1]),list(loaded_data[:, 2])

            files = self.extract_and_resize(
                dataset, raw_data_path)

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

                
                drawn = frame
                # x,y = (int((mouse_state[0] * PLAY_AREA_CAPTURE_PARAMS[0]) / FINAL_RESIZE_PERCENT),int((mouse_state[1] * PLAY_AREA_CAPTURE_PARAMS[1]) / FINAL_RESIZE_PERCENT))
                # print(x,y)
                # drawn = cv2.circle(frame,(x,y),3,(255,255,255),2)

                stacked = self.stack_frames(frame_queue, drawn)
                # transp = stacked.transpose(1, 2, 0)
                # cv2.imshow("Debug", transp)
                # cv2.waitKey(0)

                processed.append(
                    np.array([stacked, key_state, mouse_state], dtype=object))

                loader.update()
                data = self.data_to_process.get()
            loader.close()

            processed = np.stack(processed)

            print(f"Saving Dataset [{dataset[:-4]}]")
            np.save(processed_data_path, processed)

            return list(processed[:, 0]), list(processed[:, 1]),list(processed[:, 2])
        except Exception as e:

            self.data_to_process.put(None)

            traceback.print_exception()

            return [], []

    def make_training_data(self):
        try:
            self.labels = []
            self.images = []
            total_images = []
            total_mouse_coords = []
            total_keys = []
            with TemporaryDirectory() as temp_dir:
                self.temp_dir = temp_dir
                for dataset in self.datasets:
                    images,keys,coords = self.get_or_create_dataset(
                        temp_dir, dataset)
                    total_images.extend(images)
                    total_mouse_coords.extend(coords)
                    total_keys.extend(keys)

            if self.label_index == OsuDataset.LABEL_TYPE_ACTIONS:
                total_mouse_coords = []
                self.images = total_images
                self.labels = total_keys

                unique_labels = list(set(self.labels))
                counts = {}
                for label in unique_labels:
                    counts[label] = 0

                for label in self.labels:
                    counts[label] += 1

                print("Initial Data Balance",counts)
                target_ammount = max(counts.values())
                for label in counts.keys():
                    if counts[label] < target_ammount:
                        label_examples = [self.images[x]
                                          for x in range(len(self.labels)) if self.labels[x] == label]
                        len_examples = len(label_examples)
                        for i in range(target_ammount - counts[label]):
                            self.labels.append(label)
                            self.images.append(
                                label_examples[i % len_examples])

                for label in unique_labels:
                    counts[label] = 0

                for label in self.labels:
                    counts[label] += 1

                print("Final Dataset Balance", counts)
            else:
                # for i in tqdm(range(len(total_keys)),desc="Filtering out useless data"):
                #     if total_keys[i] != 0:
                #         self.labels.append(total_mouse_coords[i])
                #         self.images.append(total_images[i])
                # total_images = []
                # total_keys = []
                # total_mouse_coords = []
                self.images = total_images
                self.labels = total_mouse_coords
                print("Final Dataset Size",len(self.labels))

        except Exception as e:
            print(traceback.format_exc())

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


# np.save('test_data.npy', extract_data_from_image(
#     "D:\Github\osu-ai\data\\raw\meaning-of-love-4.62\\755.png"))
