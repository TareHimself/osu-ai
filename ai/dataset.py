import os
import re
from os import path
import cv2
import numpy as np
import traceback
from tempfile import TemporaryDirectory
import torchvision.transforms as transforms
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from ai.constants import (
    CURRENT_STACK_NUM,
    FINAL_PLAY_AREA_SIZE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    MAX_THREADS_FOR_RESIZING,
)
from collections import deque
from torch.utils.data import Dataset
from ai.enums import EModelType

image_to_pytorch_image = transforms.ToTensor()

INVALID_KEY_STATE = "An Invalid State"

KEY_STATES = {
    "00": 0,
    "01": 1,
    "10": 2,
}


class OsuDataset(Dataset):
    """ """

    FILE_REG_EXPR = r"-([0-9]+),[0-1],[0-1],[-0-9.]+,[-0-9.]+.png"

    def __init__(
        self,
        datasets: list[str],
        label_type: EModelType = EModelType.Actions,
        force_rebuild=False,
    ) -> None:
        self.datasets = datasets
        self.labels = []
        self.images = []
        self.label_type = label_type
        self.data_to_process = Queue()
        self.force_rebuild = force_rebuild
        self.make_training_data()

    @staticmethod
    def extract_info(frame, state, dims):
        width, height = dims
        # print(dims)
        _, k1, k2, x, y = state.split(",")

        # greyscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # normalize
        frame = frame / 255

        x = max(0, float(x.strip()))

        x = x if x == 0 else x / width

        y = max(0, float(y.strip()))

        y = y if y == 0 else y / height

        return frame, KEY_STATES.get(f"{k1}{k2}".strip(), 0), np.array([x, y])

    @staticmethod
    def stack_frames(previous_frames: deque, frame):
        prev_frames = list(previous_frames)
        prev_count = len(prev_frames)
        needed_count = CURRENT_STACK_NUM - prev_count

        if needed_count > 1:
            previous_frames.append(frame)
            return None
        else:
            final_frames = prev_frames[
                prev_count - (CURRENT_STACK_NUM - 1) : prev_count
            ] + [frame]
        previous_frames.append(frame)

        return np.stack(final_frames)

    def background_loader(self, dataset_dir: str, files_to_load: list[str]):
        try:
            for item in files_to_load:
                image_file = cv2.imread(path.join(dataset_dir, item), cv2.IMREAD_COLOR)
                self.data_to_process.put((image_file, item[:-4]))

            self.data_to_process.put(None)
        except:
            traceback.print_exc()

    @staticmethod
    def resize_dataset(temp_directory: str, dataset: str, source_path: str):
        files = []

        files_to_load = os.listdir(source_path)
        loading_bar = tqdm(
            desc=f"Resizing Dataset [{dataset}]", total=len(files_to_load)
        )

        data_dims = None

        def resize_image(filename):
            nonlocal temp_directory
            nonlocal data_dims

            try:
                current_item_source_path = path.join(source_path, filename)
                current_item_dest_path = path.join(temp_directory, filename)

                frame = cv2.imread(current_item_source_path, cv2.IMREAD_COLOR)

                if data_dims is None:
                    data_dims = frame.shape[:2][::-1]

                frame = cv2.resize(
                    frame, FINAL_PLAY_AREA_SIZE, interpolation=cv2.INTER_LINEAR
                )

                cv2.imwrite(current_item_dest_path, frame)
                files.append(filename)
                loading_bar.update()

            except Exception:
                traceback.print_exc()

        try:
            with ThreadPoolExecutor(MAX_THREADS_FOR_RESIZING) as executor:
                for file in files_to_load:
                    executor.submit(resize_image, file)

                executor.shutdown()

        except KeyboardInterrupt:
            pass

        return files, data_dims

    def get_or_create_dataset(
        self, temp_directory: str, dataset: str
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        try:
            processed_data_path = path.join(
                PROCESSED_DATA_DIR,
                f"{CURRENT_STACK_NUM}-{FINAL_PLAY_AREA_SIZE[0]}-{dataset}.npy",
            )
            raw_data_path = path.join(RAW_DATA_DIR, f"{dataset}")

            if not self.force_rebuild and path.exists(processed_data_path):
                loaded_data = np.load(processed_data_path, allow_pickle=True)
                return (
                    list(loaded_data[:, 0]),
                    list(loaded_data[:, 1]),
                    list(loaded_data[:, 2]),
                )

            files, data_dims = OsuDataset.resize_dataset(
                temp_directory, dataset, raw_data_path
            )

            files.sort(
                key=lambda x: int(re.search(OsuDataset.FILE_REG_EXPR, x).groups()[0])
            )

            frame_queue = deque(maxlen=CURRENT_STACK_NUM - 1)

            processed = []

            Thread(
                target=self.background_loader,
                daemon=True,
                group=None,
                args=[temp_directory, files],
            ).start()

            loader = tqdm(total=len(files), desc=f"Processing Dataset [{dataset}]")

            data = self.data_to_process.get()

            while data is not None:
                frame, state = data

                frame, key_state, mouse_state = OsuDataset.extract_info(
                    frame, state, data_dims
                )

                stacked = OsuDataset.stack_frames(frame_queue, frame)
                if stacked is None:
                    loader.update()
                    data = self.data_to_process.get()
                    continue
                # transp = stacked.transpose(1, 2, 0)
                # cv2.imshow("Debug", transp)
                # cv2.waitKey(0)

                processed.append(
                    np.array([stacked, key_state, mouse_state], dtype=object)
                )

                loader.update()
                data = self.data_to_process.get()
            loader.close()

            processed = np.stack(processed)

            print(f"Saving Dataset [{dataset}]")
            np.save(processed_data_path, processed)

            return list(processed[:, 0]), list(processed[:, 1]), list(processed[:, 2])
        except Exception:
            self.data_to_process.put(None)

            traceback.print_exc()

            return [], [], []

    def make_training_data(self):
        try:
            self.labels = []
            self.images = []
            total_images = []
            total_mouse_coordinates = []
            total_keys = []

            with TemporaryDirectory() as temp_dir:
                for dataset in self.datasets:
                    images, keys, coordinates = self.get_or_create_dataset(
                        temp_dir, dataset
                    )
                    total_images.extend(images)
                    total_mouse_coordinates.extend(coordinates)
                    total_keys.extend(keys)

            if self.label_type == EModelType.Actions:
                self.images = total_images
                self.labels = total_keys

                unique_labels = list(set(self.labels))
                counts = {}
                for label in unique_labels:
                    counts[label] = 0

                for label in self.labels:
                    counts[label] += 1

                print("Initial Data Balance", counts)
                target_amount = max(counts.values())
                for label in counts.keys():
                    if counts[label] < target_amount:
                        label_examples = [
                            self.images[x]
                            for x in range(len(self.labels))
                            if self.labels[x] == label
                        ]
                        len_examples = len(label_examples)
                        for i in range(target_amount - counts[label]):
                            self.labels.append(label)
                            self.images.append(label_examples[i % len_examples])

                for label in unique_labels:
                    counts[label] = 0

                for label in self.labels:
                    counts[label] += 1

                print("Final Dataset Balance", counts)
            elif self.label_type == EModelType.Aim:
                self.images = total_images
                self.labels = total_mouse_coordinates
                print("Final Dataset Size", len(self.labels))

            elif self.label_type == EModelType.Combined:

                def convert_label(a):
                    return np.array(
                        [a[0][0], a[0][1], 1 if a[1] == 2 else 0, 1 if a[1] == 1 else 0]
                    )

                self.labels = list(
                    map(convert_label, zip(total_mouse_coordinates, total_keys))
                )
                self.images = total_images

                unique_labels = list(set(total_keys))
                counts = {}
                for label in unique_labels:
                    counts[label] = 0

                for label in self.labels:
                    counts[KEY_STATES[f"{int(label[2])}{int(label[3])}"]] += 1

                print("Initial Data Balance", counts)
                target_amount = max(counts.values())
                for label in counts.keys():
                    if counts[label] < target_amount:
                        label_examples = [
                            (self.images[x], x)
                            for x in range(len(total_keys))
                            if total_keys[x] == label
                        ]
                        len_examples = len(label_examples)
                        for i in range(target_amount - counts[label]):
                            target_example, target_index = label_examples[
                                i % len_examples
                            ]
                            self.labels.append(self.labels[target_index])
                            self.images.append(target_example)

                for label in unique_labels:
                    counts[label] = 0

                for label in self.labels:
                    counts[KEY_STATES[f"{int(label[2])}{int(label[3])}"]] += 1

                print("Final Dataset Balance", counts)

        except Exception:
            traceback.print_exc()

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


# np.save('test_data.npy', extract_data_from_image(
#     "D:\Github\osu-ai\data\\raw\meaning-of-love-4.62\\755.png"))
