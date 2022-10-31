

import torchvision.transforms
import torch
from os import getcwd, listdir, path
import re
import cv2
import numpy as np

from re import Match

import torchvision.transforms as transforms

trans = transforms.ToTensor()


class OsuDataset(torch.utils.data.Dataset):
    IMG_SIZE = (960, 540)

    def __init__(self, project_name: str, use_clicks=True, force_rebuild=False) -> None:
        self.project_path = path.normpath(
            path.join(getcwd(), 'data', 'raw', project_name))
        print(self.project_path)
        self.processed_data_path = path.normpath(path.join(
            getcwd(), 'data', 'processed', f"{project_name}_data.npy"))
        self.processed_clicks_results_path = path.normpath(path.join(
            getcwd(), 'data', 'processed', f"{project_name}_clicks_results.npy"))
        self.processed_position_results_path = path.normpath(path.join(
            getcwd(), 'data', 'processed', f"{project_name}_position_results.npy"))
        self.data = []
        self.results = []
        self.results_to_load = self.processed_clicks_results_path if use_clicks else self.processed_position_results_path
        if not force_rebuild and path.exists(self.processed_data_path) and path.exists(self.results_to_load):
            self.data = np.load(
                self.processed_data_path, allow_pickle=True)
            self.results = np.load(
                self.results_to_load, allow_pickle=True)

        else:
            self.make_training_data()

    def make_training_data(self):
        print('Generating data')
        click_0 = []
        click_1 = []
        clicks = []
        positions = []
        for img_path in listdir(self.project_path):
            try:
                match: Match = re.match(
                    r"([a-z0-9-]+)_([0-9.]+)_([0-9.]+)_([0-9.]+)_([0-1])_([0-1])", img_path)
                if not match:
                    raise Exception("Failed To Match File {}".format(img_path))

                project, idx, x, y, k1, k2 = match.groups()
                img = cv2.imread(path.normpath(path.join(
                    self.project_path, img_path)), cv2.IMREAD_COLOR)
                self.data.append(np.array(img))
                clicks.append(
                    np.array([max(float(k1), float(k2))]))

                if max(float(k1), float(k2)) > 0.5:
                    click_1.append([project, idx, x, y, k1, k2])
                else:
                    click_0.append([project, idx, x, y, k1, k2])

                positions.append([float(x), float(y)])

            except Exception as e:
                print('ERROR WHILE LOADING', img_path, e)

        print('READ IN {} Clicks, and {} None Clicks'.format(
            len(click_1), len(click_0)))
        diff = len(click_1) - len(click_0)
        if abs(diff) > 100:
            delta = abs(diff)
            print('Balancing Clicks Due To Difference Of {}'.format(delta))
            group_to_duplicate = click_0 if diff > 1 else click_1
            group_len = len(group_to_duplicate)
            wraps = 1
            for i in range(delta):
                if i % group_len == 0 and i != 0:
                    wraps += 1

                project, idx, x, y, k1, k2 = group_to_duplicate[i % group_len]
                original_file = path.normpath(path.join(
                    self.project_path, '{}_{}_{}_{}_{}_{}.png'.format(project, idx, x, y, k1, k2)))
                new_file = path.normpath(path.join(
                    self.project_path, '{}_{}{}_{}_{}_{}_{}.png'.format(project, "0"*wraps, idx, x, y, k1, k2)))
                cv2.imwrite(new_file, cv2.imread(
                    original_file, cv2.IMREAD_COLOR))

            self.data = []
            self.results = []
            print('Done Balancing, Retrying Data Import')
            self.make_training_data()
            return

        print('Saving')
        np.save(self.processed_data_path, self.data)
        np.save(self.processed_clicks_results_path, clicks)
        np.save(self.processed_position_results_path, positions)

        self.data = np.load(
            self.processed_data_path, allow_pickle=True)
        self.results = np.load(
            self.results_to_load, allow_pickle=True)

    def __getitem__(self, idx):
        return trans(self.data[idx]), self.results[idx]

    def __len__(self):
        return len(self.data)
