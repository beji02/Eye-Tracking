from torch.utils.data.dataset import Dataset
import os
import numpy as np
import torch
from PIL import Image
from config.config import Config

class MPIIFaceGazeDataset(Dataset):
    def __init__(self, config: Config, image_dir, label_dir, train = True, is_pipeline_test = False, fold = 0, data_transformations = None):
        self._config = config
        self._image_dir = str(image_dir)
        self._label_dir = str(label_dir)
        self._transform = data_transformations
        self._train = train
        self._fold = fold
        self._is_pipeline_test = is_pipeline_test
        self._data = self._get_lines_from_files(self._label_dir)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        data_entry = self._data[idx]
        face_image_path, gaze2d = data_entry

        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        pitch = label[0] * 180 / np.pi
        yaw = label[1] * 180 / np.pi
        cont_labels = torch.FloatTensor([pitch, yaw]) 

        # Bin values
        bins = np.array(range(-42, 42,3))
        binned_pose = np.digitize([pitch, yaw], bins) - 1
        binned_labels = binned_pose

        image = Image.open(os.path.join(self._image_dir, face_image_path))
        # print(np.array(image))
        if self._transform:
            image = self._transform(image)      
        # print(np.array(image))
        # print("FUCK")
        return image, binned_labels, cont_labels

    def _get_lines_from_files(self, label_dir):
        data = []
        folder = os.listdir(label_dir)
        folder.sort()
        person_label_paths = [os.path.join(label_dir, person) for person in folder]

        if self._config.data.folding_strategy == 2:
            folds = [self._fold, (self._fold + 1) % 15, (self._fold + 2) % 15]
            if self._train:
                folds = [fold for fold in range(15) if fold not in folds]
            person_label_paths = [person_label_paths[fold] for fold in folds]
        else:
            if self._train:
                person_label_paths.pop(self._fold)
            else:
                person_label_paths = person_label_paths[self._fold:self._fold+1]


        for person_label_path in person_label_paths:
            with open(person_label_path) as file:
                lines = file.readlines()
                lines.pop(0)
                for line in lines:
                    line = line.strip().split(" ")

                    face_image_path = line[0]
                    gaze2d = line[7]
                    data.append((face_image_path, gaze2d))
        
        if self._is_pipeline_test:
            data = data[:600]
        return data
    
    def get_raw_image(self, idx):
        data_entry = self._data[idx]
        face_image_path, _ = data_entry
        image = Image.open(os.path.join(self._image_dir, face_image_path))
        return image
    
    
# class SmallEvaluationDataset(Dataset):


class InferenceDataset(Dataset):
    def __init__(self, image, data_transformations = None):
        images = np.expand_dims(image, axis=0)
        self._data = images
        self._transform = data_transformations

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = self._data[idx]
        # print(image)
        image = Image.fromarray(image)
        if self._transform:
            image = self._transform(image)  
        # print(image)
        return image
