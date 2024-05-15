from data.data_transformation import create_data_transformations_for_resnet
from data.datasets import MPIIFaceGazeDataset
from torch.utils.data import DataLoader, RandomSampler
import torch
import matplotlib.pyplot as plt
import numpy as np

data_transformations = create_data_transformations_for_resnet()
image_dir = '/mnt/d/Downloads/MPIIFaceGaze/ProcessedByMe/Image'
label_dir = '/mnt/d/Downloads/MPIIFaceGaze/ProcessedByMe/Label'

dataset = MPIIFaceGazeDataset(image_dir, label_dir, train=True, fold=0, data_transformations=data_transformations)
sampler = RandomSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler)

fig, axs = plt.subplots(1, 5, figsize=(5, 15))
for i, (image, label, cont_label) in enumerate(dataloader):
    image = image[0]
    label = label[0]
    cont_label = cont_label[0]
    print(image.shape)
    image_np = image.numpy().transpose(1, 2, 0)
    axs[i].imshow(image_np)
    axs[i].axis('off')
        
    axs[i].text(224, 500, label, horizontalalignment='center')
    axs[i].text(224, 550, cont_label, horizontalalignment='center')
    dx = -100 * np.sin(cont_label[0]) * np.cos(cont_label[1])
    dy = -100 * np.sin(cont_label[1])
    dx, dy = np.round([dx, dy])

    # axs[i].plot([224, 224+dx], [224, 224+dy], color='red', linewidth=2)
    axs[i].arrow(224, 224, dx, dy, head_width=20, head_length=20, fc='blue', ec='blue')


    if i == 4:
        break

plt.tight_layout()
plt.show()