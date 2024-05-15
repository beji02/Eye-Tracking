from torchvision import transforms
import torchvision.models as models

def create_data_transformations_for_resnet():
    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
    ])
    return transformations

def create_mobileNetV2_transformations():
    weights = models.MobileNet_V2_Weights.DEFAULT
    preprocess = weights.transforms()
    return preprocess