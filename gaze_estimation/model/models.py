import torchvision
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torch
from torchvision.io import read_image, write_png

def create_L2CS_model(arch, bin_count, device):
    model, pre_url = _get_arch_weights(arch, bin_count)
    _load_filtered_state_dict(model, model_zoo.load_url(pre_url))
    # model = nn.DataParallel(model)
    model = model.to(device)
    return model

def _get_arch_weights(arch, bins):
    if arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
        pre_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

    return model, pre_url
    
def _load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)
    
def get_ignored_params(model):
    # Generator function that yields ignored params.
    if isinstance(model, nn.DataParallel):
        model = model.module  # Unwrap the model from DataParallel

    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Set to store already encountered parameters
    seen_params = set()

    # Generator function that yields params that will be optimized.
    if isinstance(model, nn.DataParallel):
        model = model.module  # Unwrap the model from DataParallel

    b = [model.layer1]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                if param not in seen_params:
                    # Yield only if the parameter is not already encountered
                    yield param
                    seen_params.add(param)

def get_fc_params(model):
    if isinstance(model, nn.DataParallel):
        model = model.module  # Unwrap the model from DataParallel
    # Generator function that yields fc layer params.
    b = [model.fc_yaw_gaze, model.fc_pitch_gaze]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

class L2CS(nn.Module):
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(L2CS, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_yaw_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch_gaze = nn.Linear(512 * block.expansion, num_bins)

       # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # gaze
        pre_yaw_gaze =  self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)
        return pre_yaw_gaze, pre_pitch_gaze



class CustomMobileNet(nn.Module):
    def __init__(self, num_outputs=2):
        super(CustomMobileNet, self).__init__()
        
        # Load the pre-trained MobileNetV2 model
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Modify the classifier to output 2 continuous values (yaw and gaze)
        num_features = self.mobilenet.classifier[-1].in_features
        self.mobilenet.classifier[-1] = nn.Sequential(
            nn.Linear(num_features, 2 * num_outputs)
        )

    def forward(self, x):
        # Forward pass through MobileNetV2 backbone
        x = self.mobilenet.features(x)
        
        # Global average pooling
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        
        # Flatten features
        x = torch.flatten(x, 1)
        
        # Forward pass through modified classifier
        x = self.mobilenet.classifier(x)
        
        # Separate outputs for yaw and gaze
        yaw, gaze = torch.split(x, [1, 1], dim=1)
        
        return yaw, gaze
    


class MyLoveMobileNet(nn.Module):
    def __init__(self):
        super(MyLoveMobileNet, self).__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT
        pretrained_mobilenet_v2 = models.mobilenet_v2(weights=weights)
        features = pretrained_mobilenet_v2.features
        
        #freeze first 9 layers
        first_9_layers = features[:9]
        for layer in first_9_layers:
            for param in layer.parameters():
                param.requires_grad = False
        self.features = features
        self.regression = nn.Linear(in_features=1280, out_features=2)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.regression(x)
        return x
    
if __name__ == "__main__":
    print("HELOO")
    model = MyLoveMobileNet()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')

    model = create_L2CS_model('ResNet50', 28, 'cpu')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')

    model = create_L2CS_model('ResNet18', 28, 'cpu')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')
    # # Create an instance of the model
    # # model = CustomMobileNet()

    # img = read_image("/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/demo/test_images/car.png")

    # weights = models.MobileNet_V2_Weights.DEFAULT
    # model = models.mobilenet_v2(weights=weights)
    # print(model)

    # all_layers = model.features
    # input_tensor = torch.randn(10, 3, 224, 224)
    # preprocess = weights.transforms()
    # batch = preprocess(input_tensor)
    # x = batch
    # print(x.shape)

    # first_9_layers = all_layers[:9]
    # for layer in first_9_layers:
    #     for param in layer.parameters():
    #         param.requires_grad = False

    # for layer in all_layers:
    #     # print(layer)
    #     x = layer(x)
    #     print(x.shape)
    # x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
    # print(x.shape)
    # x = torch.flatten(x, 1)
    # print(x.shape)

    # for layer in model.classifier:
    #     # print(layer)
    #     x = layer(x)
    #     print(x.shape)
    # # print(layer)

    # my_model = MyLoveMobileNet()
    # x = batch
    # x = my_model.forward(x)
    # print(x.shape)

    # print(my_model)

    # model = MyLoveMobileNet()
    # print(model)


    # x = torch.randn(10, 3, 224, 224)
    # k = 0
    # for layer in model.features:
    #     k+=1
    #     if k == 10:
    #         print(";;;;")
    #     print(x.shape)
    #     x = layer(x)
    # print("SSSS")
    # print(x.shape)
    # x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
    # print(x.shape)
    # x = torch.flatten(x, 1)
    # for layer in [model.regression]:
    #     print(x.shape)
    #     x = layer(x)
    # print(x.shape)
    # from torch.utils.data import DataLoader, TensorDataset

    # # Prepare a simple dataset and dataloader
    # inputs = torch.randn(10, 3, 224, 224)  # 10 random images
    # targets = torch.randint(0, 2, (10, 2)).float()  # Random targets for regression
    # dataset = TensorDataset(inputs, targets)
    # dataloader = DataLoader(dataset, batch_size=2)

    # # Define loss function and optimizer
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # monitored_param = next(model.features[9].parameters()).clone()
    # print("Parameter before training:", monitored_param)

    # # Training loop
    # for epoch in range(1):
    #     for data in dataloader:
    #         inputs, labels = data
    #         optimizer.zero_grad()  # zero the parameter gradients
    #         outputs = model(inputs)  # forward
    #         loss = criterion(outputs, labels)
    #         loss.backward()  # backward
    #         optimizer.step()  # optimize

    # monitored_param_after = next(model.features[9].parameters()).clone()
    # print("Parameter after training:", monitored_param_after)

    # # Check if the parameter has changed
    # if torch.equal(monitored_param, monitored_param_after):
    #     print("The parameter has not changed.")
    # else:
    #     print("The parameter has changed.")

    # Check if the first 9 layers' gradients are frozen
    # frozen_gradients = True
    # for i, layer in enumerate(model.features[:9]):
    #     for param in layer.parameters():
    #         if param.grad is not None:
    #             frozen_gradients = False
    #             print(f"Layer {i} is not frozen.")
    #             break

    # if frozen_gradients:
    #     print("The first 9 layers are correctly frozen.")
    # else:
    #     print("Some layers are not frozen.")




    
    
    
    # model.eval()
    
    # # Generate dummy input
    # # dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, RGB image
    # weights = models.MobileNet_V2_Weights.DEFAULT
    # preprocess = weights.transforms()

    # batch = preprocess(img).unsqueeze(0)
    # # write_png(img, "/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/demo/test_images/mobilent_output.png")
    # prediction = model(batch).squeeze(0).softmax(0)
    # print(prediction.shape)
    # class_id = prediction.argmax().item()
    # score = prediction[class_id].item()
    # category_name = weights.meta["categories"][class_id]
    # print(f"{category_name}: {100 * score:.1f}%")
    
    # Forward pass
    # yaw, gaze = model(dummy_input)
    
    # # Print output shapes
    # print("Yaw output shape:", yaw.shape)
    # print("Gaze output shape:", gaze.shape)