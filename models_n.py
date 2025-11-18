# Global Configuration
NUM_CLASSES = 2

import torch
from torch import nn
import timm
from torchvision import models

class MultiGPUModel(nn.Module):
    def __init__(self, base_model, multi_gpu=False):
        super(MultiGPUModel, self).__init__()
        self.base_model = base_model
        if multi_gpu and torch.cuda.device_count() > 1:
            self.base_model = nn.DataParallel(self.base_model)

    def forward(self, x):
        return self.base_model(x)

class ResNetModel(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=None, pretrained=True):
        super(ResNetModel, self).__init__()
        if num_classes is None:
            num_classes = NUM_CLASSES
        self.model = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained, in_chans=1)

    def forward(self, x):
        return self.model(x)

class VGGModel(nn.Module):
    def __init__(self, model_name='vgg16', num_classes=None, pretrained=True):
        super(VGGModel, self).__init__()
        if num_classes is None:
            num_classes = NUM_CLASSES
        self.model = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained, in_chans=1)

    def forward(self, x):
        return self.model(x)

class AlexNetModel(nn.Module):
    def __init__(self, num_classes=None, pretrained=True):
        super(AlexNetModel, self).__init__()
        if num_classes is None:
            num_classes = NUM_CLASSES
        weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.alexnet(weights=weights)
        # İlk conv layer'ı 1 kanala uyarla
        original_conv = self.model.features[0]
        new_conv = nn.Conv2d(1, original_conv.out_channels, kernel_size=original_conv.kernel_size,
                             stride=original_conv.stride, padding=original_conv.padding,
                             bias=original_conv.bias is not None)
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:, 0, :, :] = original_conv.weight.mean(dim=1)
                if original_conv.bias is not None:
                    new_conv.bias = original_conv.bias.clone()
        self.model.features[0] = new_conv
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class DenseNetModel(nn.Module):
    def __init__(self, model_name='densenet121', num_classes=None, pretrained=True):
        super(DenseNetModel, self).__init__()
        if num_classes is None:
            num_classes = NUM_CLASSES
        self.model = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained, in_chans=1)

    def forward(self, x):
        return self.model(x)

class MobileNetModel(nn.Module):
    def __init__(self, model_name='mobilenetv3_large_100', num_classes=None, pretrained=True):
        super(MobileNetModel, self).__init__()
        if num_classes is None:
            num_classes = NUM_CLASSES
        if model_name in ['mobilenetv1_100', 'mobilenetv2_100']:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.mobilenet_v2(weights=weights)
            original_conv = self.model.features[0][0]
            new_conv = nn.Conv2d(1, original_conv.out_channels, kernel_size=original_conv.kernel_size,
                                 stride=original_conv.stride, padding=original_conv.padding,
                                 bias=original_conv.bias is not None)
            if pretrained:
                with torch.no_grad():
                    new_conv.weight[:, 0, :, :] = original_conv.weight.mean(dim=1)
                    if original_conv.bias is not None:
                        new_conv.bias = original_conv.bias.clone()
            self.model.features[0][0] = new_conv
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        else:
            self.model = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained, in_chans=1)

    def forward(self, x):
        return self.model(x)

class SqueezeNetModel(nn.Module):
    def __init__(self, num_classes=None, pretrained=True):
        super(SqueezeNetModel, self).__init__()
        if num_classes is None:
            num_classes = NUM_CLASSES
        weights = models.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.squeezenet1_1(weights=weights)
        original_conv = self.model.features[0]
        new_conv = nn.Conv2d(1, original_conv.out_channels, kernel_size=original_conv.kernel_size,
                             stride=original_conv.stride, padding=original_conv.padding,
                             bias=original_conv.bias is not None)
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:, 0, :, :] = original_conv.weight.mean(dim=1)
                if original_conv.bias is not None:
                    new_conv.bias = original_conv.bias.clone()
        self.model.features[0] = new_conv
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.model.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

class EfficientNetModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=None, pretrained=True):
        super(EfficientNetModel, self).__init__()
        if num_classes is None:
            num_classes = NUM_CLASSES
        self.model = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained, in_chans=1)

    def forward(self, x):
        return self.model(x)

class InceptionModel(nn.Module):
    def __init__(self, model_name='inception_v3', num_classes=None, pretrained=True):
        super(InceptionModel, self).__init__()
        if num_classes is None:
            num_classes = NUM_CLASSES
        if model_name == 'inception_v3':
            weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.inception_v3(weights=weights)
            original_conv = self.model.Conv2d_1a_3x3.conv
            new_conv = nn.Conv2d(1, original_conv.out_channels, kernel_size=original_conv.kernel_size,
                                 stride=original_conv.stride, padding=original_conv.padding,
                                 bias=original_conv.bias is not None)
            if pretrained:
                with torch.no_grad():
                    new_conv.weight[:, 0, :, :] = original_conv.weight.mean(dim=1)
                    if original_conv.bias is not None:
                        new_conv.bias = original_conv.bias.clone()
            self.model.Conv2d_1a_3x3.conv = new_conv
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, num_classes)
        elif model_name == 'inception_v4':
            try:
                self.model = timm.create_model('inception_v4', num_classes=num_classes, pretrained=pretrained, in_chans=1)
            except:
                weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
                self.model = models.inception_v3(weights=weights)
                original_conv = self.model.Conv2d_1a_3x3.conv
                new_conv = nn.Conv2d(1, original_conv.out_channels, kernel_size=original_conv.kernel_size,
                                     stride=original_conv.stride, padding=original_conv.padding,
                                     bias=original_conv.bias is not None)
                if pretrained:
                    with torch.no_grad():
                        new_conv.weight[:, 0, :, :] = original_conv.weight.mean(dim=1)
                        if original_conv.bias is not None:
                            new_conv.bias = original_conv.bias.clone()
                self.model.Conv2d_1a_3x3.conv = new_conv
                self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
                self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, num_classes)
        elif model_name == 'inception_resnet_v2':
            try:
                self.model = timm.create_model('inception_resnet_v2', num_classes=num_classes, pretrained=pretrained, in_chans=1)
            except:
                weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
                self.model = models.inception_v3(weights=weights)
                original_conv = self.model.Conv2d_1a_3x3.conv
                new_conv = nn.Conv2d(1, original_conv.out_channels, kernel_size=original_conv.kernel_size,
                                     stride=original_conv.stride, padding=original_conv.padding,
                                     bias=original_conv.bias is not None)
                if pretrained:
                    with torch.no_grad():
                        new_conv.weight[:, 0, :, :] = original_conv.weight.mean(dim=1)
                        if original_conv.bias is not None:
                            new_conv.bias = original_conv.bias.clone()
                self.model.Conv2d_1a_3x3.conv = new_conv
                self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
                self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, num_classes)

    def forward(self, x):
        if self.training and hasattr(self.model, 'AuxLogits'):
            outputs = self.model(x)
            if isinstance(outputs, tuple):
                return outputs[0]  # Ana çıktıyı döndür
            return outputs
        else:
            return self.model(x)

class GoogLeNetModel(nn.Module):
    def __init__(self, num_classes=None, pretrained=True):
        super(GoogLeNetModel, self).__init__()
        if num_classes is None:
            num_classes = NUM_CLASSES
        weights = models.GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.googlenet(weights=weights)
        original_conv = self.model.conv1.conv
        new_conv = nn.Conv2d(1, original_conv.out_channels, kernel_size=original_conv.kernel_size,
                             stride=original_conv.stride, padding=original_conv.padding,
                             bias=original_conv.bias is not None)
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:, 0, :, :] = original_conv.weight.mean(dim=1)
                if original_conv.bias is not None:
                    new_conv.bias = original_conv.bias.clone()
        self.model.conv1.conv = new_conv
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class ShuffleNetModel(nn.Module):
    def __init__(self, model_name='shufflenet_v2_x1_0', num_classes=None, pretrained=True):
        super(ShuffleNetModel, self).__init__()
        if num_classes is None:
            num_classes = NUM_CLASSES
        if model_name == 'shufflenet_v2_x0_5':
            weights = models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.shufflenet_v2_x0_5(weights=weights)
        elif model_name == 'shufflenet_v2_x1_0':
            weights = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.shufflenet_v2_x1_0(weights=weights)
        elif model_name == 'shufflenet_v2_x1_5':
            weights = models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.shufflenet_v2_x1_5(weights=weights)
        else:
            weights = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.shufflenet_v2_x1_0(weights=weights)
        original_conv = self.model.conv1[0]
        new_conv = nn.Conv2d(1, original_conv.out_channels, kernel_size=original_conv.kernel_size,
                             stride=original_conv.stride, padding=original_conv.padding,
                             bias=original_conv.bias is not None)
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:, 0, :, :] = original_conv.weight.mean(dim=1)
                if original_conv.bias is not None:
                    new_conv.bias = original_conv.bias.clone()
        self.model.conv1[0] = new_conv
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class ViTModel(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', num_classes=None, pretrained=True):
        super(ViTModel, self).__init__()
        if num_classes is None:
            num_classes = NUM_CLASSES
        self.model = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained, in_chans=1)

    def forward(self, x):
        return self.model(x)

class RegNetModel(nn.Module):
    def __init__(self, model_name='regnetx_002', num_classes=None, pretrained=True):
        super(RegNetModel, self).__init__()
        if num_classes is None:
            num_classes = NUM_CLASSES
        self.model = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained, in_chans=1)

    def forward(self, x):
        return self.model(x)

class SENetModel(nn.Module):
    def __init__(self, model_name='legacy_seresnet18', num_classes=None, pretrained=True):
        super(SENetModel, self).__init__()
        if num_classes is None:
            num_classes = NUM_CLASSES
        self.model = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained, in_chans=1)

    def forward(self, x):
        return self.model(x)

class XceptionModel(nn.Module):
    def __init__(self, num_classes=None, pretrained=True):
        super(XceptionModel, self).__init__()
        if num_classes is None:
            num_classes = NUM_CLASSES
        self.model = timm.create_model('xception', num_classes=num_classes, pretrained=pretrained, in_chans=1)

    def forward(self, x):
        return self.model(x)

class NASNetModel(nn.Module):
    def __init__(self, model_name='nasnetalarge', num_classes=None, pretrained=True):
        super(NASNetModel, self).__init__()
        if num_classes is None:
            num_classes = NUM_CLASSES
        self.model = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained, in_chans=1)

    def forward(self, x):
        return self.model(x)

def get_all_model_configs(num_classes=2):
    if num_classes is None:
        num_classes = NUM_CLASSES
    return {
        'ResNet18': lambda: ResNetModel('resnet18', num_classes),
        'ResNet34': lambda: ResNetModel('resnet34', num_classes),
        'ResNet50': lambda: ResNetModel('resnet50', num_classes),
        'ResNet101': lambda: ResNetModel('resnet101', num_classes),
        'ResNet152': lambda: ResNetModel('resnet152', num_classes),
        'VGG16': lambda: VGGModel('vgg16', num_classes),
        'VGG19': lambda: VGGModel('vgg19', num_classes),
        'AlexNet': lambda: AlexNetModel(num_classes),
        'DenseNet121': lambda: DenseNetModel('densenet121', num_classes),
        'DenseNet169': lambda: DenseNetModel('densenet169', num_classes),
        'DenseNet201': lambda: DenseNetModel('densenet201', num_classes),
        'MobileNetV1': lambda: MobileNetModel('mobilenetv1_100', num_classes),
        'MobileNetV2': lambda: MobileNetModel('mobilenetv2_100', num_classes),
        'MobileNetV3': lambda: MobileNetModel('mobilenetv3_large_100', num_classes),
        'SqueezeNet': lambda: SqueezeNetModel(num_classes),
        'EfficientNet_B0': lambda: EfficientNetModel('efficientnet_b0', num_classes),
        'EfficientNet_B1': lambda: EfficientNetModel('efficientnet_b1', num_classes),
        'EfficientNet_B2': lambda: EfficientNetModel('efficientnet_b2', num_classes),
        'InceptionResNet_v2': lambda: InceptionModel('inception_resnet_v2', num_classes),
        'GoogLeNet': lambda: GoogLeNetModel(num_classes),
        'ShuffleNet_v2_x0_5': lambda: ShuffleNetModel('shufflenet_v2_x0_5', num_classes),
        'ShuffleNet_v2_x1_0': lambda: ShuffleNetModel('shufflenet_v2_x1_0', num_classes),
        'ShuffleNet_v2_x1_5': lambda: ShuffleNetModel('shufflenet_v2_x1_5', num_classes),
        'ViT_Base': lambda: ViTModel('vit_base_patch16_224', num_classes),
        'ViT_Small': lambda: ViTModel('vit_small_patch16_224', num_classes),
    }