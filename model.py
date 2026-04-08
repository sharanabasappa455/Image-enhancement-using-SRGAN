import torch
import torch.nn as nn
from torchvision.models import vgg19

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Load VGG19 (pre-trained on ImageNet)
        # We use the features up to the 36th layer (conv5_4) for texture detection
        vgg = vgg19(pretrained=True).features[:36].eval()
        
        # Freeze VGG weights (we don't want to train the judge, just use it)
        for param in vgg.parameters():
            param.requires_grad = False
            
        self.loss_network = vgg

    def forward(self, input, target):
        # Calculate error based on "Features" (shapes/edges) instead of pixels
        input_features = self.loss_network(input)
        target_features = self.loss_network(target)
        return nn.MSELoss()(input_features, target_features)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act=True, **kwargs):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.act = nn.PReLU(num_parameters=out_channels) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.cnn(x))

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor ** 2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        return self.act(self.pixel_shuffle(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, in_channels, use_act=False, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super().__init__()
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels, num_channels, use_act=False, kernel_size=3, stride=1, padding=1)
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels, 2), UpsampleBlock(num_channels, 2))
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial
        x = self.upsamples(x)
        return torch.tanh(self.final(x))

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels, 
                    feature, 
                    kernel_size=3, 
                    stride=1 + idx % 2, 
                    padding=1, 
                    use_act=True  # Changed to LeakyReLU inside ConvBlock if needed, but PReLU is fine
                )
            )
            in_channels = feature
        
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)