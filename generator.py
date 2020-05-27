import torch
import torch.nn as nn

import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, z_dim=20, image_size=64):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, image_size * 8, kernel_size=4, stride=1),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                image_size * 8, image_size * 4, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(
                image_size * 4, image_size * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(
                image_size * 2, image_size * 1, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True),
        )

        self.last = nn.Sequential(
            nn.ConvTranspose2d(image_size, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out


def main():
    # 動作確認

    G = Generator(z_dim=20, image_size=64)

    # 入力する乱数
    input_z = torch.randn(1, 20)

    # テンソルサイズを(1, 20, 1, 1)に変形
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)

    # 偽画像を出力
    fake_images = G(input_z)

    img_transformed = fake_images[0][0].detach().numpy()
    plt.imshow(img_transformed, "gray")
    plt.show()


if __name__ == "__main__":
    main()
