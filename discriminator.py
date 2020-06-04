import torch.nn as nn
import torch
from generator import Generator


class Discriminator(nn.Module):
    def __init__(self, z_dim=20, image_size=64, nc=3):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(nc, image_size, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(image_size, image_size * 2, kernel_size=4, stride=2, padding=1,),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(
                image_size * 2, image_size * 4, kernel_size=4, stride=2, padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(
                image_size * 4, image_size * 8, kernel_size=4, stride=2, padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.last = nn.Sequential(
            nn.Conv2d(image_size * 8, 1, kernel_size=4, stride=1), nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out


def main():
    # 動作確認
    D = Discriminator(z_dim=20, image_size=64)
    G = Generator(z_dim=20, image_size=64)

    # 偽画像を生成
    input_z = torch.randn(1, 20)
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
    fake_images = G(input_z)

    # 偽画像をDに入力
    d_out = D(fake_images)

    # 出力d_outにSigmoidをかけて0から1に変換
    print(nn.Sigmoid()(d_out))


if __name__ == "__main__":
    main()
