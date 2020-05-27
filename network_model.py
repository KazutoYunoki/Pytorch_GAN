import torch.nn as nn
from generator import Generator
from discriminator import Discriminator


# ネットワークの初期化 DCGANでは以下のような初期化するとうまくいくことが多い
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # Conv2dとConvTranspose2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)  # 転置畳み込みと畳み込み層の重みは平均0、標準偏差0.02の正規分布
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        # BatchNorm2dの初期化
        nn.init.normal_(m.weight.data, 1.0, 0.02)  # 重みは平均1、標準偏差0.02、の正規分布
        nn.init.constant_(m.bias.data, 0)


def main():

    G = Generator(z_dim=20, image_size=64)
    D = Discriminator(z_dim=20, image_size=64)

    # torch.nn.Moduleの関数apply パラメータの重みを初期化
    G.apply(weights_init)
    D.apply(weights_init)

    print("ネットワークの初期化完了")


if __name__ == "__main__":
    main()
