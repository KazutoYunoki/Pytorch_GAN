import hydra
import logging
import matplotlib.pyplot as plt

import torch

from img_dataloader import make_datapath_list, GanImgDataset, ImageTransform
from generator import Generator
from discriminator import Discriminator
from network_model import weights_init, train_descriminator, train_generator


# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    # ファイルリストを作成
    train_img_list = make_datapath_list()

    # データセットを作成
    train_dataset = GanImgDataset(
        file_list=train_img_list,
        transform=ImageTransform(cfg.image.mean, cfg.image.std),
    )

    # データローダを作成
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.image.batch_size, shuffle=True
    )

    G = Generator(z_dim=cfg.input.z_dim, image_size=cfg.image.size)
    D = Discriminator(z_dim=cfg.input.z_dim, image_size=cfg.image.size)

    # torch.nn.Moduleの関数apply パラメータの重みを初期化
    G.apply(weights_init)
    D.apply(weights_init)

    # 学習回数を取得
    num_epochs = cfg.train.num_epochs

    # 学習・検証を実行
    for epoch in range(num_epochs):
        print("----------")
        print("Epoch {} / {}".format(epoch, num_epochs))

        D_update = train_descriminator(
            D, G, dataloader=train_dataloader, num_epochs=num_epochs
        )
        G_update = train_generator(
            G, D, dataloader=train_dataloader, num_epochs=num_epochs
        )

    # 生成画像と訓練データを可視化する
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 入力乱数生成
    fixed_z = torch.randn(cfg.input.batch_size, cfg.input.z_dim)
    fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)

    # 画像生成
    fake_images = G_update(fixed_z.to(device))

    # 訓練データ
    batch__iterator = iter(train_dataloader)
    images = next(batch__iterator)

    # 出力
    fig = plt.figure(figsize=(15, 6))

    for i in range(0, 5):
        # 上段に訓練データ
        fig.subplot(2, 5, i + 1)
        fig.imshow(images[i][0].cpu().detach().numpy(), "gray")

        fig.subplot(2, 5, 5 + i + 1)
        fig.imshow(fake_images[i][0].cpu().detach().numpy(), "gray")

    fig.savefig("generate_image")


if __name__ == "__main__":
    main()
