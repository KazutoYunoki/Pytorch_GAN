import hydra
import logging
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
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

    G = Generator(z_dim=cfg.input.z_dim, image_size=cfg.image.size, nc=1)
    D = Discriminator(z_dim=cfg.input.z_dim, image_size=cfg.image.size, nc=1)

    # torch.nn.Moduleの関数apply パラメータの重みを初期化
    G.apply(weights_init)
    D.apply(weights_init)

    # ネットワークの表示
    log.info(G)
    log.info(D)

    # 学習回数を取得
    num_epochs = cfg.train.num_epochs

    # 最適化手法の設定
    g_optimizer = torch.optim.Adam(
        G.parameters(), cfg.optimizer.g_lr, [cfg.optimizer.beta1, cfg.optimizer.beta2]
    )
    d_optimizer = torch.optim.Adam(
        D.parameters(), cfg.optimizer.d_lr, [cfg.optimizer.beta1, cfg.optimizer.beta2]
    )

    # 誤差関数を定義
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    # d_lossとg_lossの損失値を保存しておくリスト
    d_loss_list = []
    g_loss_list = []

    # 学習・検証を実行
    for epoch in range(num_epochs):
        print("----------")
        print("Epoch {} / {}".format(epoch + 1, num_epochs))

        D_update, d_loss = train_descriminator(
            D,
            G,
            dataloader=train_dataloader,
            criterion=criterion,
            d_optimizer=d_optimizer,
            z_dim=cfg.input.z_dim,
        )
        d_loss_list.append(d_loss)
        G_update, g_loss = train_generator(
            G,
            D,
            dataloader=train_dataloader,
            criterion=criterion,
            g_optimizer=g_optimizer,
            z_dim=cfg.input.z_dim,
        )
        g_loss_list.append(g_loss)

    # figインスタンスとaxインスタンスを作成
    fig_loss, ax_loss = plt.subplots(figsize=(10, 10))
    ax_loss.plot(range(1, num_epochs + 1, 1), d_loss_list, label="discriminator_loss")
    ax_loss.plot(range(1, num_epochs + 1, 1), g_loss_list, label="generator_loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend()
    fig_loss.savefig("loss.png")

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
        ax = fig.add_subplot(2, 5, i + 1)
        ax.imshow(images[i][0].cpu().detach().numpy(), "gray")

        ax = fig.add_subplot(2, 5, 5 + i + 1)
        ax.imshow(fake_images[i][0].cpu().detach().numpy(), "gray")

    fig.savefig("generate_image")


if __name__ == "__main__":
    main()
