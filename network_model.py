import torch
import torch.nn as nn
from generator import Generator
from discriminator import Discriminator
from img_dataloader import make_datapath_list, ImageTransform, GanImgDataset

import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def weights_init(m):
    """
    ネットワークを初期化する関数
    """
    # DCGANでは以下のような初期化するとうまくいくことが多い
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # Conv2dとConvTranspose2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)  # 転置畳み込みと畳み込み層の重みは平均0、標準偏差0.02の正規分布
    elif classname.find("BatchNorm") != -1:
        # BatchNorm2dの初期化
        nn.init.normal_(m.weight.data, 1.0, 0.02)  # 重みは平均1、標準偏差0.02、の正規分布
        nn.init.constant_(m.bias.data, 0)


def train_model(G, D, dataloader, criterion, d_optimizer, g_optimizer, z_dim):
    """
    モデルを学習させる関数
    Parameters
    ----------
    G: generator
        generaterのネットワークモデル
    D: discriminator
        discriminatorのネットワークモデル
    dataloader:
        データローダ
    """

    # GPUの初期設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス", device)

    G.to(device)
    D.to(device)

    # epochのループ
    # 開始時刻を保存
    t_epoch_start = time.time()
    epoch_g_loss = 0.0  # epochの損失和
    epoch_d_loss = 0.0  # epochの損失和

    # 画像のバッチサイズ
    batch_size = dataloader.batch_size

    # イテレータ
    iters = 0
    img_list = []

    for imges in tqdm(dataloader, leave=False):
        # 1.Discriminatorの学習
        # ミニバッチがサイズが1だと、バッチノーマライゼーションでエラーになるので避ける
        if imges.size()[0] == 1:
            continue
        # GPUが使えるならGPUにデータを送る
        imges = imges.to(device)
        # 正解ラベルと偽ラベルを作成
        # epochの最後のイテレーションはミニバッチの数が少なくなる
        mini_batch_size = imges.size()[0]  # 1次元目のバッチサイズを取得
        label_real = torch.full((mini_batch_size,), 1).to(device)
        label_fake = torch.full((mini_batch_size,), 0).to(device)
        # 真の画像を判定
        d_out_real = D(imges)
        # 偽の画像を生成して判定
        input_z = torch.randn(mini_batch_size, z_dim).to(device)
        input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
        fake_images = G(input_z)
        d_out_fake = D(fake_images)
        # 誤差を計算
        d_loss_real = criterion(d_out_real.view(-1), label_real)
        d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
        d_loss = d_loss_real + d_loss_fake
        # バックプロパゲーション
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        # 2. Generatorの学習
        # 偽の画像を生成して判定
        input_z = torch.randn(mini_batch_size, z_dim).to(device)
        input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
        fake_images = G(input_z)
        d_out_fake = D(fake_images)
        # 誤差を計算
        g_loss = criterion(d_out_fake.view(-1), label_real)
        # バックプロパゲーション
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        # 3.記録
        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()

        if iters % 1 == 0:
            with torch.no_grad():
                fake = G(input_z).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
    # epochのphaseごとのlossと正解率
    t_epoch_finish = time.time()
    print("----------")
    print(
        "Epoch_D_Loss:{:.4f} ||Epoch_G_Loss:{:.4f}".format(
            epoch_d_loss / batch_size, epoch_g_loss / batch_size
        )
    )
    print("timer: {:.4f} sec.".format(t_epoch_finish - t_epoch_start))
    t_epoch_start = time.time()
    return G, D, epoch_g_loss / batch_size, epoch_d_loss / batch_size, img_list


def main():
    # ファイルリストを作成
    train_img_list = make_datapath_list()

    # Datasetを作成
    mean = (0.5,)
    std = (0.5,)

    train_dataset = GanImgDataset(
        file_list=train_img_list, transform=ImageTransform(mean, std)
    )
    # DataLoaderを作成
    batch_size = 64

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    G = Generator(z_dim=20, image_size=64)
    D = Discriminator(z_dim=20, image_size=64)

    # torch.nn.Moduleの関数apply パラメータの重みを初期化
    G.apply(weights_init)
    D.apply(weights_init)

    print("ネットワークの初期化完了")

    # 学習・検証を実行する
    num_epochs = 200
    G_update, D_update = train_model(
        G, D, dataloader=train_dataloader, num_epochs=num_epochs
    )

    # 生成画像と訓練データを可視化する
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 入力乱数生成
    batch_size = 8
    z_dim = 20
    fixed_z = torch.randn(batch_size, z_dim)
    fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)

    # 画像生成
    fake_imges = G_update(fixed_z.to(device))

    # 訓練データ
    batch_iterator = iter(train_dataloader)  # iteratorに変換
    imges = next(batch_iterator)

    # 出力
    fig = plt.figure(figsize=(15, 6))

    for i in range(0, 5):
        # 上団に訓練データを
        ax = fig.add_subplot(2, 5, i + 1)
        ax.imshow(imges[i][0].cpu().detach().numpy(), "gray")

        # 下段に生成データ
        ax = fig.add_subplot(2, 5, 5 + i + 1)
        ax.imshow(fake_imges[i][0].cpu().detach().numpy(), "gray")

    fig.savefig("generate_image")


if __name__ == "__main__":
    main()
