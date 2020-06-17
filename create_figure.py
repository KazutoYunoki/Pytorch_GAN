import torch
import matplotlib.pyplot as plt


def create_fig(batch_size, z_dim, G, data_loader, epoch):

    # 生成画像と訓練データを可視化する
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 入力乱数生成
    fixed_z = torch.randn(batch_size, z_dim)
    fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)

    # 画像生成
    fake_images = G(fixed_z.to(device))
    # 訓練データ
    batch__iterator = iter(data_loader)
    images = next(batch__iterator)
    # 出力
    fig = plt.figure(figsize=(15, 6))
    # カラー画像用↓
    for i in range(0, 5):
        # 上段に訓練データ
        ax = fig.add_subplot(2, 5, i + 1)
        img = images[i].cpu().detach().numpy().transpose((1, 2, 0))
        img = img / 2 + 0.5
        ax.imshow(img)
        ax = fig.add_subplot(2, 5, 5 + i + 1)
        fake = fake_images[i].cpu().detach().numpy().transpose((1, 2, 0))
        fake = fake / 2 + 0.5
        ax.imshow(fake)

    figure_name = "generate_image_" + str(epoch)
    fig.savefig(figure_name)
