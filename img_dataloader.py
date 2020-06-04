import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

import pathlib


def make_datapath_list():
    """
    学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する.
    Returns : list
        画像のファイル名が入ったリスト
    """
    current_dir = pathlib.Path(__file__).resolve().parent

    train_img_list = list()  # 画像ファイルパスを格納

    for img_idx in range(200):
        img_path = str(current_dir) + "/data/img_78/img_7_" + str(img_idx) + ".jpg"
        train_img_list.append(img_path)

        img_path = str(current_dir) + "/data/img_78/img_8_" + str(img_idx) + ".jpg"
        train_img_list.append(img_path)

    return train_img_list


class ImageTransform:
    # 画像の前処理クラス

    def __init__(self, mean, std):
        self.data_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    def __call__(self, img):
        return self.data_transform(img)


class GanImgDataset(data.Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        # 画像の枚数を返す
        return len(self.file_list)

    def __getitem__(self, index):
        # 前処理した画像のTensor形式のデータを取得

        img_path = self.file_list[index]
        img = Image.open(img_path)  # 「高さ」「幅」白黒

        # 画像の前処理
        img_transformed = self.transform(img)

        return img_transformed


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

    batch_iterator = iter(train_dataloader)
    images = next(batch_iterator)
    print(images.size()[0])


if __name__ == "__main__":

    main()
