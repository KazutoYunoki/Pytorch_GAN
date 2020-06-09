import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

import os.path as osp
import pathlib
import glob


def make_datapath_list(data_dir):
    """
    学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する.
    Returns : list
        画像のファイル名が入ったリスト
    """
    current_dir = pathlib.Path(__file__).resolve().parent

    target_path = osp.join(str(current_dir) + "/data/" + data_dir + "/**/**/*.jpg")
    print(target_path)

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    print(len(path_list))

    return path_list


class ImageTransform:
    # 画像の前処理クラス

    def __init__(self, resize, mean, std):
        self.data_transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
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
    train_img_list = make_datapath_list("cat_dog")

    # Datasetを作成
    size = 224
    mean = (0.458, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_dataset = GanImgDataset(
        file_list=train_img_list, transform=ImageTransform(size, mean, std)
    )
    # DataLoaderを作成
    batch_size = 32

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    batch_iterator = iter(train_dataloader)
    images = next(batch_iterator)
    print(images.size()[0])


if __name__ == "__main__":

    main()
