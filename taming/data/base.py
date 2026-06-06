import bisect
import numpy as np
import albumentations
import os.path as osp
import cv2
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
                
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs
            
        self.strong_preprocessor = albumentations.Compose([
                albumentations.OneOf([
                    albumentations.Morphological(op=cv2.MORPH_DILATE, kernel=(2, 2), p=1.0), # 굵게
                    albumentations.Morphological(op=cv2.MORPH_ERODE, kernel=(2, 2), p=1.0),  # 얇게
                ], p=0.5),
                albumentations.ElasticTransform(alpha=2, sigma=50, alpha_affine=50, p=0.8),
                albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                albumentations.RandomBrightnessContrast(p=0.5)
            ])

    def __len__(self):
        return self._length

    # def preprocess_image(self, image_path):
    #     basename = osp.basename(osp.dirname(image_path))
            
    #     image = Image.open(image_path)
    #     image = image.convert('L')
    #     image = np.array(image).astype(np.uint8)
    #     image = self.preprocessor(image=image)["image"]
    #     if basename == "reference_images_v2":
    #         image = self.strong_preprocessor(image=image)["image"]
            
    #     image = (image/127.5 - 1.0).astype(np.float32)
    #     return image
    
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('L')
        image = np.array(image).astype(np.uint8)
        
        return self._apply_augmentation(image, image_path)


    def _apply_augmentation(self, image, image_path):
        image = self.preprocessor(image=image)["image"]
        
        basename = osp.basename(osp.dirname(image_path))
        if basename == "reference_images_v2":
            image = self.strong_preprocessor(image=image)["image"]
            
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
        

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
