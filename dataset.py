# dataset.py
import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F

class SampahDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms

        with open(annotation_file, 'r') as f:
            coco = json.load(f)

        self.img_id_to_info = {img['id']: img for img in coco['images']}

        self.img_id_to_anns = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)

        self.img_ids = [
            img_id for img_id in self.img_id_to_info
            if img_id in self.img_id_to_anns
        ]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.img_id_to_info[img_id]
        anns = self.img_id_to_anns[img_id]

        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        W, H = img.size

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in anns:
            # Bounding box: COCO format [x, y, w, h] → [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h
            if x2 <= x1 or y2 <= y1:
                continue  # skip bbox tidak valid

            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])  # 1..6
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])

            mask = self._poly_to_mask(ann['segmentation'], H, W)
            masks.append(mask)

        if len(boxes) == 0:
            return None

        # Batasi maksimum objek per gambar (cegah memory error)
        max_objects = 10
        if len(boxes) > max_objects:
            boxes = boxes[:max_objects]
            labels = labels[:max_objects]
            masks = masks[:max_objects]
            areas = areas[:max_objects]
            iscrowd = iscrowd[:max_objects]

        target = {
            'boxes'   : torch.as_tensor(boxes,    dtype=torch.float32),
            'labels'  : torch.as_tensor(labels,   dtype=torch.int64),
            'masks'   : torch.as_tensor(np.array(masks), dtype=torch.uint8),
            'image_id': torch.tensor([img_id]),
            'area'    : torch.as_tensor(areas,    dtype=torch.float32),
            'iscrowd' : torch.as_tensor(iscrowd,  dtype=torch.int64),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def _poly_to_mask(self, segmentation, H, W):
        """Konversi polygon COCO → numpy binary mask (H, W)"""
        import numpy as np
        from PIL import ImageDraw

        mask_img = Image.new('L', (W, H), 0)
        draw = ImageDraw.Draw(mask_img)
        for poly in segmentation:
            points = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
            draw.polygon(points, outline=1, fill=1)
        return np.array(mask_img, dtype=np.uint8)

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return tuple(zip(*batch))

class SimpleTransform:
    def __init__(self, train=False):
        self.train = train

    def __call__(self, image, target):
        import random
        from torchvision.transforms import ColorJitter, GaussianBlur

        if self.train:
            # 1. Horizontal Flip (50%)
            if random.random() > 0.5:
                W = image.width
                image = F.hflip(image)
                boxes = target['boxes']
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
                target['boxes'] = boxes
                target['masks'] = target['masks'].flip(-1)

            # 2. Vertical Flip (30%)
            if random.random() > 0.7:
                H = image.height
                image = F.vflip(image)
                boxes = target['boxes']
                boxes[:, [1, 3]] = H - boxes[:, [3, 1]]
                target['boxes'] = boxes
                target['masks'] = target['masks'].flip(-2)

            # 3. Color Jitter — lebih agresif untuk variasi cahaya drone
            if random.random() > 0.3:
                jitter = ColorJitter(
                    brightness=0.4, contrast=0.4,
                    saturation=0.3, hue=0.1
                )
                image = jitter(image)

            # 4. Gaussian Blur (20%)
            if random.random() > 0.8:
                blur = GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
                image = blur(image)

        image = F.to_tensor(image)
        return image, target
