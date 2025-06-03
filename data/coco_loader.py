import os
import json
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, num_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.num_samples = num_samples

        with open(annotation_file) as f:
            self.annotations = json.load(f)

        self.image_captions = []
        image_id_to_filename = {img['id']: img['file_name'] for img in self.annotations['images']}

        for ann in self.annotations['annotations']:
            if ann['image_id'] in image_id_to_filename:
                self.image_captions.append({
                    'image_id': ann['image_id'],
                    'file_name': image_id_to_filename[ann['image_id']],
                    'caption': ann['caption']
                })

        if num_samples is not None:
            self.image_captions = self.image_captions[:num_samples]

        self._prepare_diversity_subsets()

    def _prepare_diversity_subsets(self):
        sorted_captions = sorted(self.image_captions, key=lambda x: len(x['caption']))
        self.short_captions = [x['caption'] for x in sorted_captions[:1000]]
        self.long_captions = [x['caption'] for x in sorted_captions[-1000:]]

        if self.num_samples <= 4:
            self.short_captions = self.short_captions[:1]
            self.long_captions = []

    def __len__(self):
        return len(self.image_captions)

    def __getitem__(self, idx):
        img_info = self.image_captions[idx]
        img_path = os.path.join(self.root_dir, img_info['file_name'])

        try:
            image = Image.open(img_path).convert('RGB')
            caption = img_info['caption']

            if self.transform:
                image = self.transform(image)

            return image, caption
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None


def load_coco_data_batched(split='val', num_samples=None, image_size=512, batch_size=64):
    """
    Batched version of COCO data loader

    Args:
        split: 'val' or 'train'
        num_samples: Total samples to load
        image_size: Image resize dimension
        batch_size: Number of samples per batch

    Returns:
        Generator yielding batches of (images, captions)
    """

    root_dir = 'data/val2014' if split == 'val' else 'data/train2014'
    annotation_file = f'data/annotations/captions_{split}2014.json'

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    dataset = CocoDataset(
        root_dir=root_dir,
        annotation_file=annotation_file,
        transform=transform,
        num_samples=num_samples
    )

    def batch_generator():
        current_batch = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample is not None:
                current_batch.append(sample)
                if len(current_batch) >= batch_size:
                    images, captions = zip(*current_batch)
                    yield torch.stack(images), list(captions)
                    current_batch = []

        if current_batch:  # Yield remaining samples
            images, captions = zip(*current_batch)
            yield torch.stack(images), list(captions)

    return {
        'batches': batch_generator(),
        'long_captions': dataset.long_captions,
        'short_captions': dataset.short_captions,
        'total_samples': len(dataset)
    }
