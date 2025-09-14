import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import open_clip
from transformers import AutoModel
import torchvision.transforms as T
import torch.nn.functional as F


class Evaluator:
    def __init__(self, device=None):
        self.device = device
        self.dtype = torch.float16

        self.fid = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(
            device if device != 'mps' else 'cpu')

        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai'
        )
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.clip_model = self.clip_model.to(device if device != 'mps' else 'cpu').eval()

        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(
            device if device != 'mps' else 'cpu'
        ).eval()
        self.dino_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _get_clip_normalization_params(self):
        for transform in self.clip_preprocess.transforms:
            if isinstance(transform, T.Normalize):
                mean = torch.tensor(transform.mean).to(self.device if self.device != 'mps' else 'cpu')
                std = torch.tensor(transform.std).to(self.device if self.device != 'mps' else 'cpu')
                return mean, std
        return None

    def manual_clip_preprocess_tensor(self, images_tensor: torch.Tensor) -> torch.Tensor:
        target_size = self.clip_model.visual.image_size
        clip_mean, clip_std = self._get_clip_normalization_params()

        if images_tensor.ndim == 4:
            clip_mean = clip_mean.view(1, -1, 1, 1)
            clip_std = clip_std.view(1, -1, 1, 1)
        elif images_tensor.ndim == 3:
            clip_mean = clip_mean.view(-1, 1, 1)
            clip_std = clip_std.view(-1, 1, 1)

        images_tensor = F.interpolate(images_tensor, size=target_size, mode='bicubic', antialias=True,
                                      align_corners=False)

        images_tensor = images_tensor.to(torch.float32)
        images_tensor = (images_tensor - clip_mean) / clip_std

        return images_tensor.to(self.device if self.device != 'mps' else 'cpu')

    def compute_clip_score_batch(self, generated_images_tensor: torch.Tensor, captions: list[str]):
        generated_images_tensor = generated_images_tensor.to(
            self.device if self.device != 'mps' else 'cpu')

        image_input = self.manual_clip_preprocess_tensor(generated_images_tensor.to(torch.float32))
        text_input = self.clip_tokenizer(captions).to(self.device if self.device != 'mps' else 'cpu')

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_input)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T)
            clip_scores_per_item = torch.diag(similarity)

            return clip_scores_per_item.tolist()

    def compute_diversity(self, image_list):
        images = torch.stack(
            [self.dino_transform(img.to(self.device if self.device != 'mps' else 'cpu')) for img in image_list]).to(
            self.device)

        if self.device == 'mps':
            images = images.to('cpu')

        # Extract features
        with torch.no_grad():
            features = self.dino_model(images).last_hidden_state.mean(dim=1)

        # Compute pairwise distances
        distances = F.pdist(features, p=2)  # Euclidean distance

        # Diversity score is the average distance
        return distances.mean().item()


# Test the implementation
if __name__ == "__main__":
    evaluator = Evaluator(device="cuda" if torch.cuda.is_available() else "mps")
    print("Evaluator initialized successfully!")
