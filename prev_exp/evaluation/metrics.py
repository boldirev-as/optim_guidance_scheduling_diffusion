import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import open_clip
from transformers import AutoModel
import torchvision.transforms as T
import torch.nn.functional as F
from hpsv2 import img_score
import huggingface_hub
from transformers import AutoProcessor, AutoModel as HFModel

from prev_exp.utils import calc_eigvals

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

        # HPS v2.1
        self.hps_model, self.hps_preprocess, _ = img_score.create_model_and_transforms(
            "ViT-H-14", "laion2B-s32B-b79K", precision="amp", device=device if device != 'mps' else 'cpu',
            jit=False, force_quick_gelu=False, force_custom_text=False, force_patch_dropout=False,
            force_image_size=None, pretrained_image=False, image_mean=None, image_std=None,
            light_augmentation=True, aug_cfg={}, output_dict=True, with_score_predictor=False,
            with_region_predictor=False,
        )
        cp = huggingface_hub.hf_hub_download("xswu/HPSv2", "HPS_v2.1_compressed.pt")
        checkpoint = torch.load(cp, map_location=device if device != 'mps' else 'cpu')
        self.hps_model.load_state_dict(checkpoint["state_dict"])
        self.hps_model.eval()
        self.hps_tokenizer = img_score.get_tokenizer("ViT-H-14")

        # PickScore
        self.pick_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.pick_model = HFModel.from_pretrained("yuvalkirstain/PickScore_v1").to(device if device != 'mps' else 'cpu').eval()

        # Для LSCD: копим признаки реальных и сгенерированных картинок
        self.real_features = []
        self.gen_features = []

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

    def compute_hps_batch(self, images: torch.Tensor, captions: list[str]):
        # images в [0,1], shape B,C,H,W
        imgs = self.hps_preprocess(images.to(self.device if self.device != 'mps' else 'cpu'))
        tokens = self.hps_tokenizer(captions).to(self.device if self.device != 'mps' else 'cpu')
        with torch.no_grad():
            outputs = self.hps_model(imgs, tokens)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits_per_image = image_features @ text_features.T
            return torch.diag(logits_per_image).detach().cpu().tolist()

    def compute_pickscore_batch(self, images: torch.Tensor, captions: list[str]):
        # images в [0,1]
        inputs = self.pick_processor(
            images=images, text=captions, return_tensors="pt", padding="max_length", truncation=True
        ).to(self.device if self.device != 'mps' else 'cpu')
        with torch.no_grad():
            image_embs = self.pick_model.get_image_features(pixel_values=inputs["pixel_values"])
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            text_embs = self.pick_model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
            scores = self.pick_model.logit_scale.exp() * torch.diag(text_embs @ image_embs.T)
            return scores.detach().cpu().tolist()

    def add_real_features(self, images: torch.Tensor):
        imgs = torch.stack([self.dino_transform(img.to(self.device if self.device != 'mps' else 'cpu')) for img in images])
        self.real_features.append(imgs.to(self.device if self.device != 'mps' else 'cpu'))

    def add_generated_features(self, images: torch.Tensor):
        imgs = torch.stack([self.dino_transform(img.to(self.device if self.device != 'mps' else 'cpu')) for img in images])
        self.gen_features.append(imgs.to(self.device if self.device != 'mps' else 'cpu'))

    def compute_lscd(self):
        if not self.real_features or not self.gen_features:
            return None
        with torch.no_grad():
            real_feats = []
            for batch in self.real_features:
                feats = self.dino_model(batch if self.device != 'mps' else batch.cpu()).last_hidden_state.mean(dim=1)
                real_feats.append(feats.cpu())
            gen_feats = []
            for batch in self.gen_features:
                feats = self.dino_model(batch if self.device != 'mps' else batch.cpu()).last_hidden_state.mean(dim=1)
                gen_feats.append(feats.cpu())
            real_feats = torch.cat(real_feats, dim=0)
            gen_feats = torch.cat(gen_feats, dim=0)
            ref_eigs = calc_eigvals(real_feats)
            cur_eigs = calc_eigvals(gen_feats)
            return torch.sum((torch.log(ref_eigs) - torch.log(cur_eigs)) ** 2).item()


# Test the implementation
if __name__ == "__main__":
    evaluator = Evaluator(device="cuda" if torch.cuda.is_available() else "mps")
    print("Evaluator initialized successfully!")
