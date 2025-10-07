# %%
import os
import torch
import wandb
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm
from accelerate import infer_auto_device_map
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from datasets import DatasetDict, Dataset
from jiwer import wer, cer
from collections import Counter

# %%
os.makedirs("outputs", exist_ok=True)
os.makedirs("wandb_runs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)


# %%
# WandB оффлайн
# os.environ["WANDB_MODE"] = "offline"
# os.environ["WANDB_DIR"] = "./wandb_runs"
# os.environ["WANDB_CACHE_DIR"] = "./wandb_runs/cache"

wandb.init(
    project="gemma-3-ocr-training",
    name="gemma-3-4b-it-experiment",
    config={
        "model": "gemma-3-4b-it",
        "task": "OCR Text Recognition",
        "dataset": "ocr_dataset_local",
        "lora_rank": 8,
        "lora_alpha": 16,
        "learning_rate": 2e-4,
        "batch_size": 2,
        "gradient_accumulation": 4,
        "epochs": 1
    }
)

# %%
dataset_path = "./datasets/ocr_dataset_local"
dataset = load_from_disk(dataset_path)

dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = dataset["train"], dataset["test"]


wandb.log({
    "dataset_size": len(train_ds),
    "dataset_columns": list(train_ds.features.keys())
})


# %%
def log_dataset_samples(dataset, num_samples=5):
    images = []
    captions = []

    for i in range(min(num_samples, len(dataset))):
        image = dataset[i]['image']
        text = dataset[i]['text']
        images.append(wandb.Image(image, caption=f"Text: {text}"))
        captions.append(text)

    wandb.log({
        "dataset_samples": images,
        "sample_texts": captions
    })

# %%
instruction = 'Расшифруй что написано'


def convert_to_conversation(sample):
    conversation = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': instruction},
                {'type': 'image', 'image': sample['image']}
            ]
        },
        {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': sample['text']}
            ]
        },
    ]
    return {'messages': conversation}


# %%
converted_dataset_train = [convert_to_conversation(s) for s in train_ds]
converted_dataset_eval = [convert_to_conversation(s) for s in eval_ds]

# %%
model_path = "./models/gemma-3-4b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

wandb.config.update({
    "quantization_config": {
        "load_in_4bit": True,
        "double_quant": True,
        "quant_type": "nf4",
        "compute_dtype": "bfloat16"
    }
})

print("Загружаем модель и процессор с локального диска...")

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    device_map="balanced",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

model.config.use_cache = False

processor = AutoProcessor.from_pretrained(
    model_path,
    local_files_only=True
)

wandb.log({
    "model_loaded": True,
    "model_name": "gemma-3-4b-it",
    "device_map": str(model.hf_device_map),
    "parameter_count": sum(p.numel() for p in model.parameters())
})

print("Gemma-3-4B-IT успешно загружена с локального диска!")

# %%
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0,
    r=8,
    bias="none",
    target_modules=[
        'down_proj', 'o_proj', 'k_proj', 'q_proj',
        'gate_proj', 'up_proj', 'v_proj'
    ],
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)

wandb.config.update({
    "lora_alpha": 16,
    "lora_dropout": 0,
    "lora_rank": 8,
    "lora_target_modules": [
        'down_proj', 'o_proj', 'k_proj', 'q_proj',
        'gate_proj', 'up_proj', 'v_proj'
    ]
})

# %%
args = SFTConfig(
    per_device_train_batch_size=2,
    eval_strategy="steps",
    per_device_eval_batch_size=1,
    torch_empty_cache_steps=5,
    eval_steps=40,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    num_train_epochs=10,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    report_to="wandb",  # вместо none
    logging_first_step=True,
    logging_nan_inf_filter=False,
    save_strategy='steps',
    save_steps=40,
    save_total_limit=4,
    optim='adamw_torch',
    weight_decay=0.01,
    lr_scheduler_type='linear',
    seed=3407,
    output_dir='outputs',
    remove_unused_columns=False,
    dataset_text_field='',
    dataset_kwargs={'skip_prepare_dataset': True},
)

wandb.config.update({
    "train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "epochs": 1,
    "warmup_steps": 10,
    "optimizer": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler": "linear"
})


# %%
def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        for element in content:
            if isinstance(element, dict) and (
                    "image" in element or element.get("type") == "image"
            ):
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                try:
                    image_inputs.append(image.convert("RGB"))
                except Exception:
                    raise Exception(messages)
    return image_inputs


def collate_fn(examples):
    texts = []
    images = []

    # Специальный токен для изображения
    boi_token = processor.tokenizer.special_tokens_map.get("boi_token", processor.tokenizer.bos_token)

    for ex in examples:
        # достаём все картинки из сообщений
        image_inputs = process_vision_info(ex["messages"])
        images.append(image_inputs)

        # собираем текст только из text-частей
        text_parts = []
        for msg in ex["messages"]:
            for elem in msg.get("content", []):
                if elem.get("type") == "text":
                    text_parts.append(elem["text"])

        text_content = " ".join(text_parts).strip()
        texts.append(f"{boi_token} {text_content}")

    # прогон через процессор
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # подготовка лейблов
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # игнорируем специальные токены для изображения
    if "boi_token" in processor.tokenizer.special_tokens_map:
        image_token_id = processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
        if isinstance(image_token_id, list):
            for tid in image_token_id:
                labels[labels == tid] = -100
        else:
            labels[labels == image_token_id] = -100

    labels[labels == 262144] = -100  # защита от случайных паддингов
    batch["labels"] = labels

    return batch


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Декодируем в строки
    preds = processor.batch_decode(predictions, skip_special_tokens=True)
    refs  = processor.batch_decode(labels, skip_special_tokens=True)

    # Считаем WER и CER
    wer_values = [wer(r, p) for r, p in zip(refs, preds)]
    cer_values = [cer(r, p) for r, p in zip(refs, preds)]

    mean_wer = sum(wer_values) / len(wer_values)
    mean_cer = sum(cer_values) / len(cer_values)

    return {
        "wer": mean_wer,
        "cer": mean_cer,
    }


# %%
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=converted_dataset_train,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    eval_dataset=converted_dataset_eval,
)

wandb.watch(model, log="all", log_freq=20)
wandb.log({"training_status": "started"})

print("Начинаем тренировку...")

# %%
try:
    training_results = trainer.train()
    wandb.log({
        "training_status": "completed",
        "final_loss": training_results.metrics['train_loss'],
        "training_time": training_results.metrics['train_runtime'],
        "training_samples_per_second": training_results.metrics['train_samples_per_second'],
        "total_steps": training_results.metrics['train_steps'],
    })

    if hasattr(trainer.state, 'log_history'):
        for log_entry in trainer.state.log_history:
            if 'loss' in log_entry:
                wandb.log({
                    "training_loss": log_entry['loss'],
                    "step": log_entry.get('step', 0)
                })

except Exception as e:
    wandb.log({
        "training_status": "failed",
        "error": str(e)
    })
    raise e

# %%
trainer.save_model()
trainer.save_state()
wandb.log({"model_saved": True})

artifact = wandb.Artifact('trained-gemma-ocr-model', type='model')
artifact.add_dir('outputs')
wandb.log_artifact(artifact)

# %%
gpu_count = torch.cuda.device_count()
gpu_info = []

for i in range(gpu_count):
    gpu_props = torch.cuda.get_device_properties(i)
    allocated = torch.cuda.memory_allocated(i) / 1e9
    reserved = torch.cuda.memory_reserved(i) / 1e9

    gpu_info.append({
        "name": gpu_props.name,
        "memory_total_gb": gpu_props.total_memory / 1e9,
        "memory_allocated_gb": allocated,
        "memory_reserved_gb": reserved,
        "compute_capability": f"{gpu_props.major}.{gpu_props.minor}"
    })

wandb.log({
    "gpu_count": gpu_count,
    "gpu_info": gpu_info
})

print(f"Количество GPU: {gpu_count}")

for i in range(gpu_count):
    wandb.log({
        f"gpu_{i}_final_allocated_gb": torch.cuda.memory_allocated(i) / 1e9,
        f"gpu_{i}_final_reserved_gb": torch.cuda.memory_reserved(i) / 1e9
    })

wandb.finish()

print("Тренировка завершена и все залогировано!")

del model
del trainer
torch.cuda.empty_cache()
