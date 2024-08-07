import runpod
from utils import create_error_response
from typing import Any
import torch
from PIL import Image
import open_clip
import aiohttp

model, _, preprocess = open_clip.create_model_and_transforms(
    "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
)
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer("hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K")


async def async_generator_handler(job: dict[str, Any]):
    text = job.get("text")
    image = job.get("image")

    output = {
        "text_embedding": None,
        "image_embedding": None,
    }

    if image:
        async with aiohttp.ClientSession() as session:
            async with session.get(image) as response:
                image = Image.open(await response.read())

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = None
        if image:
            image_input = preprocess(image).unsqueeze(0).cuda()
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            output["image_embedding"] = image_features.cpu().numpy().tolist()

        text_features = None
        if text:
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            output["text_embedding"] = text_features.cpu().numpy().tolist()

    return output


if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": async_generator_handler,
        }
    )
