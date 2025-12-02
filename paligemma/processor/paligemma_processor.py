# Imports
from typing import List, Tuple
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def prepend_image_tokens_to_prompt(prompt, bos_token, num_image_tokens, image_token):
    return f"{image_token * num_image_tokens}{bos_token}{prompt}\n"


def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling,
) -> np.ndarray:
    height, width = size
    resized_image = image.resize(
        (width, height), 
        resample=resample
    )
    return resized_image


def rescale(
    image: np.ndarray, 
    scale: float, 
    dtype: np.dtype = np.float32
) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def normalize(
    image: np.ndarray,
    mean: List[float],
    std: List[float]
) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image


def process_images(
    images: List[Image.Image],
    size: Tuple[int, int],
    resample: Image.Resampling,
    rescale_factor: float,
    image_mean: List[float],
    image_std: List[float]
) -> List[np.ndarray]:
    
    height, width = size[0], size[1]
    
    images = [resize(image=image, size=(height, width), resample=resample) for image in images]
    # Convert each image to a numpy array
    images = [np.array(image) for image in images]
    # Rescale the pixel values to be in the range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Normalize the images to have mean 0 and standard deviation 1
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # Make channel dimension the first dimension as the model expects images in the format [Channel, Height, Width]
    images = [image.transpose(2, 0, 1) for image in images]
    
    return images


class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(
        self, 
        tokenizer: AutoTokenizer, 
        num_image_tokens: int, 
        image_size:int
    ):
        super().__init__()

        self.num_image_tokens = num_image_tokens
        self.image_size = image_size
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
    ) -> dict:
        
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        # Convert the list of numpy arrays to a single numpy array with shape [Batch_Size, Channels, Height, Width]
        pixel_values = np.stack(pixel_values, axis=0)
        
        # Convert the numpy array to a PyTorch tensor
        pixel_values = torch.tensor(pixel_values)

        # Prepend image tokens to the prompt
        input_strings = [
            prepend_image_tokens_to_prompt(
                prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                num_image_tokens=self.num_image_tokens,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Get the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
        )

        return_data = {
            "pixel_values": pixel_values, 
            **inputs
        }

        return return_data