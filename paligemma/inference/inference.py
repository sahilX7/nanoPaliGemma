# Imports
from PIL import Image
import torch
from paligemma.processor.paligemma_processor import PaliGemmaProcessor
from paligemma.models.gemma import KVCache
from paligemma.core.paligemma import PaliGemmaForConditionalGeneration
from paligemma.utils.utils import load_hf_model

# Move inputs to CPU/GPU
def move_inputs_to_device(
    model_inputs: dict, 
    device: str
):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs

# Prepare inputs for model (image + prompt)
def get_model_inputs(
    processor: PaliGemmaProcessor, 
    prompt: str, 
    image_path: str, 
    device: str
):
    image = Image.open(image_path)
    images = [image] # Wrap in a list as processor expects batch
    prompts = [prompt] # Wrap in a list as processor expects batch
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs

# Token-by-token generation
def inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_path: str,
    max_tokens_to_generate: int,
):
    # Get model inputs
    model_inputs = get_model_inputs(processor, prompt, image_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    # KV cache
    kv_cache = KVCache()

    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    # Generate tokens
    for _ in range(max_tokens_to_generate):
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :] # tensor([[0.70, ....., 5.20]]) Note: outputs["logits"] is a 3-dimensional tensor
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True) # tensor([[30871]])
        next_token = next_token.squeeze(0)  # tensor([30871]) - Remove batch dimension 
        generated_tokens.append(next_token)

        # Stop if the stop token has been generated
        if next_token.item() == stop_token:
            break

        # Append the next token to the input
        input_ids = next_token.unsqueeze(-1) # tensor([[30871]])

        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    generated_tokens = torch.cat(generated_tokens, dim=-1) # tensor([30871, 1]) where 1 = stop token id

    # Decode the generated tokens
    decoded_tokens = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return decoded_tokens

# Load model + run inference
def ask_paligemma(
    model_path: str,
    prompt: str,
    image_path: str,
    max_tokens_to_generate: int = 50,
    only_cpu: bool = False,
):
    device = "cpu"
    if not only_cpu and torch.cuda.is_available():
        device = "cuda"

    print("\nDevice in use:", device)

    print(f"Loading model...")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval() # Switch to inference mode

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running inference...")
    with torch.no_grad():
        return inference(
            model,
            processor,
            device,
            prompt,
            image_path,
            max_tokens_to_generate
        )