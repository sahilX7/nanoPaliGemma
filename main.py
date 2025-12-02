# Imports
from paligemma.inference.inference import ask_paligemma

# Input setup
MODEL_PATH = "weights"
IMAGE_PATH = "test_images/img1.jpg"
MAX_TOKENS_TO_GENERATE = 10
ONLY_CPU = False

# Inference execution
if __name__ == "__main__":
    user_prompt = input("Ask anything:\n")
    assistant_response = ask_paligemma(
        model_path=MODEL_PATH,
        prompt=user_prompt,
        image_path=IMAGE_PATH,
        max_tokens_to_generate=MAX_TOKENS_TO_GENERATE,
        only_cpu=ONLY_CPU
    )

    print("\nUSER:", user_prompt)
    print("ASSISTANT:", assistant_response)