import torch
from transformers import CLIPProcessor, CLIPModel, GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image
import requests

def generate_caption(image_path, prompt="a picture of"):
    print(f"Loading image from: {image_path}")
    
    # Load pretrained models
    clip_model_name = "openai/clip-vit-base-patch32"
    llm_model_name = "gpt2"
    
    print("Loading CLIP processor...")
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(llm_model_name)
    print("Loading GPT-2 model...")
    llm = GPT2LMHeadModel.from_pretrained(llm_model_name)

    # Add padding token for GPT-2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading and processing image...")
    # Load and process the image
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=None, images=image, return_tensors="pt", padding=True)
    
    print("Extracting image features...")
    # Get image features from CLIP
    image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
    
    print("Setting up projection layer...")
    # Prepare inputs for GPT-2
    # We need to project the image features to the same dimension as the text embeddings
    # This is a simplified approach, a real VLM would have a projection layer.
    # For this experiment, we will use a simple linear layer.
    
    llm_embedding_size = llm.config.hidden_size
    clip_embedding_size = image_features.shape[1]
    
    print(f"CLIP embedding size: {clip_embedding_size}, LLM embedding size: {llm_embedding_size}")
    
    projection = torch.nn.Linear(clip_embedding_size, llm_embedding_size)
    image_embeds = projection(image_features)
    
    print("Processing text prompt...")
    # Create a dummy text input
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    text_embeds = llm.transformer.wte(input_ids)
    
    print("Combining embeddings...")
    # Concatenate image and text embeddings
    inputs_embeds = torch.cat((image_embeds.unsqueeze(1), text_embeds), dim=1)
    
    print("Generating caption...")
    # Generate text
    with torch.no_grad():
        output = llm.generate(inputs_embeds=inputs_embeds, max_length=50, do_sample=True, temperature=0.7)
    
    print("Decoding result...")
    # Decode and print the caption
    caption = tokenizer.decode(output[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    image_file = "download.jpeg"
    try:
        caption = generate_caption(image_file)
        print(f"Generated Caption: {caption}")
    except Exception as e:
        print(f"Error generating caption: {e}")
        import traceback
        traceback.print_exc() 