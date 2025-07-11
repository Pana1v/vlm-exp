import torch
from transformers import (
    CLIPProcessor, CLIPModel, GPT2Tokenizer, GPT2LMHeadModel,
    BlipProcessor, BlipForConditionalGeneration,
    AutoProcessor, AutoModelForCausalLM
)
from PIL import Image
import requests
import time

class VLMExperiment:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def experiment_1_basic_clip_gpt2(self, image_path, prompt="Describe this image:"):
        """Basic CLIP + GPT-2 with linear projection"""
        print("\n=== Experiment 1: Basic CLIP + GPT-2 ===")
        
        # Load models
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        llm = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Process image
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(self.device)
        image_features = clip_model.get_image_features(**inputs)
        
        # Project to GPT-2 space
        projection = torch.nn.Linear(image_features.shape[1], llm.config.hidden_size).to(self.device)
        image_embeds = projection(image_features)
        
        # Create text embeddings
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        text_embeds = llm.transformer.wte(input_ids)
        
        # Combine embeddings
        inputs_embeds = torch.cat((image_embeds.unsqueeze(1), text_embeds), dim=1)
        
        # Generate
        with torch.no_grad():
            output = llm.generate(
                inputs_embeds=inputs_embeds, 
                max_length=50, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        caption = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Result: {caption}")
        return caption
    
    def experiment_2_blip(self, image_path):
        """Use BLIP model (state-of-the-art image captioning)"""
        print("\n=== Experiment 2: BLIP Model ===")
        
        try:
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
            
            image = Image.open(image_path).convert("RGB")
            inputs = processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = model.generate(**inputs, max_length=50)
            
            caption = processor.decode(out[0], skip_special_tokens=True)
            print(f"Result: {caption}")
            return caption
        except Exception as e:
            print(f"BLIP experiment failed: {e}")
            return None
    
    def experiment_3_clip_with_templates(self, image_path):
        """Use CLIP for zero-shot classification with templates"""
        print("\n=== Experiment 3: CLIP Zero-shot Classification ===")
        
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        
        # Define templates and possible descriptions
        templates = [
            "a photo of a {}",
            "a picture of a {}",
            "an image of a {}",
            "a {} in the image",
            "this is a {}"
        ]
        
        candidates = [
            "dog", "cat", "car", "person", "building", "tree", "flower", "food",
            "landscape", "animal", "vehicle", "house", "street", "nature", "sky",
            "water", "mountain", "city", "beach", "forest"
        ]
        
        image = Image.open(image_path).convert("RGB")
        
        # Create all text combinations
        texts = []
        for template in templates:
            for candidate in candidates:
                texts.append(template.format(candidate))
        
        # Process
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probs[0], 5)
        
        print("Top predictions:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            print(f"{i+1}. {texts[idx]} (confidence: {prob:.3f})")
        
        return texts[top_indices[0]]
    
    def experiment_4_different_clip_models(self, image_path):
        """Compare different CLIP model sizes"""
        print("\n=== Experiment 4: Different CLIP Models ===")
        
        models = [
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14"
        ]
        
        results = {}
        image = Image.open(image_path).convert("RGB")
        
        for model_name in models:
            try:
                print(f"\nTesting {model_name}...")
                processor = CLIPProcessor.from_pretrained(model_name)
                model = CLIPModel.from_pretrained(model_name).to(self.device)
                
                # Simple classification task
                texts = ["a photo of a dog", "a photo of a cat", "a photo of a car", "a photo of food"]
                inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(self.device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)
                
                best_idx = probs.argmax().item()
                results[model_name] = {
                    "prediction": texts[best_idx],
                    "confidence": probs[0][best_idx].item()
                }
                print(f"Prediction: {texts[best_idx]} (confidence: {probs[0][best_idx]:.3f})")
                
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def experiment_5_different_prompts(self, image_path):
        """Test different prompting strategies"""
        print("\n=== Experiment 5: Different Prompting Strategies ===")
        
        prompts = [
            "Describe this image:",
            "What do you see in this picture?",
            "This image shows",
            "In this photo, there is",
            "The main subject of this image is",
            "Caption:",
            "",  # No prompt
        ]
        
        results = {}
        
        for prompt in prompts:
            try:
                print(f"\nTesting prompt: '{prompt}'")
                result = self.experiment_1_basic_clip_gpt2(image_path, prompt)
                results[prompt] = result
            except Exception as e:
                print(f"Failed with prompt '{prompt}': {e}")
                results[prompt] = f"Error: {e}"
        
        return results
    
    def run_all_experiments(self, image_path):
        """Run all experiments on the given image"""
        print(f"Running all experiments on: {image_path}")
        print("=" * 60)
        
        results = {}
        
        # Experiment 1: Basic CLIP + GPT-2
        try:
            results["clip_gpt2"] = self.experiment_1_basic_clip_gpt2(image_path)
        except Exception as e:
            results["clip_gpt2"] = f"Error: {e}"
        
        # Experiment 2: BLIP
        try:
            results["blip"] = self.experiment_2_blip(image_path)
        except Exception as e:
            results["blip"] = f"Error: {e}"
        
        # Experiment 3: CLIP Classification
        try:
            results["clip_classification"] = self.experiment_3_clip_with_templates(image_path)
        except Exception as e:
            results["clip_classification"] = f"Error: {e}"
        
        # Experiment 4: Different CLIP models
        try:
            results["clip_models"] = self.experiment_4_different_clip_models(image_path)
        except Exception as e:
            results["clip_models"] = f"Error: {e}"
        
        # Experiment 5: Different prompts
        try:
            results["prompting"] = self.experiment_5_different_prompts(image_path)
        except Exception as e:
            results["prompting"] = f"Error: {e}"
        
        return results

if __name__ == "__main__":
    experiments = VLMExperiment()
    
    # Test on both available images
    images = ["images/download.jpeg", "images/images (1).jpeg"]
    
    for image in images:
        print(f"\n{'='*80}")
        print(f"TESTING IMAGE: {image}")
        print(f"{'='*80}")
        
        try:
            results = experiments.run_all_experiments(image)
            
            print(f"\n{'='*60}")
            print("SUMMARY OF RESULTS:")
            print(f"{'='*60}")
            
            for experiment, result in results.items():
                print(f"\n{experiment.upper()}:")
                if isinstance(result, dict):
                    for k, v in result.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"  {result}")
                    
        except Exception as e:
            print(f"Failed to process {image}: {e}")
            import traceback
            traceback.print_exc() 