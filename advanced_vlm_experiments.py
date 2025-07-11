import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPProcessor, CLIPModel, GPT2Tokenizer, GPT2LMHeadModel,
    BlipProcessor, BlipForConditionalGeneration,
    AutoProcessor, AutoModelForVision2Seq
)
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import requests
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class AdvancedVLMExperiments:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def experiment_attention_visualization(self, image_path):
        """Visualize attention patterns in CLIP"""
        print("\n=== Experiment: Attention Visualization ===")
        
        # Load CLIP model
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Process image
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        
        # Get attention weights from the last layer
        with torch.no_grad():
            outputs = model.vision_model(**inputs, output_attentions=True)
            
        # Extract attention from the last layer
        attention = outputs.attentions[-1]  # Shape: (batch, heads, tokens, tokens)
        
        # Average across heads and get attention to [CLS] token
        cls_attention = attention[0].mean(0)[0, 1:]  # Skip [CLS] token itself
        
        # Reshape to spatial dimensions (14x14 for ViT-B/32)
        grid_size = int(np.sqrt(len(cls_attention)))
        attention_map = cls_attention.reshape(grid_size, grid_size)
        
        # Visualize
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(attention_map, cmap='hot', interpolation='nearest')
        plt.title("Attention Map")
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        # Overlay attention on image
        resized_attention = np.array(Image.fromarray(attention_map.numpy()).resize(image.size))
        plt.imshow(image)
        plt.imshow(resized_attention, alpha=0.6, cmap='hot')
        plt.title("Attention Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'attention_viz_{image_path.split(".")[0]}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Attention visualization saved as attention_viz_{image_path.split('.')[0]}.png")
        
    def experiment_multimodal_retrieval(self, image_paths, text_queries):
        """Test multimodal retrieval using CLIP embeddings"""
        print("\n=== Experiment: Multimodal Retrieval ===")
        
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Encode images
        image_embeddings = []
        images = []
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            images.append(image)
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                img_emb = model.get_image_features(**inputs)
                image_embeddings.append(img_emb.cpu().numpy())
        
        image_embeddings = np.vstack(image_embeddings)
        
        # Encode text queries
        text_embeddings = []
        for query in text_queries:
            inputs = processor(text=query, return_tensors="pt", padding=True)
            with torch.no_grad():
                txt_emb = model.get_text_features(**inputs)
                text_embeddings.append(txt_emb.cpu().numpy())
        
        text_embeddings = np.vstack(text_embeddings)
        
        # Compute similarities
        similarities = cosine_similarity(text_embeddings, image_embeddings)
        
        print("Retrieval Results:")
        for i, query in enumerate(text_queries):
            print(f"\nQuery: '{query}'")
            scores = similarities[i]
            ranked_images = np.argsort(scores)[::-1]
            
            for j, img_idx in enumerate(ranked_images):
                print(f"  {j+1}. {image_paths[img_idx]} (score: {scores[img_idx]:.3f})")
        
        return similarities
    
    def experiment_trainable_projection(self, image_path):
        """Experiment with a trainable projection layer"""
        print("\n=== Experiment: Trainable Projection Layer ===")
        
        class TrainableVLM(nn.Module):
            def __init__(self, clip_dim=512, gpt_dim=768, hidden_dim=256):
                super().__init__()
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
                self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                
                if self.gpt_tokenizer.pad_token is None:
                    self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
                
                # Trainable projection with multiple layers
                self.projection = nn.Sequential(
                    nn.Linear(clip_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, gpt_dim),
                    nn.LayerNorm(gpt_dim)
                )
                
                # Freeze pre-trained models
                for param in self.clip_model.parameters():
                    param.requires_grad = False
                for param in self.gpt_model.parameters():
                    param.requires_grad = False
            
            def forward(self, images, text_prompts):
                # Get image features
                clip_inputs = self.clip_processor(images=images, return_tensors="pt")
                img_features = self.clip_model.get_image_features(**clip_inputs)
                
                # Project to GPT space
                projected_features = self.projection(img_features)
                
                # Get text embeddings
                text_inputs = self.gpt_tokenizer(text_prompts, return_tensors="pt", padding=True)
                text_embeds = self.gpt_model.transformer.wte(text_inputs['input_ids'])
                
                # Combine
                combined_embeds = torch.cat([
                    projected_features.unsqueeze(1), 
                    text_embeds
                ], dim=1)
                
                return combined_embeds
        
        # Create model
        model = TrainableVLM()
        
        # Dummy training setup (would need real data for actual training)
        image = Image.open(image_path).convert("RGB")
        prompt = "This image shows"
        
        combined_embeds = model([image], [prompt])
        
        print(f"Combined embeddings shape: {combined_embeds.shape}")
        print("This demonstrates a more sophisticated projection layer that could be trained.")
        print("In practice, you would train this on image-caption pairs.")
        
        return model
    
    def experiment_prompt_engineering(self, image_path):
        """Test advanced prompt engineering techniques"""
        print("\n=== Experiment: Advanced Prompt Engineering ===")
        
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        image = Image.open(image_path).convert("RGB")
        
        # Different prompting strategies
        strategies = {
            "unconditional": "",
            "conditional_basic": "a photo of",
            "conditional_detailed": "a detailed photo of",
            "question_based": "what is in this image?",
            "style_guided": "a professional photograph showing",
            "context_guided": "in this scene, we can see",
            "technical": "this image contains",
            "creative": "this beautiful image captures"
        }
        
        results = {}
        
        for strategy_name, prompt in strategies.items():
            print(f"\nTesting strategy: {strategy_name}")
            print(f"Prompt: '{prompt}'")
            
            if prompt:
                inputs = processor(image, prompt, return_tensors="pt")
            else:
                inputs = processor(image, return_tensors="pt")
            
            with torch.no_grad():
                out = model.generate(**inputs, max_length=50, num_beams=5)
            
            caption = processor.decode(out[0], skip_special_tokens=True)
            results[strategy_name] = caption
            print(f"Result: {caption}")
        
        return results
    
    def experiment_image_text_matching(self, image_paths, captions):
        """Test image-text matching capabilities"""
        print("\n=== Experiment: Image-Text Matching ===")
        
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Create all combinations
        results = {}
        
        for i, img_path in enumerate(image_paths):
            image = Image.open(img_path).convert("RGB")
            results[img_path] = {}
            
            for j, caption in enumerate(captions):
                inputs = processor(text=caption, images=image, return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                
                results[img_path][caption] = probs[0][0].item()
        
        # Display results as a heatmap
        image_names = [path.split('.')[0] for path in image_paths]
        scores_matrix = np.array([[results[img][cap] for cap in captions] for img in image_paths])
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(scores_matrix, 
                   xticklabels=captions,
                   yticklabels=image_names,
                   annot=True, 
                   fmt='.3f',
                   cmap='viridis')
        plt.title("Image-Text Matching Scores")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('image_text_matching.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return results
    
    def experiment_feature_analysis(self, image_path):
        """Analyze different layers of CLIP features"""
        print("\n=== Experiment: Feature Analysis ===")
        
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        
        # Get features from different layers
        with torch.no_grad():
            outputs = model.vision_model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states
        
        print(f"Number of layers: {len(hidden_states)}")
        
        # Analyze each layer
        layer_stats = {}
        for i, layer_output in enumerate(hidden_states):
            # Take the [CLS] token representation
            cls_features = layer_output[0, 0, :]  # [CLS] token
            
            layer_stats[f"layer_{i}"] = {
                "mean": cls_features.mean().item(),
                "std": cls_features.std().item(),
                "max": cls_features.max().item(),
                "min": cls_features.min().item(),
                "norm": cls_features.norm().item()
            }
        
        # Plot feature evolution across layers
        layers = list(range(len(hidden_states)))
        means = [layer_stats[f"layer_{i}"]["mean"] for i in layers]
        stds = [layer_stats[f"layer_{i}"]["std"] for i in layers]
        norms = [layer_stats[f"layer_{i}"]["norm"] for i in layers]
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(layers, means, 'b-o')
        plt.title("Feature Means Across Layers")
        plt.xlabel("Layer")
        plt.ylabel("Mean Activation")
        
        plt.subplot(1, 3, 2)
        plt.plot(layers, stds, 'r-o')
        plt.title("Feature Std Across Layers")
        plt.xlabel("Layer")
        plt.ylabel("Std Activation")
        
        plt.subplot(1, 3, 3)
        plt.plot(layers, norms, 'g-o')
        plt.title("Feature Norms Across Layers")
        plt.xlabel("Layer")
        plt.ylabel("L2 Norm")
        
        plt.tight_layout()
        plt.savefig(f'feature_analysis_{image_path.split(".")[0]}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return layer_stats

if __name__ == "__main__":
    experiments = AdvancedVLMExperiments()
    
    # Available images
    image_paths = ["download.jpeg", "images (1).jpeg"]
    
    # Test captions for matching experiment
    test_captions = [
        "a robot with weapons",
        "a car driving on the road", 
        "a person walking",
        "a mechanical device",
        "futuristic technology"
    ]
    
    # Text queries for retrieval
    text_queries = [
        "robot or mechanical device",
        "vehicle or transportation",
        "person or human figure",
        "outdoor scene"
    ]
    
    print("Running Advanced VLM Experiments...")
    print("="*60)
    
    try:
        # Experiment 1: Attention visualization
        experiments.experiment_attention_visualization(image_paths[0])
        
        # Experiment 2: Multimodal retrieval
        experiments.experiment_multimodal_retrieval(image_paths, text_queries)
        
        # Experiment 3: Trainable projection
        experiments.experiment_trainable_projection(image_paths[0])
        
        # Experiment 4: Prompt engineering
        experiments.experiment_prompt_engineering(image_paths[0])
        
        # Experiment 5: Image-text matching
        experiments.experiment_image_text_matching(image_paths, test_captions)
        
        # Experiment 6: Feature analysis
        experiments.experiment_feature_analysis(image_paths[0])
        
    except Exception as e:
        print(f"Error in experiments: {e}")
        import traceback
        traceback.print_exc() 