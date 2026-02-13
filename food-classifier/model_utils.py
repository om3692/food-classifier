import io
import torch
import random
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

class FoodFreshnessClassifier:
    def __init__(self):
        print("Loading Vision Transformer Model... this may take a moment.")
        
        # We use a standard pre-trained ViT model (google/vit-base-patch16-224)
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        
        # --- MEMORY FIX APPLIED HERE ---
        # We added 'low_cpu_mem_usage=True' to prevent the OSError (Page File too small)
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            low_cpu_mem_usage=True
        )
        
        self.model.eval() # Set to evaluation mode

    def transform_image(self, image_bytes):
        """Converts raw bytes to a PIL Image."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image

    def predict(self, image_bytes):
        """
        1. Identifies the object using the Real AI Model.
        2. Simulates a freshness score (Prototype Logic).
        """
        image = self.transform_image(image_bytes)
        
        # 1. Real AI Inference (Object Detection)
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the predicted class (e.g., "Granny Smith Apple")
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        detected_food = self.model.config.id2label[predicted_class_idx]

        # 2. Freshness Logic (Simulated for Prototype)
        # PRO TIP: To make this real, you would retrain the model on a dataset of 
        # rotten vs fresh fruits and replace the logic below.
        
        freshness_score = round(random.uniform(0.70, 0.99), 2)
        
        if freshness_score > 0.85:
            label = "Fresh"
            color_code = "#28a745" # Green
        elif freshness_score > 0.50:
            label = "Okay (Consume Soon)"
            color_code = "#ffc107" # Orange
        else:
            label = "Avoid / Rotten"
            color_code = "#dc3545" # Red

        return {
            "detected_item": detected_food,
            "freshness_label": label,
            "confidence": f"{int(freshness_score * 100)}%",
            "color": color_code
        }

# Initialize the model once when this module is imported
classifier = FoodFreshnessClassifier()