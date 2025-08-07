from flask import Flask, request, render_template
from PIL import Image
import os
import torch
from torchvision import transforms
from transformers import AutoProcessor, AutoModelForImageClassification
import requests

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Load Food101 model
processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

# NutritionDB for basic info (example)
nutrition_data = {
    "pizza": {"calories": 266, "protein": 11, "carbs": 33, "fat": 10},
    "apple": {"calories": 52, "protein": 0.3, "carbs": 14, "fat": 0.2},
    "burger": {"calories": 295, "protein": 17, "carbs": 30, "fat": 13},
    "salad": {"calories": 152, "protein": 2.9, "carbs": 11, "fat": 11},
    "ice cream": {"calories": 207, "protein": 3.5, "carbs": 24, "fat": 11},
}

def get_nutrition(food_name):
    return nutrition_data.get(food_name.lower())

def suggest_intake(height, weight, cholesterol):
    bmi = weight / ((height / 100) ** 2)
    base_cal = 25 * weight
    if cholesterol == "high":
        fat = 50
    else:
        fat = 70
    return {
        "calories": int(base_cal),
        "protein": int(weight * 1.2),
        "carbs": int(base_cal * 0.5 / 4),
        "fat": fat,
    }

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    suggestion = None
    chart_data = None

    if request.method == "POST":
        image = request.files["image"]
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        cholesterol = request.form["cholesterol"]

        if image:
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
            image.save(img_path)

            # Process image
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_class = outputs.logits.argmax(-1).item()
                food_name = model.config.id2label[predicted_class]

            nutrition = get_nutrition(food_name)

            if nutrition:
                chart_data = {
                    "protein": nutrition["protein"],
                    "carbs": nutrition["carbs"],
                    "fat": nutrition["fat"],
                }

            suggestion = suggest_intake(height, weight, cholesterol)

            result = {
                "food_name": food_name,
                "nutrition": nutrition,
                "image_url": img_path,
            }

    return "<h1>Hello</h1>"

if __name__ == "__main__":
    app.run(debug=True)

