from flask import Flask, render_template, request, jsonify
from PIL import Image
import google.generativeai as genai
import io
import base64

app = Flask(__name__)

genai.configure(api_key="Put your API")
model = genai.GenerativeModel("gemini-2.5-flash")

RECIPE = """
Tomato & Onion Omelette Recipe

Ingredients (1-2 servings):
2 large eggs
1 small tomato, finely chopped
1 small onion, finely chopped
1 green chili (optional), finely chopped
1-2 tbsp milk or water
1-2 tbsp oil or butter
Salt, to taste
Black pepper, to taste
Fresh herbs (optional: coriander, parsley, or chives)

Instructions:
Prep vegetables
Finely chop the tomato and onion.
If using chili, chop it finely.
Beat the eggs
Crack the eggs into a bowl.
Add milk or water, salt, and black pepper.
Beat well until fluffy.
Cook the vegetables
Heat oil or butter in a non-stick pan over medium heat.
Add onions and sauté for 1-2 minutes until slightly translucent.
Add tomatoes (and chili, if using) and sauté for another 1-2 minutes.
Add the eggs
Pour the beaten eggs evenly over the cooked vegetables.
Tilt the pan to spread the eggs evenly.
Cook the omelette
Let it cook on medium heat for 2-3 minutes.
Gently lift the edges with a spatula to allow uncooked egg to flow underneath.
Optional: Cover the pan with a lid to cook the top faster.
Finish & serve
Once the omelette is mostly set but still soft on top, fold it in half.
Slide onto a plate and garnish with fresh herbs if desired.
Serve hot with bread or toast.
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get("question")
    image_data = data.get("image")

    # Decode image from base64
    img_bytes = base64.b64decode(image_data.split(",")[1])
    img = Image.open(io.BytesIO(img_bytes))

    # Send image + question to Gemini
    response = model.generate_content([RECIPE, img, question])

    return jsonify({"answer": response.text})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
