from flask import Flask, render_template, request, jsonify
from PIL import Image
import google.generativeai as genai
import io
import base64

app = Flask(__name__)

# 🔑 Configure Gemini
genai.configure(api_key="AIzaSyBMyTwJZKn5ARhTNGxUx8wv3bXPZKa8KYc")
model = genai.GenerativeModel("gemini-2.5-flash")

# 📖 Example recipe
RECIPE = """
Pasta with Tomato Sauce:
1. Boil water and cook pasta for 8 minutes.
2. Heat olive oil and sauté garlic until golden.
3. Add chopped tomatoes and cook until soft.
4. Mix in cooked pasta and stir for 2 minutes.
5. Garnish with basil and serve.
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
