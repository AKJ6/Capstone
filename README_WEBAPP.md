# Recipe Generator Web App - Quick Start Guide

##  Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure Ollama is Running

Make sure Ollama is installed and running on your system:
- Download from: https://ollama.ai
- Start Ollama service
- Pull a model: `ollama pull llama3.2:1b`

### 3. Start the Flask Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### 4. Open the Web App

Open your browser and navigate to:
```
http://localhost:5000
```

##  Features

-  **Recipe Generation** - Generate recipes from ingredients or recipe names
-  **RAG Support** - Use your recipe corpus for enhanced context-aware generation
-  **Dietary Personalization** - Adapt recipes for vegan, vegetarian, gluten-free, etc.
-  **Health Goals** - Optimize recipes for weight loss, muscle gain, heart health, etc.
-  **Beautiful UI** - Modern, animated interface with smooth transitions
-  **Real-time Updates** - See recipe generation progress and results instantly

##  Usage

1. **Fill in the form** with your ingredients, cuisine preference, servings, etc.
2. **Optional**: Add a corpus path for RAG-enhanced generation
3. **Click "Generate Recipe"** and wait for the AI to create your recipe
4. **Personalize** the recipe using the dietary or health goals buttons
5. **View results** in the formatted recipe display

##  Configuration

### Environment Variables

- `OLLAMA_HOST` - Ollama server URL (default: `http://localhost:11434`)
- `LLAMA_MODEL` - Default model name (default: `llama3.2:1b`)
- `PORT` - Flask server port (default: `5000`)

### Example

```bash
export OLLAMA_HOST="http://localhost:11434"
export LLAMA_MODEL="llama3.2:1b"
python app.py
```

## Project Structure

```
.
├── app.py                      # Flask backend API
├── llama_recipe_personalized.py # Core recipe generation logic
├── requirements.txt            # Python dependencies
├── static/
│   ├── index.html             # Frontend HTML
│   ├── styles.css             # Styling and animations
│   └── script.js              # Frontend JavaScript
└── README_WEBAPP.md           # This file
```

##  Troubleshooting

### Cannot connect to API
- Ensure Flask server is running: `python app.py`
- Check if port 5000 is available
- Verify CORS is enabled (should be automatic)

### Ollama connection errors
- Verify Ollama is running: `ollama list`
- Check `OLLAMA_HOST` environment variable
- Ensure the model is pulled: `ollama pull llama3.2:1b`

### RAG not working
- Verify corpus file path is correct
- Check file format (CSV or TXT)
- Ensure file is readable

##  Example Workflow

1. Enter ingredients: `chicken, tomato, onion, spices`
2. Select cuisine: `Indian`
3. Set servings: `4`
4. Add corpus path: `./capstone 71/Cleaned_Indian_Food_Dataset.csv`
5. Click "Generate Recipe"
6. Wait for recipe generation
7. Click "Personalize for Dietary Preference" → Enter `vegetarian`
8. View personalized recipe

##  Notes

- The web app uses the same core logic as `llama_recipe_personalized.py`
- All features from the CLI version are available in the web interface
- Recipes are displayed in real-time as they're generated
- Personalization can be applied multiple times to the same recipe









