# Llama Recipe Generator with RAG and Personalization

A powerful recipe generation tool that uses local Llama models via Ollama to create personalized recipes. Features include Retrieval-Augmented Generation (RAG) for context-aware recipe creation and multi-stage personalization based on dietary preferences and health goals.

## Features

- 🍳 **AI-Powered Recipe Generation**: Generate complete recipes using local Llama models via Ollama
- 🔍 **Retrieval-Augmented Generation (RAG)**: Enhance recipe quality by retrieving relevant context from your recipe corpus
- 🎯 **Dietary Personalization**: Adapt recipes to specific dietary preferences (vegan, vegetarian, gluten-free, keto, etc.)
- 💪 **Health Goals Optimization**: Personalize recipes for health goals (weight loss, muscle gain, heart health, etc.)
- 📊 **Comprehensive Recipe Output**: Includes ingredients, equipment, instructions, timing, nutrition, allergens, and variations
- 📁 **Multiple Output Formats**: Saves recipes and references to timestamped text files
- 🔧 **Flexible Input**: Support for ingredients, recipe names, cuisine styles, servings, and calorie limits

## Requirements

### Prerequisites

1. **Python 3.7+**
2. **Ollama** installed and running locally
   - Download from: https://ollama.ai
   - Install and ensure the service is running
   - Pull at least one Llama model (e.g., `ollama pull llama3.2:1b`)

### Python Dependencies

**Required:**
- `requests` - For API communication with Ollama

**Optional (for enhanced RAG):**
- `scikit-learn` - For TF-IDF-based semantic retrieval (falls back to Jaccard similarity if not available)

### Installation

1. Install required dependencies:
```bash
pip install requests
```

2. (Optional) Install scikit-learn for better RAG performance:
```bash
pip install scikit-learn
```

3. Ensure Ollama is running:
```bash
# Check if Ollama is running
ollama list

# If not running, start Ollama service
# On Windows: Start Ollama from Start Menu
# On Linux/Mac: ollama serve
```

4. Pull a Llama model (if not already done):
```bash
# For systems with limited memory, use a smaller model:
ollama pull llama3.2:1b

# For systems with more memory:
ollama pull llama3.1
# or
ollama pull llama3.2
```

## Usage

### Basic Recipe Generation

Generate a recipe from ingredients:
```bash
python llama_recipe_personalized.py --ingredients "tomato, onion, garlic" --cuisine "Indian" --servings 2
```

Generate a recipe by name:
```bash
python llama_recipe_personalized.py --recipe_name "Chicken Tikka Masala" --cuisine "Indian" --servings 4
```

### RAG-Based Recipe Generation

Use a recipe corpus for enhanced context-aware generation:
```bash
python llama_recipe_personalized.py \
  --ingredients "tomato, onion, garlic, paneer" \
  --cuisine "Indian" \
  --servings 2 \
  --corpus ".\capstone 71\Cleaned_Indian_Food_Dataset.csv" \
  --top_k 5 \
  --max_context_chars 1500 \
  --model llama3.2:1b
```

### With Personalization

Generate a recipe and personalize it interactively:
```bash
python llama_recipe_personalized.py \
  --ingredients "chicken, tomato, onion" \
  --cuisine "Indian" \
  --servings 2 \
  --corpus ".\capstone 71\Cleaned_Indian_Food_Dataset.csv"
```

The script will prompt you for:
1. **Dietary preference** (e.g., vegetarian, vegan, gluten-free)
2. **Health goals** (e.g., weight loss, heart health, high protein)

### Skip Personalization

Generate recipe without personalization prompts:
```bash
python llama_recipe_personalized.py \
  --ingredients "tomato, pasta, basil" \
  --cuisine "Italian" \
  --servings 2 \
  --skip_personalization
```

### Interactive Mode

Run without arguments to enter interactive mode:
```bash
python llama_recipe_personalized.py
```

The script will prompt you for all required information.

## Command-Line Arguments

### Recipe Input Options

| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--ingredients` | string | Comma-separated list of ingredients | `"tomato, onion, garlic"` |
| `--recipe_name` | string | Name of the dish to generate | `"Chicken Tikka Masala"` |
| `--cuisine` | string | Cuisine style | `"Indian"`, `"Italian"`, `"Mexican"` |
| `--diet` | string | Dietary preference | `"vegetarian"`, `"vegan"`, `"gluten-free"` |
| `--servings` | int | Target number of servings | `2`, `4`, `6` |
| `--max_calories` | int | Maximum calories per serving | `300`, `500` |
| `--equipment` | string | Available equipment/tools | `"oven, blender, skillet"` |
| `--health_goals` | string | Health goals to optimize for | `"weight loss, heart health"` |

### RAG Options

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--rag` | flag | Enable RAG (auto-enabled if `--corpus` provided) | `False` |
| `--corpus` | string | Path to corpus file (CSV or TXT) | `""` |
| `--top_k` | int | Number of retrieved snippets to include | `3` |
| `--max_context_chars` | int | Maximum total characters of retrieved context | `1200` |

### Model Configuration

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--model` | string | Ollama model name | `llama3.1` (or `LLAMA_MODEL` env var) |
| `--host` | string | Ollama server host | `http://localhost:11434` |
| `--temperature` | float | Sampling temperature (0.0-1.0) | `0.7` |
| `--top_p` | float | Nucleus sampling top_p | `0.95` |
| `--seed` | int | Random seed for deterministic output | `None` |

### Output Options

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--out` | string | Path to save generated recipe | Auto-generated timestamped file |
| `--skip_personalization` | flag | Skip personalization prompts | `False` |

## RAG (Retrieval-Augmented Generation)

### How It Works

1. **Corpus Loading**: The script loads your recipe corpus (CSV or TXT format)
2. **Query Construction**: Builds a query from your ingredients, cuisine, diet, and health goals
3. **Retrieval**: Uses TF-IDF (or Jaccard similarity) to find the most relevant recipe snippets
4. **Context Injection**: Injects retrieved context into the prompt to guide recipe generation
5. **Enhanced Output**: Generates recipes informed by your corpus

### Corpus Format

**CSV Format:**
The script automatically detects common CSV fields:
- `translatedRecipeName`, `name`, `title` - Recipe name
- `ingredients_list`, `ingredients`, `TranslatedIngredients`, `cleaned_ingredients` - Ingredients
- `translatedRecipeInstructions`, `instructions`, `method` - Instructions
- `cuisine`, `region`, `course` - Cuisine information

**TXT Format:**
Plain text files are split by blank lines into chunks.

### Example RAG Usage

```bash
python llama_recipe_personalized.py \
  --ingredients "chicken, tomato, onion, spices" \
  --cuisine "Indian" \
  --corpus "recipes.csv" \
  --top_k 5 \
  --max_context_chars 1500
```

## Personalization

### Dietary Preference Personalization

After generating the initial recipe, you can personalize it for specific dietary needs:

**Supported dietary preferences:**
- `vegan` - No animal products
- `vegetarian` - No meat, but may include dairy/eggs
- `gluten-free` - No gluten-containing ingredients
- `keto` - Low-carb, high-fat
- `paleo` - Paleolithic diet
- `halal` - Halal-compliant
- `kosher` - Kosher-compliant
- And more...

**Example:**
```bash
python llama_recipe_personalized.py --ingredients "chicken, tomato" --cuisine "Indian"
# When prompted, enter: vegetarian
```

### Health Goals Personalization

Optimize recipes for specific health objectives:

**Supported health goals:**
- `weight loss` - Reduced calories, increased fiber
- `muscle gain` - Increased protein content
- `heart health` - Reduced saturated fats, sodium; increased omega-3s
- `low sodium` - Reduced salt, herbs/spices for flavor
- `high protein` - Increased protein sources
- `low carb` / `keto` - Reduced carbohydrates
- `high fiber` - Added whole grains, legumes, vegetables
- `energy boost` - Complex carbs, B vitamins, iron-rich foods
- `anti-inflammatory` - Omega-3s, antioxidants, turmeric, ginger

**Example:**
```bash
python llama_recipe_personalized.py --ingredients "chicken, tomato" --cuisine "Indian"
# When prompted, enter: weight loss, heart health
```

## Output Files

The script generates multiple output files:

1. **Base Recipe**: `generated_recipe_llama_YYYYMMDD_HHMMSS.txt`
   - Contains the initial generated recipe

2. **References** (if RAG enabled): `generated_recipe_llama_YYYYMMDD_HHMMSS_refs.txt`
   - Lists retrieved recipe sources with similarity scores

3. **Dietary Personalized**: `generated_recipe_llama_YYYYMMDD_HHMMSS_dietary_personalized.txt`
   - Recipe adapted for dietary preferences (if personalization used)

4. **Health Goals Personalized**: `generated_recipe_llama_YYYYMMDD_HHMMSS_health_goals_personalized.txt`
   - Recipe optimized for health goals (if personalization used)

## Recipe Output Format

Each generated recipe includes:

- **Title** - Name of the dish
- **Description** - Brief overview
- **Ingredients** - Complete list with amounts
- **Equipment** - Required kitchen tools
- **Instructions** - Numbered step-by-step cooking instructions
- **Timing** - Preparation and cooking times
- **Serving Size** - Number of servings
- **Variations and Substitutions** - Alternative ingredients/methods
- **Allergens** - Common allergens present
- **Approximate Nutrition** - Per-serving nutritional information

## Environment Variables

- `LLAMA_MODEL` - Default model name (overrides `llama3.1`)
- `OLLAMA_HOST` - Default Ollama host (overrides `http://localhost:11434`)

## Troubleshooting

### Connection Errors

**Error**: `Could not connect to Ollama at http://localhost:11434`

**Solution**:
1. Ensure Ollama is installed and running
2. Check if Ollama service is active: `ollama list`
3. Verify the host with `--host` if using a remote server

### Model Memory Errors

**Error**: `model requires more system memory than is currently available`

**Solution**:
- Use a smaller model: `--model llama3.2:1b`
- Or: `--model llama3.2:3b`
- Check available models: `ollama list`

### RAG Issues

**Error**: `Corpus file not found`

**Solution**:
- Verify the corpus file path is correct
- Use absolute paths if relative paths fail
- Ensure the file exists and is readable

**Low-quality retrieval**:
- Increase `--top_k` to retrieve more snippets
- Increase `--max_context_chars` for more context
- Ensure your corpus contains relevant recipes

### Timeout Errors

**Error**: `Request to Llama (Ollama) timed out`

**Solution**:
- Use a smaller/faster model
- Reduce `--max_context_chars` if using RAG
- Check system resources (CPU, memory)

## Examples

### Example 1: Simple Recipe Generation
```bash
python llama_recipe_personalized.py \
  --ingredients "tomato, pasta, basil, olive oil" \
  --cuisine "Italian" \
  --servings 2 \
  --skip_personalization
```

### Example 2: RAG-Based Indian Recipe
```bash
python llama_recipe_personalized.py \
  --ingredients "chicken, tomato, onion, garlic, spices" \
  --cuisine "Indian" \
  --servings 4 \
  --corpus ".\capstone 71\Cleaned_Indian_Food_Dataset.csv" \
  --top_k 5 \
  --model llama3.2:1b
```

### Example 3: Health-Optimized Recipe
```bash
python llama_recipe_personalized.py \
  --recipe_name "Chicken Curry" \
  --cuisine "Indian" \
  --servings 2 \
  --max_calories 400 \
  --health_goals "weight loss, high protein" \
  --skip_personalization
```

### Example 4: Full Workflow with Personalization
```bash
python llama_recipe_personalized.py \
  --ingredients "paneer, tomato, onion, spices" \
  --cuisine "Indian" \
  --servings 2 \
  --corpus "recipes.csv" \
  --model llama3.2:1b
# Then interactively provide:
# - Dietary preference: vegetarian
# - Health goals: weight loss, heart health
```

## Advanced Usage

### Custom Model Configuration
```bash
python llama_recipe_personalized.py \
  --ingredients "chicken, tomato" \
  --model llama3.2 \
  --temperature 0.8 \
  --top_p 0.9 \
  --seed 42
```

### Remote Ollama Server
```bash
python llama_recipe_personalized.py \
  --ingredients "chicken, tomato" \
  --host "http://192.168.1.100:11434" \
  --model llama3.1
```

### Custom Output Path
```bash
python llama_recipe_personalized.py \
  --ingredients "chicken, tomato" \
  --out "my_recipe.txt" \
  --skip_personalization
```

## Limitations

- Requires Ollama to be running locally or accessible remotely
- Model quality depends on the Llama model used (larger models = better quality but more memory)
- RAG retrieval uses TF-IDF or Jaccard similarity (not semantic embeddings)
- Nutrition information is approximate and should be verified
- Recipe quality depends on corpus quality (for RAG mode)

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is provided as-is for educational and personal use.

## Acknowledgments

- Built using [Ollama](https://ollama.ai) for local LLM inference
- Optional RAG enhancement using scikit-learn
- Designed for recipe generation and personalization use cases

