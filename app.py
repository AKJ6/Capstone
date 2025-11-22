from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
from typing import Optional, List

# Import functions from the recipe generation script
from llama_recipe_personalized import (
    build_system_prompt,
    build_user_prompt,
    build_personalization_prompt,
    build_health_goals_personalization_prompt,
    call_ollama_chat,
    _load_corpus,
    _retrieve_context,
    OLLAMA_HOST_DEFAULT,
    DEFAULT_MODEL,
)

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

# Configuration
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", OLLAMA_HOST_DEFAULT)
DEFAULT_MODEL_NAME = os.environ.get("LLAMA_MODEL", "llama3.2:1b")


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')


@app.route('/api/generate-recipe', methods=['POST'])
def generate_recipe():
    """Generate a recipe based on user inputs"""
    try:
        if not request.json:
            return jsonify({
                'success': False,
                'error': 'No JSON data received'
            }), 400
        
        data = request.json
        print(f"Received request: {data}")
        
        # Extract parameters
        ingredients_str = data.get('ingredients', '') or ''
        ingredients = [s.strip() for s in ingredients_str.split(',') if s.strip()] if ingredients_str else []
        recipe_name = data.get('recipe_name', '').strip() if data.get('recipe_name') else ''
        cuisine = data.get('cuisine', '').strip() or None
        diet = data.get('diet', '').strip() or None
        servings = data.get('servings')
        max_calories = data.get('max_calories')
        equipment_str = data.get('equipment', '') or ''
        equipment = [s.strip() for s in equipment_str.split(',') if s.strip()] if equipment_str else None
        health_goals = data.get('health_goals', '').strip() or None
        corpus_path = data.get('corpus_path', '').strip() if data.get('corpus_path') else ''
        top_k = data.get('top_k', 3)
        max_context_chars = data.get('max_context_chars', 1200)
        model = data.get('model', DEFAULT_MODEL_NAME)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.95)
        seed = data.get('seed')
        
        print(f"Parsed ingredients: {ingredients}, recipe_name: {recipe_name}")
        
        # Validate inputs
        if not ingredients and not recipe_name:
            return jsonify({
                'success': False,
                'error': 'Either ingredients or recipe_name must be provided'
            }), 400
        
        # RAG retrieval if corpus is provided
        rag_context = None
        rag_sources = []
        if corpus_path and os.path.exists(corpus_path):
            try:
                documents = _load_corpus(corpus_path)
                query_pieces = [
                    "ingredients: " + ", ".join(ingredients) if ingredients else "",
                    f"diet: {diet}" if diet else "",
                    f"cuisine: {cuisine}" if cuisine else "",
                    f"health_goals: {health_goals}" if health_goals else "",
                    f"recipe_name: {recipe_name}" if recipe_name else "",
                ]
                query_text = " | ".join([p for p in query_pieces if p]) or "general cooking recipe"
                rag_context, rag_sources = _retrieve_context(
                    query_text=query_text,
                    documents=documents,
                    top_k=max(1, int(top_k)),
                    max_context_chars=max(300, int(max_context_chars)),
                )
            except Exception as e:
                return jsonify({
                    'error': f'RAG retrieval failed: {str(e)}'
                }), 500
        
        # Build prompts
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(
            ingredients=ingredients,
            diet=diet,
            cuisine=cuisine,
            servings=servings,
            max_calories=max_calories,
            extra_tools=equipment,
            health_goals=health_goals,
            recipe_name=recipe_name if recipe_name else None,
            retrieved_context=rag_context,
        )
        
        # Generate recipe
        print(f"Calling Ollama with model: {model}, host: {OLLAMA_HOST}")
        recipe_content = call_ollama_chat(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            host=OLLAMA_HOST,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        
        if not recipe_content or not recipe_content.strip():
            return jsonify({
                'success': False,
                'error': 'Received empty recipe from model. Please try again or check if Ollama is running correctly.'
            }), 500
        
        print(f"Recipe generated successfully, length: {len(recipe_content)}")
        
        return jsonify({
            'success': True,
            'recipe': recipe_content,
            'rag_sources': [{'meta': meta, 'score': float(score)} for meta, score in rag_sources],
            'model': model,
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in generate_recipe: {str(e)}")
        print(f"Traceback: {error_trace}")
        return jsonify({
            'success': False,
            'error': f'Recipe generation failed: {str(e)}'
        }), 500


@app.route('/api/personalize-dietary', methods=['POST'])
def personalize_dietary():
    """Personalize a recipe based on dietary preference"""
    try:
        data = request.json
        
        original_recipe = data.get('original_recipe', '')
        dietary_preference = data.get('dietary_preference', '').strip()
        ingredients = [s.strip() for s in data.get('ingredients', '').split(',') if s.strip()]
        cuisine = data.get('cuisine', '').strip() or None
        servings = data.get('servings')
        max_calories = data.get('max_calories')
        model = data.get('model', DEFAULT_MODEL_NAME)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.95)
        seed = data.get('seed')
        
        if not original_recipe or not dietary_preference:
            return jsonify({
                'error': 'original_recipe and dietary_preference are required'
            }), 400
        
        personalization_prompt = build_personalization_prompt(
            original_recipe=original_recipe,
            dietary_preference=dietary_preference,
            ingredients=ingredients,
            cuisine=cuisine,
            servings=servings,
            max_calories=max_calories,
        )
        
        personalized_recipe = call_ollama_chat(
            model=model,
            system_prompt=build_system_prompt(),
            user_prompt=personalization_prompt,
            host=OLLAMA_HOST,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        
        return jsonify({
            'success': True,
            'recipe': personalized_recipe,
            'dietary_preference': dietary_preference,
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Personalization failed: {str(e)}'
        }), 500


@app.route('/api/personalize-health', methods=['POST'])
def personalize_health():
    """Personalize a recipe based on health goals"""
    try:
        data = request.json
        
        original_recipe = data.get('original_recipe', '')
        health_goals = data.get('health_goals', '').strip()
        ingredients = [s.strip() for s in data.get('ingredients', '').split(',') if s.strip()]
        cuisine = data.get('cuisine', '').strip() or None
        servings = data.get('servings')
        max_calories = data.get('max_calories')
        model = data.get('model', DEFAULT_MODEL_NAME)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.95)
        seed = data.get('seed')
        
        if not original_recipe or not health_goals:
            return jsonify({
                'error': 'original_recipe and health_goals are required'
            }), 400
        
        health_goals_prompt = build_health_goals_personalization_prompt(
            original_recipe=original_recipe,
            health_goals=health_goals,
            ingredients=ingredients,
            cuisine=cuisine,
            servings=servings,
            max_calories=max_calories,
        )
        
        personalized_recipe = call_ollama_chat(
            model=model,
            system_prompt=build_system_prompt(),
            user_prompt=health_goals_prompt,
            host=OLLAMA_HOST,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        
        return jsonify({
            'success': True,
            'recipe': personalized_recipe,
            'health_goals': health_goals,
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Health personalization failed: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'service': 'recipe-generator-api'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='127.0.0.1', port=port, debug=True)

