import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import faiss
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
import re

class RecipeRAG:
    def __init__(self, 
                 dataset_path: str = "capstone 71/ouutput.csv",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 generation_model: str = None,  # Set to None by default
                 device: Optional[str] = None):
        """
        Initialize the Recipe RAG system
        
        Args:
            dataset_path: Path to the recipe dataset CSV
            embedding_model: Model to use for embeddings
            generation_model: Model to use for text generation
            device: Device to use (cuda or cpu)
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load dataset
        print("Loading dataset...")
        self.df = pd.read_csv(dataset_path)
        
        # Clean dataset
        self.df = self.clean_dataset(self.df)
        
        # Load embedding model
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(embedding_model)
        self.embedder.to(self.device)
        
        # Load generation model
        print("Loading generation model...")
        
        # Define model paths to try
        model_paths = [
            # Try the passed model path first
            generation_model,
            # Then try common local paths
            "capstone 71/gpt2-recipes",
            "capstone 71/capstone 71/gpt2-recipes",
            os.path.join(os.getcwd(), "capstone 71/gpt2-recipes"),
            os.path.join(os.getcwd(), "capstone 71/capstone 71/gpt2-recipes"),
            "C:/Users/apeks/Downloads/capstone 71/capstone 71/gpt2-recipes",
        ]
        
        # Filter out None values
        model_paths = [path for path in model_paths if path is not None]
        
        # Try to load from each path
        model_loaded = False
        for path in model_paths:
            if os.path.isdir(path):
                print(f"Trying to load model from: {path}")
                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(path)
                    self.model = GPT2LMHeadModel.from_pretrained(path)
                    print(f"Successfully loaded model from {path}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"Error loading model from {path}. Error: {e}")
        
        # Fall back to default gpt2 if no local model is found
        if not model_loaded:
            print("Falling back to default gpt2 model...")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create vector index
        print("Creating vector index...")
        self.create_index()
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the dataset"""
        # Fill NA values
        for col in ['TranslatedIngredients', 'Cuisine', 'Dietary Preference']:
            if col in df.columns:
                df[col] = df[col].fillna("")
        
        # Create combined text for embedding
        df["combined"] = df["TranslatedIngredients"].astype(str) + " " + \
                        df["Cuisine"].astype(str) + " " + \
                        df["Dietary Preference"].astype(str)
        
        return df
    
    def create_index(self):
        """Create FAISS index for fast similarity search"""
        # Create embeddings
        print("Creating embeddings...")
        corpus_embeddings = self.embedder.encode(
            self.df["combined"].tolist(), 
            convert_to_numpy=True, 
            show_progress_bar=True,
            device=self.device
        )
        
        # Create FAISS index
        dimension = corpus_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(corpus_embeddings)
        
        print(f"Index created with {self.index.ntotal} vectors of dimension {dimension}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant recipes based on query
        
        Args:
            query: User query
            top_k: Number of recipes to retrieve
            
        Returns:
            List of retrieved recipes
        """
        # Encode query
        query_embedding = self.embedder.encode([query], convert_to_numpy=True, device=self.device)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get relevant recipes
        retrieved_recipes = []
        for i, idx in enumerate(indices[0]):
            recipe = self.df.iloc[idx].to_dict()
            recipe['relevance_score'] = float(1 / (1 + distances[0][i]))  # Convert distance to score
            retrieved_recipes.append(recipe)
        
        return retrieved_recipes
    
    def generate(self, 
                 dietary_pref: str, 
                 ingredients: str, 
                 cuisine: str, 
                 top_k: int = 5, 
                 max_length: int = 800,
                 temperature: float = 0.7) -> str:
        """
        Generate a recipe using RAG
        
        Args:
            dietary_pref: Dietary preference (e.g., Vegan, Vegetarian)
            ingredients: Comma-separated list of ingredients
            cuisine: Type of cuisine
            top_k: Number of recipes to retrieve
            max_length: Maximum length of generated recipe
            temperature: Sampling temperature
            
        Returns:
            Generated recipe
        """
        try:
            # Create query
            query = f"{ingredients} {cuisine} {dietary_pref}"
            
            # Retrieve similar recipes
            retrieved_recipes = self.retrieve(query, top_k)
            
            # Create a detailed but concise prompt with clear structure expectations
            prompt = f"Write a complete {dietary_pref} {cuisine} recipe using these ingredients: {ingredients}.\n\n"
            prompt += "Format the recipe as follows:\n"
            prompt += "1. Recipe title\n"
            prompt += "2. List of all ingredients with quantities\n"
            prompt += "3. Numbered step-by-step cooking instructions\n\n"
            
            # Add example structure to guide the generation
            prompt += "Example structure (but create a different recipe):\n\n"
            prompt += "# Spicy Tomato Curry\n\n"
            prompt += "## Ingredients\n"
            prompt += "- 2 cups spinach, chopped\n"
            prompt += "- 3 tomatoes, diced\n"
            prompt += "- 4 cloves garlic, minced\n"
            prompt += "- 1 tbsp olive oil\n"
            prompt += "- 1 tsp cumin\n"
            prompt += "- Salt to taste\n\n"
            prompt += "## Instructions\n"
            prompt += "1. Heat oil in a pan over medium heat\n"
            prompt += "2. Add garlic and sauté until fragrant\n"
            prompt += "3. Add tomatoes and cook for 5 minutes\n"
            prompt += "4. Add spices and stir\n"
            prompt += "5. Add spinach and cook until wilted\n"
            prompt += "6. Serve hot\n\n"
            
            prompt += "Now create a complete, original recipe using the ingredients provided:\n\n"
            
            # Tokenize with truncation to avoid exceeding the model's context length
            input_ids = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512  # Longer context to accommodate the prompt
            ).to(self.device)
            
            # Generate text with proper attention mask
            attention_mask = torch.ones_like(input_ids)
            
            # Generate with smaller output size
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=500,  # Longer generation for complete recipe
                    do_sample=True,
                    temperature=temperature,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2  # Avoid repetition
                )
            
            # Decode
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract the generated recipe (remove the prompt part)
            recipe_text = generated_text[len(prompt):].strip()
            
            # Post-process to ensure we have a complete recipe
            recipe_text = self.post_process_recipe(recipe_text, ingredients, cuisine, dietary_pref)
                
            return recipe_text
            
        except Exception as e:
            print(f"Error during generation: {e}")
            # Fallback recipe generation
            return self.generate_fallback_recipe(dietary_pref, ingredients, cuisine)
    
    def post_process_recipe(self, recipe_text, ingredients, cuisine, dietary_pref):
        """Clean up and enhance the generated recipe if needed"""
        # Check if we have a title, ingredients and instructions
        has_title = "#" in recipe_text or "Title:" in recipe_text
        has_ingredients = "## Ingredients" in recipe_text or "Ingredients:" in recipe_text
        has_instructions = "## Instructions" in recipe_text or "Instructions:" in recipe_text
        
        # If missing major sections, regenerate with template
        if not (has_title and has_ingredients and has_instructions):
            print("Missing sections in recipe, applying template...")
            
            # Extract any useful content that was generated
            lines = recipe_text.split('\n')
            instructions = []
            for line in lines:
                # Look for numbered instructions or steps
                if re.match(r'^\d+\.?\s+\S+', line):
                    instructions.append(line)
            
            # If we found instructions, use them, otherwise start from scratch
            if not instructions:
                return self.generate_fallback_recipe(dietary_pref, ingredients, cuisine)
            
            # Build a structured recipe using what we have
            ingredient_list = [ing.strip() for ing in ingredients.split(",")]
            
            # Create structured output
            processed_recipe = f"# {cuisine.title()} {dietary_pref} with {ingredient_list[0].title()}\n\n"
            processed_recipe += "## Ingredients\n"
            
            # Add quantities to ingredients
            for ing in ingredient_list:
                quantity = "1 cup" if "spinach" in ing or "rice" in ing else "2-3"
                processed_recipe += f"- {quantity} {ing}\n"
            
            # Add basic spices based on cuisine
            if "Indian" in cuisine:
                processed_recipe += "- 1 tsp cumin\n"
                processed_recipe += "- 1/2 tsp turmeric\n"
                processed_recipe += "- 1 tsp garam masala\n"
            elif "Italian" in cuisine:
                processed_recipe += "- 1 tsp dried oregano\n"
                processed_recipe += "- 1 tsp dried basil\n"
            
            processed_recipe += "- Salt to taste\n"
            processed_recipe += "- 2 tbsp cooking oil\n\n"
            
            # Add instructions
            processed_recipe += "## Instructions\n"
            for i, instruction in enumerate(instructions):
                # Clean up the instruction format
                instruction = re.sub(r'^\d+\.?\s*', '', instruction)
                processed_recipe += f"{i+1}. {instruction}\n"
            
            return processed_recipe
        
        return recipe_text
    
    def generate_fallback_recipe(self, dietary_pref: str, ingredients: str, cuisine: str) -> str:
        """Generate a fallback recipe when the model fails"""
        print("Using fallback recipe generation...")
        fallback = f"# {cuisine.title()} {dietary_pref} Recipe\n\n"
        fallback += f"## Ingredients\n"
        
        # Format the ingredients list with quantities
        ingredient_list = [ing.strip() for ing in ingredients.split(",")]
        for ing in ingredient_list:
            # Add sensible quantities based on ingredient type
            if "spinach" in ing.lower():
                fallback += f"- 2 cups {ing}\n"
            elif "tomato" in ing.lower():
                fallback += f"- 3 {ing}s, chopped\n"
            elif "garlic" in ing.lower():
                fallback += f"- 4 cloves of {ing}, minced\n"
            elif "onion" in ing.lower():
                fallback += f"- 1 {ing}, finely chopped\n"
            else:
                fallback += f"- 1 cup {ing}\n"
        
        # Add common spices based on cuisine
        if "Indian" in cuisine:
            fallback += "- 1 tsp cumin seeds\n"
            fallback += "- 1/2 tsp turmeric powder\n"
            fallback += "- 1 tsp garam masala\n"
            fallback += "- 1 tsp coriander powder\n"
        elif "Italian" in cuisine:
            fallback += "- 1 tsp dried oregano\n"
            fallback += "- 1 tsp dried basil\n"
            fallback += "- 1/2 tsp red pepper flakes (optional)\n"
        elif "Mexican" in cuisine:
            fallback += "- 1 tsp cumin powder\n"
            fallback += "- 1 tsp chili powder\n"
            fallback += "- 1 lime, juiced\n"
        
        fallback += "- 2 tbsp cooking oil\n"
        fallback += "- Salt to taste\n"
        
        fallback += f"\n## Instructions\n"
        
        # Create more detailed, cuisine-specific instructions
        if "Indian" in cuisine:
            fallback += "1. Heat oil in a pan over medium heat.\n"
            fallback += "2. Add cumin seeds and let them splutter.\n"
            
            if "garlic" in ingredients:
                fallback += "3. Add minced garlic and sauté until golden brown.\n"
                step = 4
            else:
                step = 3
                
            if "onion" in ingredients: 
                fallback += f"{step}. Add chopped onions and sauté until translucent.\n"
                step += 1
            
            if "tomato" in ingredients:
                fallback += f"{step}. Add chopped tomatoes and cook until soft and oil separates.\n"
                step += 1
                
            fallback += f"{step}. Add turmeric, coriander powder, and mix well.\n"
            step += 1
            
            if "spinach" in ingredients:
                fallback += f"{step}. Add spinach and cook until wilted, about 3-4 minutes.\n"
                step += 1
            
            for ing in ingredient_list:
                if ing.lower() not in ["spinach", "tomato", "garlic", "onion"]:
                    fallback += f"{step}. Add {ing} and mix well.\n"
                    step += 1
            
            fallback += f"{step}. Add salt to taste and garam masala.\n"
            step += 1
            fallback += f"{step}. Cover and simmer for 5 minutes on low heat.\n"
            step += 1
            fallback += f"{step}. Serve hot with rice or roti.\n"
            
        elif "Italian" in cuisine:
            fallback += "1. Heat oil in a pan over medium heat.\n"
            
            if "garlic" in ingredients.lower():
                fallback += "2. Add minced garlic and sauté until fragrant, about 30 seconds.\n"
                step = 3
            else:
                step = 2
            
            if "onion" in ingredients.lower():
                fallback += f"{step}. Add chopped onions and sauté until translucent.\n"
                step += 1
            
            # Handle pasta differently
            has_pasta = any("pasta" in ing.lower() for ing in ingredient_list)
            
            if has_pasta:
                fallback += f"{step}. Bring a large pot of salted water to a boil for the pasta.\n"
                step += 1
            
            if "mushroom" in ingredients.lower():
                fallback += f"{step}. Add mushrooms and cook until they release their moisture and start to brown, about 5-7 minutes.\n"
                step += 1
                
            if "bell pepper" in ingredients.lower() or "pepper" in ingredients.lower():
                fallback += f"{step}. Add bell peppers and cook until slightly softened, about 3-4 minutes.\n"
                step += 1
            
            if "tomato" in ingredients.lower():
                fallback += f"{step}. Add chopped tomatoes, dried herbs, and cook for 5 minutes.\n"
                step += 1
            
            if "spinach" in ingredients.lower():
                fallback += f"{step}. Add spinach and cook until wilted, about 2 minutes.\n"
                step += 1
            
            # Add other ingredients that aren't mentioned specifically
            for ing in ingredient_list:
                if not any(x in ing.lower() for x in ["spinach", "tomato", "garlic", "onion", "pasta", "olive oil", "mushroom", "bell pepper", "pepper"]):
                    fallback += f"{step}. Add {ing} and cook for 3-4 minutes.\n"
                    step += 1
            
            # Add olive oil if present
            if "olive oil" in ingredients.lower():
                fallback += f"{step}. Drizzle with olive oil.\n"
                step += 1
            
            fallback += f"{step}. Season with salt, pepper, and dried herbs to taste.\n"
            step += 1
            
            # Cook pasta if present
            if has_pasta:
                fallback += f"{step}. Cook pasta according to package instructions until al dente.\n"
                step += 1
                fallback += f"{step}. Drain pasta, reserving 1/4 cup of pasta water.\n"
                step += 1
                fallback += f"{step}. Add pasta to the sauce along with a splash of pasta water and toss to combine.\n"
                step += 1
            
            fallback += f"{step}. Serve hot, optionally garnished with grated cheese for non-vegan option.\n"
            
        else:
            fallback += "1. Heat oil in a pan over medium heat.\n"
            
            if "garlic" in ingredients:
                fallback += "2. Add minced garlic and sauté until fragrant.\n"
                step = 3
            else:
                step = 2
                
            if "onion" in ingredients:
                fallback += f"{step}. Add chopped onions and sauté until translucent.\n"
                step += 1
            
            if "tomato" in ingredients:
                fallback += f"{step}. Add chopped tomatoes and cook for 5 minutes.\n"
                step += 1
            
            for ing in ingredient_list:
                if ing.lower() not in ["spinach", "tomato", "garlic", "onion"]:
                    fallback += f"{step}. Add {ing} and cook for 3-4 minutes.\n"
                    step += 1
            
            if "spinach" in ingredients:
                fallback += f"{step}. Add spinach last and cook until wilted, about 2-3 minutes.\n"
                step += 1
            
            fallback += f"{step}. Season to taste with salt and pepper.\n"
            step += 1
            fallback += f"{step}. Serve hot.\n"
        
        return fallback
    
    def save_recipe(self, recipe: str, filename: str):
        """Save a generated recipe to a file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(recipe)
        print(f"Recipe saved to {filename}")

# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = RecipeRAG()
    
    # User inputs
    dietary_pref = "Vegan"
    ingredients = "spinach, tomato, garlic, chickpeas, coconut milk"
    cuisine = "Indian"
    
    # Generate recipe
    recipe = rag.generate(dietary_pref, ingredients, cuisine)
    
    print("\n" + "="*50 + "\n")
    print("Generated Recipe:\n")
    print(recipe)
    print("\n" + "="*50 + "\n")
    
    # Save recipe
    rag.save_recipe(recipe, "generated_recipe.txt") 