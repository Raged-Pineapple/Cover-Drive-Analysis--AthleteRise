"""
Gemini 2.5 Pro API integration for cricket cover drive analysis.
Provides intelligent analysis of evaluation data.
"""

import os
import json
from typing import Dict, Any
import google.generativeai as genai
import pandas as pd

class GeminiAnalyzer:
    """Analyzes cricket cover drive data using Gemini 2.5 Pro AI."""
    
    def __init__(self, api_key: str = None):
        """Initialize with optional API key."""
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided or set as GOOGLE_API_KEY")
        
        # Configure the Gemini client
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.5-pro")
    
    def analyze_evaluation(self, evaluation: Dict[str, Any]) -> Dict[str, str]:
        """Analyze evaluation data and return Gemini's analysis.
        
        Converts DataFrames to JSON-compatible format before sending to Gemini.
        """
        # Convert any DataFrames to JSON-compatible format
        def convert_dataframe(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            return obj
            
        # Convert evaluation data to JSON-compatible format
        json_compatible_data = {}
        for key, value in evaluation.items():
            json_compatible_data[key] = convert_dataframe(value)
        
        prompt = f"""
        You are a sports performance analysis AI. Based on this evaluation data:
        {json.dumps(json_compatible_data, indent=2)}
        
        Generate:
        - Phase Analysis (2-4 sentences)
        - Contact Analysis (2-4 sentences)
        - Movement Smoothness (2-4 sentences)
        
        Return the results in valid JSON format with keys:
        phase_analysis, contact_analysis, movement_smoothness
        """
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Extract JSON from the response even if extra text is present
            json_start = result_text.find("{")
            json_end = result_text.rfind("}") + 1
            json_str = result_text[json_start:json_end]
            
            analysis = json.loads(json_str)
            return {
                'phase_analysis': analysis.get('phase_analysis', ''),
                'contact_analysis': analysis.get('contact_analysis', ''),
                'movement_smoothness': analysis.get('movement_smoothness', '')
            }
        except Exception as e:
            return {
                'phase_analysis': f"Error: {str(e)}",
                'contact_analysis': f"Error: {str(e)}",
                'movement_smoothness': f"Error: {str(e)}"
            }

    def get_gemini_analysis(self, evaluation_path: str) -> Dict[str, str]:
        """Get analysis from Gemini for a given evaluation file."""
        try:
            with open(evaluation_path, 'r') as f:
                evaluation = json.load(f)
            return self.analyze_evaluation(evaluation)
        except Exception as e:
            return {
                'phase_analysis': f"Error: {str(e)}",
                'contact_analysis': f"Error: {str(e)}",
                'movement_smoothness': f"Error: {str(e)}"
            }
    def extract_text(response):
        """Safely extract text from a Gemini response."""
        texts = []
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    texts.append(part.text)
        return "\n".join(texts) if texts else "⚠️ Gemini returned no usable text."



if __name__ == "__main__":
    # Example usage: reads API key from environment for safety
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("GOOGLE_API_KEY not set. Create a .env file and set GOOGLE_API_KEY=<your_key>.")
    else:
        analyzer = GeminiAnalyzer(api_key=api_key)
        # Provide a valid path to an evaluation JSON file if you want to test locally
        eval_path = os.getenv('EVALUATION_JSON', 'evaluation.json')
        analysis = analyzer.get_gemini_analysis(eval_path)
        print(json.dumps(analysis, indent=2))
