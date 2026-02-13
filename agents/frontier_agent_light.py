import os
import json
from google import genai
from google.genai import types

def frontier_predict(features: dict, retrieved_cases: list):
    """
    Hybrid Predictor using the unified Google Gen AI SDK.
    Handles both AI Studio (API Key) and Vertex AI (IAM) authentication.
    """
    # 1. Configuration & Auth Setup
    # Modal secrets provide these env vars. 
    # If using Vertex AI, ensure GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION are set.
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    # 2. Setup Prompt
    prompt = f"""
    You are an insurance pricing assistant. 
    Use the retrieved similar cases to estimate a fair insurance price.

    Customer features:
    {json.dumps(features, indent=2)}

    Retrieved similar cases:
    {json.dumps(retrieved_cases, indent=2)}

    Your task:
    1. Estimate a numeric insurance price (USD).
    2. Provide a clear explanation of the reasoning.
    3. Return ONLY valid JSON in this exact format:
    {{
      "price": <numeric>,
      "explanation": "<string>"
    }}
    """

    # 3. Execution
    try:
        # Check if we should use Vertex AI (Enterprise) or Gemini Developer API (Studio)
        if project_id:
            # Initialize for Vertex AI
            client = genai.Client(
                vertexai=True, 
                project=project_id, 
                location=location
            )
        else:
            # Fallback to AI Studio API Key
            if not api_key:
                return None, "Error: No API Key or Project ID found."
            client = genai.Client(api_key=api_key)

        

        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            )
        )
        
        # Parse response
        data = json.loads(response.text)
        return float(data.get("price")), data.get("explanation", "No explanation provided.")
    
    except Exception as e:
        # Log the detailed error (like the 404 you saw) for debugging
        print(f"DEBUG: Gemini Call Failed: {str(e)}")
        return None, f"Gemini Error: {str(e)}"