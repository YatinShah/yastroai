import vertexai
from vertexai.generative_models import GenerativeModel

PROJECTID= "atroai"
REGION="us-central1"

def test_vertex():
    vertexai.init(project=PROJECTID,location=REGION)
    # Load the Gemini 1.5 Pro model (great for multimodal tasks)
    model = GenerativeModel("gemini-1.5-pro-001")
    
    # Send a test prompt
    try:
        response = model.generate_content("Hello! Are you receiving this from my local Ubuntu environment?")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Failed to generate content: {e}")


def check_model_availability(project=PROJECTID,location="global"):
    # Initialize the Vertex AI SDK for the specific region
    vertexai.init(project=project, location=location)

    # List of models you want to check
    models_to_test = [
        "gemini-1.5-pro-001",
        "gemini-1.5-flash-001",
        "gemini-1.0-pro-001" ,
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-3.1-pro-preview"
    ]

    for model_name in models_to_test:
        try:
            model = GenerativeModel(model_name)
            # We send a tiny prompt just to validate the endpoint
            model.generate_content("test") 
            print(f"✅ {model_name} is AVAILABLE in {REGION}")
        except Exception as e:
            print(f"❌ {model_name} is NOT available in {REGION}. Error: {e}")

if __name__=="__main__":
    check_model_availability(PROJECTID,REGION)
    test_vertex()