import requests
from google.auth import default
from google.auth.transport.requests import Request
from google.genai import Client

PROJECTID = "atroai"
REGION = "us-central1"


def init_vertexai_client(project=PROJECTID, location=REGION):
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    if not credentials.valid:
        credentials.refresh(Request())
    return Client(vertexai=True, credentials=credentials, project=project, location=location)


def extract_text_from_response(response):
    if not response or not response.candidates:
        return ""

    text_parts = []
    for candidate in response.candidates:
        content = getattr(candidate, "content", None)
        if not content or not getattr(content, "parts", None):
            continue
        for part in content.parts:
            part_text = getattr(part, "text", None)
            if part_text:
                text_parts.append(part_text)

    return "".join(text_parts).strip()


def test_vertex():
    client = init_vertexai_client()
    try:
        response = client.models.generate_content(
            model="gemini-1.5-pro-001",
            contents="Hello! Are you receiving this from my local Ubuntu environment?",
        )
        print(f"Response: {extract_text_from_response(response)}")
    except Exception as e:
        print(f"❌ Failed to generate content: {e}")


def list_available_models(project=PROJECTID, location=REGION):
    try:
        client = init_vertexai_client(project=project, location=location)
        models = client.models.list(config={})

        print(f"Available models in {location} (from GenAI Vertex AI API):")
        found = False
        for model in models:
            found = True
            display_name = getattr(model, "display_name", None) or model.name
            print(f"- {display_name} ({model.name})")

        if not found:
            print("No models found in this project/region via the API.")
    except Exception as e:
        print(f"❌ Failed to list models via API: {e}")


def list_embedding_models(project=PROJECTID, location=REGION):
    try:
        client = init_vertexai_client(project=project, location=location)
        models = client.models.list(config={})

        embedding_models = []
        for model in models:
            name = getattr(model, "name", "")
            display_name = getattr(model, "display_name", "") or ""
            if "embed" in name.lower() or "embed" in display_name.lower():
                version = getattr(model, "version", "")
                embedding_models.append((name, version))

        print(f"Available embedding models in {location} (from GenAI Vertex AI API):")
        if not embedding_models:
            print("No embedding models found.")
            return

        for name, version in embedding_models:
            print(f"- {name}@{version}")
    except Exception as e:
        print(f"❌ Failed to list embedding models via GenAI API: {e}")


def check_model_availability(project=PROJECTID, location=REGION):
    client = init_vertexai_client(project=project, location=location)

    models_to_test = [
        "gemini-1.5-pro-001",
        "gemini-1.5-flash-001",
        "gemini-1.0-pro-001",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-3.1-pro-preview",
    ]

    for model_name in models_to_test:
        try:
            client.models.generate_content(model=model_name, contents="test")
            print(f"✅ {model_name} is AVAILABLE in {location}")
        except Exception as e:
            print(f"❌ {model_name} is NOT available in {location}. Error: {e}")


if __name__ == "__main__":
    list_available_models(PROJECTID, REGION)
    list_embedding_models(PROJECTID, REGION)
    check_model_availability(PROJECTID, REGION)
    test_vertex()