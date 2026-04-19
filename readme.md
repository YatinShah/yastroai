
# Update package list and install dependencies
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates gnupg curl

# Add the Google Cloud public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

# Add the gcloud CLI distribution URI as a package source
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Update and install the CLI
sudo apt-get update && sudo apt-get install google-cloud-cli
# Log in and set your default project
gcloud init

# Set up your Application Default Credentials (ADC)
gcloud auth application-default login

# Set the default region
gcloud config set compute/region us-central1

# Set the default zone within that region
gcloud config set compute/zone us-central1-a

gcloud config list

# models in central1 region
gcloud ai models list --region=us-central1

to run the webui run, `streamlit run 5app.py`

- list of gemini embedding models. (https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/)
- Embedding model compare: (https://milvus.io/blog/choose-embedding-model-rag-2026.md)^^VGood^^

--TODO
Use newer libraries, and remove deprecated libraries/ objects from use. eg. use GenAi library instead of VertexAI
Use dockerized embedding model. e.g. Qwen3-VL-2B 

