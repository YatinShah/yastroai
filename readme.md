
- The document to train the model was downloaded from project gutenburge, related to astrology

## run following commands to setup the projects first time with glcoud.

### Update package list and install dependencies
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates gnupg curl

### Add the Google Cloud public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

### Add the gcloud CLI distribution URI as a package source
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

### Update and install the CLI
sudo apt-get update && sudo apt-get install google-cloud-cli
### Log in and set your default project
gcloud init

### Set up your Application Default Credentials (ADC)
gcloud auth application-default login

### Set the default region
gcloud config set compute/region us-central1

### Set the default zone within that region
gcloud config set compute/zone us-central1-a

gcloud config list

### models in central1 region
gcloud ai models list --region=us-central1

### Generate credentials/vertex-ai-key.json file, as gemini to create `service account credentials JSON file`.

### To run the application with creating embeddings follow these commands locally:
- convert the data/*.txt files to pdf files in same folder.
```
$> python 31multimodal_ingest.py #reads all the pdf files from the data folder, it reads text as well as images. Creates embeddings in the vector db.
$> python 4ask_rag.py #creates rag pipeline from the vector db
$> streamlit run 5app.py #invokes the app and presents nice UI.
```

### To deploy to gcloud, this only runs the ui, the vector db needs to be moved to cloud or use google's VertexAI Vector db solution:
```
$> gcloud run deploy my-rag-app \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501


```



# Refs:
- list of gemini embedding models. (https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/)
- Embedding model compare: (https://milvus.io/blog/choose-embedding-model-rag-2026.md)^^VGood^^





# TODO
Use newer libraries, and remove deprecated libraries/ objects from use. eg. use GenAi library instead of VertexAI
Use dockerized embedding model. e.g. Qwen3-VL-2B 
Use a 3072 vector embeddings instead of 768 currently in use.

