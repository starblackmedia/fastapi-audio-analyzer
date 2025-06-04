from google.cloud import aiplatform

aiplatform.init(project='tuk-8cf2f', location='us-central1')

model = aiplatform.Model.upload(
    display_name="african-genre-model",
    artifact_uri="gs://tuk-8cf2f-models",  # <--- just the bucket or folder, no file name
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest",
)

print(f"Model uploaded: {model.resource_name}")
