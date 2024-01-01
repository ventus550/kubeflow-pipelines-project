from google.cloud import secretmanager
import yaml

class Secret:
    def __init__(self, secret_manager_resource_name):
        self.client = secretmanager.SecretManagerServiceClient()
        self.payload = self.client.access_secret_version(request={"name": secret_manager_resource_name}).payload
        self.data = yaml.safe_load(self.payload.data.decode("ascii"))
    
    def __getitem__(self, item):
        return self.data[item]
    
    def __dict__(self):
        return self.data
    
class Configs:
    def __init__(self, version="latest"):
        self.resource = f"projects/429426973958/secrets/protocell-config/versions/{version}"
        self.secret = None
    
    @property
    def data(self):
        return self.secret or Secret(self.resource).data
    
    @property
    def bucket(self):
        return self.data["BUCKET"]
    
    @property
    def location(self):
        return self.data["LOCATION"]
    
    @property
    def project(self):
        return self.data["PROJECT"]
    
    @property
    def artifactory(self):
        return f"https://{self.location}-kfp.pkg.dev/{self.project}/kubeflows"
    
    @property
    def docker(self):
        return self.data["DOCKER"]
    
    @property
    def model(self):
        return self.data["MODEL"]
    
    @property
    def pipeline_name(self):
        return f"{self.model}.yaml"
    
    @property
    def keras_image(self):
        return f"{self.location}-docker.pkg.dev/{self.project}/{self.docker}/keras:latest"
    
    @property
    def pipeline_directory(self):
        return f"{self.bucket}/pipeline"
    
    @property
    def data_directory(self):
        return f"{self.bucket}/data"
    
    @property
    def service_account(self):
        return self.data["SERVICE_ACCOUNT"]
        
configs = Configs()