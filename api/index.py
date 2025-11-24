from mangum import Mangum
from main import app

# Wrap the FastAPI app with Mangum for serverless
handler = Mangum(app, lifespan="off")
