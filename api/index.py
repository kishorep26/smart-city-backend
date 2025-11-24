import os
os.chdir('/var/task')

from main import app
from mangum import Mangum

handler = Mangum(app, lifespan="off")
