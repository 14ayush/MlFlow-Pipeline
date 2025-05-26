import os
import certifi

# Set SSL_CERT_FILE to certifi's CA bundle
os.environ["SSL_CERT_FILE"] = certifi.where()

import httpx

response = httpx.get("https://www.google.com")
print(response.status_code)
