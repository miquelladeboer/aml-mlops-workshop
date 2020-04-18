"""
Helper to generate an AAD token
"""
from azureml.core.authentication import AzureCliAuthentication


print(
    AzureCliAuthentication().get_authentication_header().get("Authorization")
)
