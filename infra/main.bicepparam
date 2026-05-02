using './main.bicep'

param baseName = 'stonksai'
param containerImage = 'stonksaiacr.azurecr.io/stonksai-agent:latest'
param azureOpenAiEndpoint = 'https://<your-resource-name>.openai.azure.com/'
param azureOpenAiDeployment = 'gpt-4o'

// azureOpenAiApiKey must be supplied at deploy time via --parameters azureOpenAiApiKey=<value>
// Never commit secrets to source control.
