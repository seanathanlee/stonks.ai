// ============================================================
// Azure Container App module
// ============================================================

@description('Name of the Container App.')
param name string

@description('Azure region.')
param location string

@description('Resource ID of the Container Apps Managed Environment.')
param containerAppsEnvId string

@description('Full image reference, e.g. <acr>.azurecr.io/stonksai-agent:latest')
param containerImage string

@description('ACR login server hostname.')
param acrLoginServer string

@description('ACR admin username.')
param acrAdminUsername string

@description('ACR admin password.')
@secure()
param acrAdminPassword string

@description('Azure OpenAI endpoint URL.')
param azureOpenAiEndpoint string

@description('Azure OpenAI API key.')
@secure()
param azureOpenAiApiKey string

@description('Azure OpenAI deployment name.')
param azureOpenAiDeployment string

@description('CPU cores (e.g. "0.5").')
param cpuCores string

@description('Memory (e.g. "1.0Gi").')
param memoryGi string

resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: name
  location: location
  properties: {
    managedEnvironmentId: containerAppsEnvId
    configuration: {
      secrets: [
        {
          name: 'acr-password'
          value: acrAdminPassword
        }
        {
          name: 'azure-openai-api-key'
          value: azureOpenAiApiKey
        }
      ]
      registries: [
        {
          server: acrLoginServer
          username: acrAdminUsername
          passwordSecretRef: 'acr-password'
        }
      ]
      ingress: {
        external: true
        targetPort: 8080
        transport: 'auto'
      }
    }
    template: {
      containers: [
        {
          name: 'agent'
          image: containerImage
          resources: {
            cpu: json(cpuCores)
            memory: memoryGi
          }
          env: [
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: azureOpenAiEndpoint
            }
            {
              name: 'AZURE_OPENAI_API_KEY'
              secretRef: 'azure-openai-api-key'
            }
            {
              name: 'AZURE_OPENAI_DEPLOYMENT'
              value: azureOpenAiDeployment
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 3
      }
    }
  }
}

output fqdn string = containerApp.properties.configuration.ingress.fqdn
