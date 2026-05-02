// ============================================================
// Main Bicep template – Stonks.ai Azure infrastructure
// Provisions:
//   • Azure Container Registry (ACR)
//   • Log Analytics workspace
//   • Azure Container Apps Environment
//   • Azure Container App (the AI agent)
// ============================================================

@description('Base name used to derive all resource names.')
param baseName string = 'stonksai'

@description('Azure region for all resources.')
param location string = resourceGroup().location

@description('Container image to deploy, e.g. <acr>.azurecr.io/stonksai-agent:latest')
param containerImage string

@description('Azure OpenAI endpoint URL.')
param azureOpenAiEndpoint string

@description('Azure OpenAI API key (stored as a Container Apps secret).')
@secure()
param azureOpenAiApiKey string

@description('Azure OpenAI deployment name.')
param azureOpenAiDeployment string = 'gpt-4o'

@description('Container CPU allocation in cores.')
param cpuCores string = '0.5'

@description('Container memory allocation.')
param memoryGi string = '1.0Gi'

// ============================================================
// Azure Container Registry
// ============================================================

module acr 'modules/container-registry.bicep' = {
  name: 'acr'
  params: {
    name: '${baseName}acr'
    location: location
  }
}

// ============================================================
// Log Analytics workspace (required by Container Apps)
// ============================================================

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${baseName}-logs'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// ============================================================
// Container Apps Environment
// ============================================================

resource containerAppsEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: '${baseName}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// ============================================================
// Container App
// ============================================================

module containerApp 'modules/container-app.bicep' = {
  name: 'containerApp'
  params: {
    name: '${baseName}-agent'
    location: location
    containerAppsEnvId: containerAppsEnv.id
    containerImage: containerImage
    acrLoginServer: acr.outputs.loginServer
    acrAdminUsername: acr.outputs.adminUsername
    acrAdminPassword: acr.outputs.adminPassword
    azureOpenAiEndpoint: azureOpenAiEndpoint
    azureOpenAiApiKey: azureOpenAiApiKey
    azureOpenAiDeployment: azureOpenAiDeployment
    cpuCores: cpuCores
    memoryGi: memoryGi
  }
}

// ============================================================
// Outputs
// ============================================================

output acrLoginServer string = acr.outputs.loginServer
output containerAppFqdn string = containerApp.outputs.fqdn
