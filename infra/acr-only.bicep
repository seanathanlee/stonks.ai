// ============================================================
// ACR-only Bicep template – Stonks.ai
// Provisions only the Azure Container Registry so it exists
// before the main Bicep deployment runs. The deploy workflow
// uses this to create the ACR, push the real image, then run
// main.bicep with the real image URI — eliminating the need
// for a placeholder image.
// ============================================================

@description('Base name used to derive resource names.')
param baseName string = 'stonksai'

@description('Azure region for all resources.')
param location string = resourceGroup().location

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: '${baseName}acr'
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
    publicNetworkAccess: 'Enabled'
  }
}

output acrLoginServer string = acr.properties.loginServer
output acrName string = acr.name
