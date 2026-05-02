// ============================================================
// Azure Container Registry module
// ============================================================

@description('Name of the Container Registry (must be globally unique, lowercase, 5-50 chars).')
param name string

@description('Azure region.')
param location string

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: name
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    // Admin user is enabled so Container Apps can pull images using the
    // built-in ACR credential secret.  For production workloads with a
    // Standard or Premium SKU, prefer managed identity + RBAC instead.
    adminUserEnabled: true
  }
}

output loginServer string = acr.properties.loginServer
output adminUsername string = acr.listCredentials().username
output adminPassword string = acr.listCredentials().passwords[0].value
