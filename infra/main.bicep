// ============================================================
// Main Bicep template – Stonks.ai Azure infrastructure
// Provisions:
//   • Azure AI Services (Cognitive Services) account
// ============================================================

@description('Base name used to derive all resource names.')
param baseName string = 'stonksai'

@description('Azure region for all resources.')
param location string = resourceGroup().location

@description('SKU for the Cognitive Services account.')
@allowed(['S0'])
param sku string = 'S0'

// ============================================================
// Azure AI Services (Cognitive Services) account
// ============================================================

resource cognitiveServices 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: '${baseName}-ai'
  location: location
  kind: 'AIServices'
  sku: {
    name: sku
  }
  properties: {
    customSubDomainName: '${baseName}-ai'
    publicNetworkAccess: 'Enabled'
  }
}

// ============================================================
// Outputs
// ============================================================

output cognitiveServicesEndpoint string = cognitiveServices.properties.endpoint
output cognitiveServicesId string = cognitiveServices.id
