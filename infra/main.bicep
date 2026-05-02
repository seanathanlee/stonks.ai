// ============================================================
// Main Bicep template – Stonks.ai Azure infrastructure
// Provisions:
//   • Azure AI Services (Cognitive Services) account
//   • Azure Data Explorer (Kusto) cluster + database + tables
// ============================================================

@description('Base name used to derive all resource names.')
param baseName string = 'stonksai'

@description('Azure region for all resources.')
param location string = resourceGroup().location

@description('SKU for the Cognitive Services account.')
@allowed(['S0'])
param sku string = 'S0'

@description('SKU name for the Azure Data Explorer cluster.')
@allowed(['Dev(No SLA)_Standard_D11_v2', 'Standard_D11_v2', 'Standard_D12_v2'])
param adxSkuName string = 'Dev(No SLA)_Standard_D11_v2'

@description('SKU tier for the Azure Data Explorer cluster.')
@allowed(['Basic', 'Standard'])
param adxSkuTier string = 'Basic'

@description('Number of ADX cluster instances.')
param adxCapacity int = 1

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
// Azure Data Explorer (Kusto) cluster
// ============================================================

resource adxCluster 'Microsoft.Kusto/clusters@2023-08-15' = {
  name: '${baseName}adx'
  location: location
  sku: {
    name: adxSkuName
    tier: adxSkuTier
    capacity: adxCapacity
  }
  properties: {
    enableStreamingIngest: true
    enablePurge: false
    publicNetworkAccess: 'Enabled'
  }
}

// ============================================================
// ADX database: stonksai
// ============================================================

resource adxDatabase 'Microsoft.Kusto/clusters/databases@2023-08-15' = {
  name: 'stonksai'
  parent: adxCluster
  location: location
  kind: 'ReadWrite'
  properties: {
    softDeletePeriod: 'P365D'
    hotCachePeriod: 'P31D'
  }
}

// ============================================================
// ADX table: dailyStockPrice
//   Columns: reportTime (datetime), symbol (string),
//            price (real), priceDate (datetime)
// ============================================================

resource dailyStockPriceTable 'Microsoft.Kusto/clusters/databases/scripts@2023-08-15' = {
  name: 'create-daily-stock-price-table'
  parent: adxDatabase
  properties: {
    scriptContent: '''
.create-merge table dailyStockPrice (
    reportTime: datetime,
    symbol: string,
    price: real,
    priceDate: datetime
)

.alter-merge table dailyStockPrice policy retention softdelete = 365d

.create-or-alter table dailyStockPrice ingestion json mapping 'dailyStockPriceMapping'
'[{"column":"reportTime","path":"$.reportTime","datatype":"datetime"},{"column":"symbol","path":"$.symbol","datatype":"string"},{"column":"price","path":"$.price","datatype":"real"},{"column":"priceDate","path":"$.priceDate","datatype":"datetime"}]'
'''
    continueOnErrors: false
  }
}

// ============================================================
// ADX table: agentStockForecast
//   Columns: reportTime (datetime), agentName (string),
//            symbol (string), horizon (string),
//            expectedReturn (real), rank (int)
// ============================================================

resource agentStockForecastTable 'Microsoft.Kusto/clusters/databases/scripts@2023-08-15' = {
  name: 'create-agent-stock-forecast-table'
  parent: adxDatabase
  properties: {
    scriptContent: '''
.create-merge table agentStockForecast (
    reportTime: datetime,
    agentName: string,
    symbol: string,
    horizon: string,
    expectedReturn: real,
    rank: int
)

.alter-merge table agentStockForecast policy retention softdelete = 365d

.create-or-alter table agentStockForecast ingestion json mapping 'agentStockForecastMapping'
'[{"column":"reportTime","path":"$.reportTime","datatype":"datetime"},{"column":"agentName","path":"$.agentName","datatype":"string"},{"column":"symbol","path":"$.symbol","datatype":"string"},{"column":"horizon","path":"$.horizon","datatype":"string"},{"column":"expectedReturn","path":"$.expectedReturn","datatype":"real"},{"column":"rank","path":"$.rank","datatype":"int"}]'
'''
    continueOnErrors: false
  }
}

// ============================================================
// Outputs
// ============================================================

output cognitiveServicesEndpoint string = cognitiveServices.properties.endpoint
output cognitiveServicesId string = cognitiveServices.id
output adxClusterUri string = adxCluster.properties.uri
output adxClusterId string = adxCluster.id
output adxDatabaseName string = adxDatabase.name
