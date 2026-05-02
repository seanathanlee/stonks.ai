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

@description('Azure region for the Static Web App. Must be a region that supports Azure Static Web Apps.')
@allowed([
  'centralus'
  'eastus2'
  'eastasia'
  'westeurope'
  'westus2'
])
param swaLocation string = 'centralus'

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
    #disable-next-line use-secure-value-for-secure-inputs
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
    #disable-next-line use-secure-value-for-secure-inputs
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
// ADX materialized view: dailyStockPriceMV
//   Deduplicates dailyStockPrice by (symbol, priceDate),
//   retaining the row with the latest reportTime.
// ============================================================

resource dailyStockPriceMVScript 'Microsoft.Kusto/clusters/databases/scripts@2023-08-15' = {
  name: 'create-daily-stock-price-mv'
  parent: adxDatabase
  dependsOn: [dailyStockPriceTable]
  properties: {
    #disable-next-line use-secure-value-for-secure-inputs
    scriptContent: '''
.create-or-alter materialized-view with (backfill=true) dailyStockPriceMV on table dailyStockPrice
{
    dailyStockPrice
    | summarize arg_max(reportTime, price) by symbol, priceDate
}
'''
    continueOnErrors: false
  }
}

// ============================================================
// ADX materialized view: agentStockForecastMV
//   Deduplicates agentStockForecast by (agentName, symbol,
//   horizon), retaining the row with the latest reportTime.
// ============================================================

resource agentStockForecastMVScript 'Microsoft.Kusto/clusters/databases/scripts@2023-08-15' = {
  name: 'create-agent-stock-forecast-mv'
  parent: adxDatabase
  dependsOn: [agentStockForecastTable]
  properties: {
    #disable-next-line use-secure-value-for-secure-inputs
    scriptContent: '''
.create-or-alter materialized-view with (backfill=true) agentStockForecastMV on table agentStockForecast
{
    agentStockForecast
    | summarize arg_max(reportTime, expectedReturn, rank) by agentName, symbol, horizon
}
'''
    continueOnErrors: false
  }
}

// ============================================================
// Azure Static Web App — skewthis.com frontend
// ============================================================

resource staticWebApp 'Microsoft.Web/staticSites@2023-01-01' = {
  name: '${baseName}-web'
  location: swaLocation
  sku: {
    name: 'Free'
    tier: 'Free'
  }
  properties: {
    buildProperties: {
      skipGithubActionWorkflowGeneration: true
    }
  }
}

// Apex domain: skewthis.com — requires a DNS TXT record for validation.
// Before this resource can reach the "Validated" state you must add a
// TXT record at your DNS provider:
//   Name:  @ (or skewthis.com)
//   Type:  TXT
//   Value: <validationToken shown in the Azure Portal for this domain>
resource customDomainApex 'Microsoft.Web/staticSites/customDomains@2023-01-01' = {
  name: 'skewthis.com'
  parent: staticWebApp
  properties: {
    validationMethod: 'dns-txt-token'
  }
}

// www subdomain: www.skewthis.com — requires a CNAME record for validation.
// Add a DNS CNAME record at your DNS provider:
//   Name:  www
//   Type:  CNAME
//   Value: ${staticWebApp.properties.defaultHostname}
resource customDomainWww 'Microsoft.Web/staticSites/customDomains@2023-01-01' = {
  name: 'www.skewthis.com'
  parent: staticWebApp
  properties: {
    validationMethod: 'cname-delegation'
  }
  dependsOn: [customDomainApex]
}

// ============================================================
// Outputs
// ============================================================

output cognitiveServicesEndpoint string = cognitiveServices.properties.endpoint
output cognitiveServicesId string = cognitiveServices.id
output adxClusterUri string = adxCluster.properties.uri
output adxClusterId string = adxCluster.id
output adxDatabaseName string = adxDatabase.name
output staticWebAppName string = staticWebApp.name
output staticWebAppDefaultHostname string = staticWebApp.properties.defaultHostname
