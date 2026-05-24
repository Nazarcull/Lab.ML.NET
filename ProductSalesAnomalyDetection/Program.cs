using Microsoft.ML;
using ProductSalesAnomalyDetection;

// ── Paths & constants ──────────────────────────────────────────────────────
string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "product-sales.csv");
const int _docsize = 36;

// ── Initialize MLContext ───────────────────────────────────────────────────
MLContext mlContext = new MLContext();

// ── Load data ──────────────────────────────────────────────────────────────
IDataView dataView = mlContext.Data.LoadFromTextFile<ProductSalesData>(
    path: _dataPath,
    hasHeader: true,
    separatorChar: ',');

// ── Run detections ─────────────────────────────────────────────────────────
DetectSpike(mlContext, _docsize, dataView);
DetectChangepoint(mlContext, _docsize, dataView);

// ═══════════════════════════════════════════════════════════════════════════
// CreateEmptyDataView — produces an empty IDataView with the correct schema
// ═══════════════════════════════════════════════════════════════════════════
IDataView CreateEmptyDataView(MLContext mlContext)
{
    IEnumerable<ProductSalesData> enumerableData = new List<ProductSalesData>();
    return mlContext.Data.LoadFromEnumerable(enumerableData);
}

// ═══════════════════════════════════════════════════════════════════════════
// DetectSpike — IID Spike Detection
// ═══════════════════════════════════════════════════════════════════════════
void DetectSpike(MLContext mlContext, int docSize, IDataView productSales)
{
    Console.WriteLine("=============== Training the model ===============");
    Console.WriteLine("=============== End of training process ===============");

    var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(
        outputColumnName: nameof(ProductSalesPrediction.Prediction),
        inputColumnName:  nameof(ProductSalesData.numSales),
        confidence:           95d,
        pvalueHistoryLength:  docSize / 4);

    ITransformer iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyDataView(mlContext));

    IDataView transformedData = iidSpikeTransform.Transform(productSales);

    var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(
        transformedData, reuseRowObject: false);

    Console.WriteLine("Alert\tScore\tP-Value");

    foreach (var p in predictions)
    {
        if (p.Prediction is not null)
        {
            var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}";

            if (p.Prediction[0] == 1)
            {
                results += " <-- Spike detected";
            }

            Console.WriteLine(results);
        }
    }

    Console.WriteLine("");
}

// ═══════════════════════════════════════════════════════════════════════════
// DetectChangepoint — IID Change Point Detection
// ═══════════════════════════════════════════════════════════════════════════
void DetectChangepoint(MLContext mlContext, int docSize, IDataView productSales)
{
    Console.WriteLine("=============== Training the model Using Change Point Detection Algorithm===============");
    Console.WriteLine("=============== End of training process ===============");

    var iidChangePointEstimator = mlContext.Transforms.DetectIidChangePoint(
        outputColumnName:    nameof(ProductSalesPrediction.Prediction),
        inputColumnName:     nameof(ProductSalesData.numSales),
        confidence:          95d,
        changeHistoryLength: docSize / 4);

    var iidChangePointTransform = iidChangePointEstimator.Fit(CreateEmptyDataView(mlContext));

    IDataView transformedData = iidChangePointTransform.Transform(productSales);

    var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(
        transformedData, reuseRowObject: false);

    Console.WriteLine("Alert\tScore\tP-Value\tMartingale value");

    foreach (var p in predictions)
    {
        if (p.Prediction is not null)
        {
            var results =
                $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}\t{p.Prediction[3]:F2}";

            if (p.Prediction[0] == 1)
            {
                results += " <-- alert is on, predicted changepoint";
            }

            Console.WriteLine(results);
        }
    }

    Console.WriteLine("");
}
