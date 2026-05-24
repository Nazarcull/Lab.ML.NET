using Microsoft.ML.Data;

namespace ProductSalesAnomalyDetection;

public class ProductSalesData
{
    [LoadColumn(0)]
    public string? Month;

    [LoadColumn(1)]
    public float numSales;
}

public class ProductSalesPrediction
{
    // vector to hold alert, score, p-value values (spike)
    // or alert, score, p-value, martingale value (change point)
    [VectorType(4)]
    public double[]? Prediction { get; set; }
}
