using Microsoft.ML;
using GitHubIssueClassification;

// ── Paths ──────────────────────────────────────────────────────────────────
string _appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]) ?? ".";
string _trainDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
string _testDataPath  = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
string _modelPath     = Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

// ── Globals ────────────────────────────────────────────────────────────────
MLContext _mlContext;
PredictionEngine<GitHubIssue, IssuePrediction> _predEngine = null!;
ITransformer _trainedModel = null!;
IDataView _trainingDataView = null!;

// ── Entry point ────────────────────────────────────────────────────────────
_mlContext = new MLContext(seed: 0);

_trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);

var pipeline         = ProcessData();
var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);

Evaluate(_trainingDataView.Schema);

PredictIssue();

// ═══════════════════════════════════════════════════════════════════════════
// ProcessData — extract features and transform the data
// ═══════════════════════════════════════════════════════════════════════════
IEstimator<ITransformer> ProcessData()
{
    var pipeline = _mlContext.Transforms.Conversion
        .MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
        .Append(_mlContext.Transforms.Text.FeaturizeText(
            inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(
            inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
        .Append(_mlContext.Transforms.Concatenate(
            "Features", "TitleFeaturized", "DescriptionFeaturized"))
        .AppendCacheCheckpoint(_mlContext);

    return pipeline;
}

// ═══════════════════════════════════════════════════════════════════════════
// BuildAndTrainModel — build pipeline, train, and do a quick prediction
// ═══════════════════════════════════════════════════════════════════════════
IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView,
                                            IEstimator<ITransformer> pipeline)
{
    var trainingPipeline = pipeline
        .Append(_mlContext.MulticlassClassification.Trainers
            .SdcaMaximumEntropy("Label", "Features"))
        .Append(_mlContext.Transforms.Conversion
            .MapKeyToValue("PredictedLabel"));

    _trainedModel = trainingPipeline.Fit(trainingDataView);

    _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);

    GitHubIssue issue = new GitHubIssue()
    {
        Title       = "WebSockets communication is slow in my machine",
        Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
    };

    var prediction = _predEngine.Predict(issue);

    Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");

    return trainingPipeline;
}

// ═══════════════════════════════════════════════════════════════════════════
// Evaluate — load test data, compute metrics, save model
// ═══════════════════════════════════════════════════════════════════════════
void Evaluate(Microsoft.ML.DataViewSchema trainingDataViewSchema)
{
    var testDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_testDataPath, hasHeader: true);

    var testMetrics = _mlContext.MulticlassClassification.Evaluate(
        _trainedModel.Transform(testDataView));

    Console.WriteLine($"*******************************************************************************");
    Console.WriteLine($"* Metrics for Multi-class Classification model - Test Data                    ");
    Console.WriteLine($"*------------------------------------------------------------------------------");
    Console.WriteLine($"* MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
    Console.WriteLine($"* MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
    Console.WriteLine($"* LogLoss:          {testMetrics.LogLoss:#.###}");
    Console.WriteLine($"* LogLossReduction: {testMetrics.LogLossReduction:#.###}");
    Console.WriteLine($"*******************************************************************************");

    SaveModelAsFile(_mlContext, trainingDataViewSchema, _trainedModel);
}

// ═══════════════════════════════════════════════════════════════════════════
// PredictIssue — load saved model and predict a new issue
// ═══════════════════════════════════════════════════════════════════════════
void PredictIssue()
{
    ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

    GitHubIssue singleIssue = new GitHubIssue()
    {
        Title       = "Entity Framework crashes",
        Description = "When connecting to the database, EF is crashing"
    };

    _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);

    var prediction = _predEngine.Predict(singleIssue);

    Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
}

// ═══════════════════════════════════════════════════════════════════════════
// SaveModelAsFile — serialize the trained model to a .zip file
// ═══════════════════════════════════════════════════════════════════════════
void SaveModelAsFile(MLContext mlContext, Microsoft.ML.DataViewSchema trainingDataViewSchema,
                     ITransformer model)
{
    // Ensure Models directory exists
    Directory.CreateDirectory(Path.GetDirectoryName(_modelPath)!);

    mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);

    Console.WriteLine($"The model is saved to {_modelPath}");
}
