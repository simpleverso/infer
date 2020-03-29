using cyclingtime;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Algorithms;

public class CyclistMixedPrediction : CyclistMixedBase
{
    private Gaussian TomorrowsTimeDist;
    private Variable<double> TomorrowsTime;

    public override void CreateModel()
    {
        base.CreateModel();
        Variable<int> componentIndex = Variable.Discrete(MixingCoefficients);
        TomorrowsTime = Variable.New<double>();
        using (Variable.Switch(componentIndex))
        {
            TomorrowsTime.SetTo(
            Variable.GaussianFromMeanAndPrecision(
            AverageTime[componentIndex],
            TrafficNoise[componentIndex]));
        }
    }

    public Gaussian InferTomorrowsTime()
    {
        TomorrowsTimeDist = InferenceEngine.Infer<Gaussian>(TomorrowsTime);
        return TomorrowsTimeDist;
    }
}