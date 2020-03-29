using cyclingtime;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Algorithms;

public class CyclistMixedTraining : CyclistMixedBase
{
    protected Variable<int> NumTrips;
    protected VariableArray<double> TravelTimes;
    protected VariableArray<int> ComponentIndices;

    public override void CreateModel()
    {
        base.CreateModel();
        NumTrips = Variable.New<int>();
        Range tripRange = new Range(NumTrips);
        TravelTimes = Variable.Array<double>(tripRange);
        ComponentIndices = Variable.Array<int>(tripRange);
        using (Variable.ForEach(tripRange))
        {
            ComponentIndices[tripRange] =
            Variable.Discrete(MixingCoefficients);
            using (Variable.Switch(ComponentIndices[tripRange]))
            {
                TravelTimes[tripRange].SetTo(
                Variable.GaussianFromMeanAndPrecision(
                AverageTime[ComponentIndices[tripRange]],
                TrafficNoise[ComponentIndices[tripRange]])
                );
            }
        }
    }

    public ModelDataMixed InferModelData(double[] trainingData) //Training model
    {
        ModelDataMixed posteriors;
        NumTrips.ObservedValue = trainingData.Length;
        TravelTimes.ObservedValue = trainingData;
        posteriors.AverageTimeDist = InferenceEngine.Infer<Gaussian[]>(AverageTime);
        posteriors.TrafficNoiseDist = InferenceEngine.Infer<Gamma[]>(TrafficNoise);
        posteriors.MixingDist = InferenceEngine.Infer<Dirichlet>(MixingCoefficients);
        return posteriors;
    }
}