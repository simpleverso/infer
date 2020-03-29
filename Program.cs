/*
TODO
[] Update priors with posteriors on each iteration
[x] Implement Greedy sampling
[x] Implement Thompson sampling
[x] Normalize Bag of Words values from training
[x] Use bag of words values for cold start initialization
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic;
using Newtonsoft.Json.Linq;
namespace thesis
{
    class Program
    {

        // static void Main(string[] args)
        // {
        //     CyclingTime2 test = new CyclingTime2();
        //     test.RunCyclingTime2();
        //     // test.RunTest();
        // }


    }

    public class CyclingTime2
    {
        public void RunCyclingTime2()
        {
            double[][] trainingDataModel = new double[5][];
            trainingDataModel[0] = new double[5] {0.13, 0.17, 0.16, 0.12, 0.13};
            trainingDataModel[1] = new double[5] {0.13, 0.17, 0.16, 0.12, 0.13};
            trainingDataModel[2] = new double[5] {0.13, 0.17, 0.16, 0.12, 0.13};
            trainingDataModel[3] = new double[5] {0.13, 0.17, 0.16, 0.12, 0.13};
            trainingDataModel[4] = new double[5] {0.13, 0.17, 0.16, 0.12, 0.13};

            // trainingDataModel[0] = new double[5] {0.13, 0.17, 0.16, 0.12, 0.13};
            // trainingDataModel[1] = new double[5] {0.13, 0.0, 0.16, 0.12, 0.13};
            // trainingDataModel[2] = new double[5] {0.13, 0.0, 0.16, 0.12, 0.0};
            // trainingDataModel[3] = new double[5] {0.13, 0.0, 0.16, 0.0, 0.0};
            // trainingDataModel[4] = new double[5] {0.0, 0.0, 0.16, 0.0, 0.0};

            ModelData initPriors = new ModelData(
                Gaussian.FromMeanAndPrecision(1.0, 0.01),
                Util.ArrayInit(5, _ => Gaussian.FromMeanAndPrecision(1.0, 0.01))
                );

            // Train the model
            CyclistTraining cyclistTraining = new CyclistTraining();
            cyclistTraining.CreateModel();
            cyclistTraining.SetModelData(initPriors);
            for(int i = 0; i < 5; ++i)
            {   
                ModelData posteriors1 = cyclistTraining.InferModelData(trainingDataModel[i]);
                Console.WriteLine("Average travel time = " + posteriors1.AverageTimeDist);
                Console.WriteLine("Average travel time array = {0} {1} {2} {3} {4}", 
                    posteriors1.AverageTimeDistArray[0],
                    posteriors1.AverageTimeDistArray[1],
                    posteriors1.AverageTimeDistArray[2],
                    posteriors1.AverageTimeDistArray[3],
                    posteriors1.AverageTimeDistArray[4]);

                cyclistTraining.SetModelData(posteriors1);
            }
        }
        public class CyclistBase
        {
            public InferenceEngine InferenceEngine;

            protected Variable<double> AverageTime;
            protected Variable<Gaussian> AverageTimePrior;

            protected VariableArray<double> AverageTimeArray;
            protected VariableArray<Gaussian> AverageTimePriorArray;

            public virtual void CreateModel()
            {
                AverageTimePrior = Variable.New<Gaussian>();
                AverageTime = Variable<double>.Random(AverageTimePrior);

                Range numAverages = new Range(5);
                AverageTimePriorArray = Variable.Array<Gaussian>(numAverages);
                AverageTimeArray = Variable.Array<double>(numAverages);
                AverageTimeArray[numAverages] = Variable<double>.Random(AverageTimePriorArray[numAverages]);

                if (InferenceEngine == null)
                {
                    InferenceEngine = new InferenceEngine();
                }
            }

            public virtual void SetModelData(ModelData priors)
            {
                AverageTimePrior.ObservedValue = priors.AverageTimeDist;
                AverageTimePriorArray.ObservedValue = priors.AverageTimeDistArray;
            }
        }

        public class CyclistTraining : CyclistBase
        {
            protected VariableArray<double> TravelTimes;
            protected Variable<int> NumTrips;

            protected VariableArray2D<double> TravelTimesArray;
            protected VariableArray<int> NumTripsArray;

            public override void CreateModel()
            {
                base.CreateModel();
                NumTrips = Variable.New<int>();
                Range tripRange = new Range(NumTrips);
                TravelTimes = Variable.Array<double>(tripRange);
                using (Variable.ForEach(tripRange))
                {
                    TravelTimes[tripRange] = Variable.GaussianFromMeanAndPrecision(AverageTime, 0.01);
                }

                // NumTripsArray = Variable.Array<int>(NumTrips); ///////////////// Working here 
            }

            public ModelData InferModelData(double[] trainingData)
            {
                ModelData posteriors;

                NumTrips.ObservedValue = trainingData.Length;
                TravelTimes.ObservedValue = trainingData;

                posteriors.AverageTimeDist = InferenceEngine.Infer<Gaussian>(AverageTime);
                posteriors.AverageTimeDistArray = InferenceEngine.Infer<Gaussian[]>(AverageTimeArray);
                return posteriors;
            }
        }

        public struct ModelData
        {
            // public Gaussian[] AverageTestDist;
            public Gaussian AverageTimeDist;
            public Gaussian[] AverageTimeDistArray;

            public ModelData(Gaussian mean, Gaussian[] meanArray)
            {
                AverageTimeDist = mean;
                AverageTimeDistArray = meanArray;
            }
        }
    }
   
    


}
