using System;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;

//
// The model estimates the (discrete) probability distribution over the number of balls in an urn given observations.
// The observations can be noisy, as governed by the "switchedColor" random variable. Set the Bernoulli parameter to 0 to have noiseless version.
//

namespace Counting
{
	class Program
	{
		static void Main(string[] args)
		{
			bool[] data = { true, true, true, true, true, true, true, true, true, true };
            //bool[] data = { true, true, true, true, true, false, false, false, false, false };

			// The probabilistic program
			// -------------------------

			// Variables describing the population
			int maxBalls = 8;
			Range ball = new Range(maxBalls+1); // so that numBalls = (0,...,maxBalls)
			Variable<int> numBalls = Variable.DiscreteUniform(ball);
			VariableArray<bool> isBlue = Variable.Array<bool>(ball);
            isBlue[ball] = Variable.Bernoulli(0.5).ForEach(ball);

            // Variables describing the observations
            Range draw = new Range(data.Length);
			VariableArray<bool> observedBlue = Variable.Array<bool>(draw);
            VariableArray<int> ballIndex = Variable.Array<int>(draw);
            using (Variable.ForEach(draw))
            {
                ballIndex[draw] = Variable.DiscreteUniform(numBalls);
                Variable<bool> switchedColor = Variable.Bernoulli(0.0);
                using (Variable<bool>.Switch(ballIndex[draw])) {
                    using (Variable.If(switchedColor))
                    {
                        observedBlue[draw] = !isBlue[ballIndex[draw]];
                    }
                    using (Variable.IfNot(switchedColor))
                    {
                        observedBlue[draw] = isBlue[ballIndex[draw]];
                    }
                }
            }

            observedBlue.ObservedValue = data;

            // Inference queries about the program
            // -----------------------------------
            InferenceEngine engine = new InferenceEngine();
            // ...add code here...
            Discrete numberOfBalls = engine.Infer<Discrete>( numBalls );

            Console.WriteLine("Distribution over number of balls: " + numberOfBalls);
			Console.WriteLine("Press any key...");
			Console.ReadKey();

			// Answer key
			// ----------
			// 10 blue, without noise:
			// numBalls = Discrete(0 0.5079 0.3097 0.09646 0.03907 0.02015 0.01225 0.008336 0.006133)
			// 10 blue, with 20% noise:
			// numBalls = Discrete(0 0.463 0.2354 0.1137 0.06589 0.04392 0.0322 0.02521 0.02068)
			// 10 blue, with 50% noise:
			// numBalls = Discrete(0 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125)
			// 5 blue/5 green, with 20% noise:
			// numBalls = Discrete(0 0.08198 0.09729 0.1102 0.1217 0.1324 0.1425 0.1523 0.1617)
		}
	}
}
