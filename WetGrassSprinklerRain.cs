using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;


namespace BayesInferCore
{
    public class WetGrassSprinklerRain
    {
        public void Run()
        {
            // Set random seed for repeatable example
            Rand.Restart(12347);

            // Create a new model
            WetGlassSprinklerRainModel model = new WetGlassSprinklerRainModel();
            if (model.Engine.Algorithm is Microsoft.ML.Probabilistic.Algorithms.GibbsSampling)
            {
                Console.WriteLine("This example does not run with Gibbs Sampling");
                return;
            }
            // Neste exemplo, cada variável leva apenas dois estados - verdadeiro (índice 0) e falso (índice 1)
            // O exemplo é escrito de forma que as extensões para problemas de mais de dois estados sejam simples.

            // -------------------------------------------------------------
            // Usage 1: Query the model when we know the parameters exactly
            // -------------------------------------------------------------
            Console.WriteLine("\n*********************************************");
            Console.WriteLine("Querying the model with a known ground truth");
            Console.WriteLine("*********************************************");

            Vector probCloudy = Vector.FromArray(0.5, 0.5);
            Vector[] cptSprinkler = new Vector[] { Vector.FromArray(0.1, 0.9) /* cloudy */, Vector.FromArray(0.5, 0.5) /* not cloudy */ };
            Vector[] cptRain = new Vector[] { Vector.FromArray(0.8, 0.2) /* cloudy */, Vector.FromArray(0.2, 0.8) /* not cloudy */ };
            Vector[][] cptWetGrass = new Vector[][]
            {
                new Vector[] { Vector.FromArray(0.99, 0.01) /* rain */,  Vector.FromArray(0.9, 0.1) /* not rain */ }, // Sprinkler
                new Vector[] { Vector.FromArray(0.9, 0.1) /* rain */, Vector.FromArray(0.0, 1.0) /* not rain */ }  // Not sprinkler
            };

            double probRainGivenWetGrass = model.ProbRain(null, null, 0, probCloudy, cptSprinkler, cptRain, cptWetGrass);
            double probRainGivenWetGrassNotCloudy = model.ProbRain(1, null, 0, probCloudy, cptSprinkler, cptRain, cptWetGrass);

            Console.WriteLine("P(rain | grass is wet)              = {0:0.0000}", probRainGivenWetGrass);
            Console.WriteLine("P(rain | grass is wet, not cloudy ) = {0:0.0000}", probRainGivenWetGrassNotCloudy);

            // -------------------------------------------------------------
            // Usage 2: Learn posterior distributions for the parameters
            // -------------------------------------------------------------
            // Para validar o aprendizado, primeiro amostra de um modelo conhecido
            int[][] sample = WetGlassSprinklerRainModel.Sample(1000, probCloudy, cptSprinkler, cptRain, cptWetGrass);

            Console.WriteLine("\n*********************************************");
            Console.WriteLine("Learning parameters from data (uniform prior)");
            Console.WriteLine("*********************************************");

            // Agora veja se podemos recuperar os parâmetros a partir dos dados - assumam antecedentes uniformes
            model.LearnParameters(sample[0], sample[1], sample[2], sample[3]);

            // Os posteriores são distribuições sobre as probabilidades e CPTs.
            // Imprima os meios dessas distribuições e compare com a verdade básica
            Console.WriteLine("Prob. Cloudy:                              Ground truth: {0:0.00}, Inferred: {1:0.00}", 0.5, model.ProbCloudyPosterior.GetMean()[0]);
            Console.WriteLine("Prob. Sprinkler | Cloudy:                  Ground truth: {0:0.00}, Inferred: {1:0.00}", 0.1, model.CPTSprinklerPosterior[0].GetMean()[0]);
            Console.WriteLine("Prob. Sprinkler | Not Cloudy:              Ground truth: {0:0.00}, Inferred: {1:0.00}", 0.5, model.CPTSprinklerPosterior[1].GetMean()[0]);
            Console.WriteLine("Prob. Rain      | Cloudy:                  Ground truth: {0:0.00}, Inferred: {1:0.00}", 0.8, model.CPTRainPosterior[0].GetMean()[0]);
            Console.WriteLine("Prob. Rain      | Not Cloudy:              Ground truth: {0:0.00}, Inferred: {1:0.00}", 0.2, model.CPTRainPosterior[1].GetMean()[0]);
            Console.WriteLine("Prob. Wet Grass | Sprinkler, Rain:         Ground truth: {0:0.00}, Inferred: {1:0.00}", 0.99, model.CPTWetGrassPosterior[0][0].GetMean()[0]);
            Console.WriteLine("Prob. Wet Grass | Sprinkler, Not Rain      Ground truth: {0:0.00}, Inferred: {1:0.00}", 0.9, model.CPTWetGrassPosterior[0][1].GetMean()[0]);
            Console.WriteLine("Prob. Wet Grass | Not Sprinkler, Rain:     Ground truth: {0:0.00}, Inferred: {1:0.00}", 0.9, model.CPTWetGrassPosterior[1][0].GetMean()[0]);
            Console.WriteLine("Prob. Wet Grass | Not Sprinkler, Not Rain: Ground truth: {0:0.00}, Inferred: {1:0.00}", 0.0, model.CPTWetGrassPosterior[1][1].GetMean()[0]);

            // -------------------------------------------------------------
            // Usage 3: Consultando o modelo levando em conta a incerteza dos parâmetros
            //   
            // -------------------------------------------------------------
            // Use posteriores que acabamos de aprender
            Console.WriteLine("\n**********************************************");
            Console.WriteLine("Querying the model with uncertain ground truth");
            Console.WriteLine("**********************************************");
            double probRainGivenWetGrass1 = model.ProbRain(null, null, 0, model.ProbCloudyPosterior, model.CPTSprinklerPosterior, model.CPTRainPosterior, model.CPTWetGrassPosterior);
            double probRainGivenWetGrassNotCloudy1 = model.ProbRain(1, null, 0, model.ProbCloudyPosterior, model.CPTSprinklerPosterior, model.CPTRainPosterior, model.CPTWetGrassPosterior);
            Console.WriteLine("P(rain | grass is wet)              = {0:0.0000}", probRainGivenWetGrass1);
            Console.WriteLine("P(rain | grass is wet, not cloudy ) = {0:0.0000}", probRainGivenWetGrassNotCloudy1);
            Console.WriteLine("");
        }
    }
}
