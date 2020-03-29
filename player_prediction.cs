﻿//https://docs.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/matchup-app-infer-net
using System;
using System.Linq;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;

namespace player
{
    class Program
    {
        static void Main(string[] args)
        {
            // the winner and loser in each of 6 games
            var winnerData = new[] { 0, 0, 0, 1, 3, 4 };
            var loserData = new[] { 1, 3, 4, 2, 1, 2 };
            // Define the statistical model as a probabilistic program
            var game = new Range(winnerData.Length);
            var player = new Range(winnerData.Concat(loserData).Max() + 1);
            var playerSkills = Variable.Array<double>(player);
            playerSkills[player] = Variable.GaussianFromMeanAndVariance(6, 9).ForEach(player);

            var winners = Variable.Array<int>(game);
            var losers = Variable.Array<int>(game);

            using (Variable.ForEach(game))
            {
                // the player performance is noisy version of their skill
                var winnerPerformance = Variable.GaussianFromMeanAndVariance(playerSkills[winners[game]], 1.0);
                var loserPerformance = Variable.GaussianFromMeanAndVariance(playerSkills[losers[game]], 1.0);
                // The winner performed better in this game
                Variable.ConstrainTrue(winnerPerformance > loserPerformance);
            }
            // attach data to model
            winners.ObservedValue = winnerData;
            losers.ObservedValue = loserData;
            // runing inference
            var inferenceEngine = new InferenceEngine();
            var inferredSkills = inferenceEngine.Infer<Gaussian[]>(playerSkills);

            // the inferred skills are uncertain, which is captured in their variance
            var orderedPlayerSkills = inferredSkills
            .Select((s, i) => new { Player = i, Skill = s })
            .OrderByDescending(ps => ps.Skill.GetMean());

            foreach (var playerSkill in orderedPlayerSkills)
            {
                Console.WriteLine($"Player {playerSkill.Player} skill: {playerSkill.Skill}");
            }

        }
    }
}
