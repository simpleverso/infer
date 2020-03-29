using Microsoft.ML;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;

namespace ClinicalTrial {

  class Program {
    static void Main(string[] args) {
      // Randomize patients equally to treatment or control
      // Measure their outcome (true for good or false for bad)
      VariableArray<bool> controlGroup = Variable.Observed(
        new bool[] { false, false, false, false, false, true, false, true, true, false }
      );
      VariableArray<bool> treatmentGroup = Variable.Observed(
        new bool[] { true, false, true, true, true, false, true, true, true, true }
      );
      Range i = controlGroup.Range; 
      Range j = treatmentGroup.Range;
      // Assume 10 % chance that treatment is effective
      Variable<bool> isEffective = Variable.Bernoulli(0.25);

      Variable<double> probIfTreated, probIfControl;
      // If treatment is effective, groups have different chances of good outcome
      // Patients within the same group have the same chance of good outcome
      using (Variable.If(isEffective)) {
        probIfControl = Variable.Beta(1, 1);
        probIfTreated = Variable.Beta(1, 1);
        controlGroup[i] = Variable.Bernoulli(probIfControl).ForEach(i);  
        treatmentGroup[j] = Variable.Bernoulli(probIfTreated).ForEach(j);  
      }
      // If treatment is not effective, everyone has the same chance of good outcome
      using (Variable.IfNot(isEffective)) {
        Variable<double> probAll = Variable.Beta(1, 1);
        controlGroup[i] = Variable.Bernoulli(probAll).ForEach(i);  
        treatmentGroup[j] = Variable.Bernoulli(probAll).ForEach(j);  
      }
      // Initialize inference engine
      InferenceEngine ie = new InferenceEngine();
      // Infer parameters of interest
      string effect = ie.Infer(isEffective).ToString();
      string treated = ie.Infer(probIfTreated).ToString();
      string control = ie.Infer(probIfControl).ToString();
      // Print out results 
      System.Console.WriteLine("P(effect | data) = " + effect);
      System.Console.WriteLine("P(good | treatment) = " + treated);  
      System.Console.WriteLine("P(good | control) = " + control);
    }
  }
}
