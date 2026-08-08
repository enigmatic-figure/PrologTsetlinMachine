using System;
using System.IO;
using System.Collections.Generic;

// adapted from TsetlinMachine.pyx (Pyrex)
// at github.com/cair/TsetlinMachine/tree/master

namespace TsetlinMachineBinary
{
  internal class TsetlinMachineBinaryProgram
  {
    static void Main(string[] args)
    {
      Console.WriteLine("\nBegin Tsetlin Machine binary" +
        " classification demo ");

      // 1. Load data
      Console.WriteLine("\nLoading Iris binarized-features" +
          " two-class train (80) and test (20) from file ");
      string trainFile =
        "..\\..\\..\\Data\\iris_two_classes_train_80.txt";
      int[][] trainX = MatUtils.MatLoad(trainFile,
        new int[] { 0, 1, 2, 3, 4, 5,6, 7, 8, 9, 10, 11,
          12, 13, 14, 15 }, ',', "#");
      int[] trainY =
        MatUtils.MatToVec(MatUtils.MatLoad(trainFile,
        new int[] { 16 }, ',', "#"));

      string testFile =
        "..\\..\\..\\Data\\iris_two_classes_test_20.txt";
      int[][] testX = MatUtils.MatLoad(testFile,
        new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
          12, 13, 14, 15 }, ',', "#");
      int[] testY =
        MatUtils.MatToVec(MatUtils.MatLoad(testFile,
        new int[] { 16 }, ',', "#"));
      Console.WriteLine("Done ");

      Console.WriteLine("\nFirst three train X items: ");
      for (int i = 0; i < 3; ++i)
        MatUtils.VecShow(trainX[i], 1);

      Console.WriteLine("\nFirst three y target" +
        " (0-1) values: ");
      for (int i = 0; i < 3; ++i)
        Console.WriteLine(trainY[i].ToString());

      // 2, create model
      int nClauses = 20; // number sort-of rules per class
      int nFeatures = 16;
      int nStates = 50;
      double s = 3.0;  // frequency updates using 1/s
      int threshold = 10;  // aka T clips voting
      int seed = 0;

      Console.WriteLine("\nSetting nClauses = " +
        nClauses);
      Console.WriteLine("Setting nFeatures = " +
        nFeatures);
      Console.WriteLine("Setting nStates = " +
        nStates);
      Console.WriteLine("Setting s (random update " +
        "inverse frequency) = " + s.ToString("F1"));

      Console.WriteLine("Setting threshold (voting " +
        "max/min clip) = " + threshold);
      Console.WriteLine("Creating Tsetlin Machine " +
        "binary classifier ");
      TsetlinMachine tm = new TsetlinMachine(nClauses,
        nFeatures, nStates, s, threshold, seed);
      Console.WriteLine("Done ");

      // 3. train model
      int maxEpochs = 100;
      Console.WriteLine("\nSetting training maxEpochs = " +
        maxEpochs);
      Console.WriteLine("Starting training ");
      tm.Train(trainX, trainY, maxEpochs);
      Console.WriteLine("Done ");

      //Console.WriteLine("\ntaState: ");
      //MatUtils.ShowCuboid(tm.taState, 3);
      //Console.ReadLine();

      // 4. evaluate model
      double trainAcc = tm.Accuracy(trainX, trainY);
      Console.WriteLine("\nAccuracy (train) = " +
        trainAcc.ToString("F4"));

      double testAcc = tm.Accuracy(testX, testY);
      Console.WriteLine("Accuracy (test) = " +
        testAcc.ToString("F4"));

      // 5. use model
      Console.WriteLine("\nPredicting class" +
        " for trainX[0] ");
      int pred_y = tm.Predict(trainX[0]);
      Console.WriteLine("Predicted y = " + pred_y);

      // 6. TODO: save model

      Console.WriteLine("\nEnd Tsetlin demo ");
      Console.ReadLine();
    } // Main()

  } // class Program

  // ========================================================

  public class TsetlinMachine
  {
    public int nClauses;
    public int nFeatures;
    public int nStates;
    public double s;
    public int threshold;
    private Random rnd;

    public int[][][] taState;
    public int[] clauseSign;
    public int[] clauseOutput;
    public int[] feedbackToClauses;

    public TsetlinMachine(int nClauses, int nFeatures,
      int nStates, double s, int threshold, int seed)
    {
      this.nClauses = nClauses;
      this.nFeatures = nFeatures;
      this.nStates = nStates;
      this.s = s;
      this.threshold = threshold;
      this.rnd = new Random(seed);

      this.taState =
        MatUtils.MakeCuboid(nClauses, nFeatures, 2);
      for (int i = 0; i < nClauses; ++i)
      {
        for (int j = 0; j < nFeatures; ++j)
        {
          double p1 = this.rnd.NextDouble();
          if (p1 < 0.50)
            this.taState[i][j][0] = nStates;
          else
            this.taState[i][j][0] = nStates + 1;

          double p2 = this.rnd.NextDouble();
          if (p2 < 0.50)
            this.taState[i][j][1] = nStates;
          else
            this.taState[i][j][1] = nStates + 1;
        }
      }

      this.clauseOutput = new int[nClauses];
      this.feedbackToClauses = new int[nClauses];
      this.clauseSign = new int[nClauses];  // sign is -1 or +1
      for (int j = 0; j < nClauses; ++j)
      {
        if (j % 2 == 0)
          this.clauseSign[j] = 1;
        else
          this.clauseSign[j] = -1;
      }

    } // ctor()

    // ------------------------------------------------------

    private void CalculateClauseOutput(int[] x)
    {
      // each cell clauseOutput is 0 or 1
      // but is modified by clauseSign
      for (int j = 0; j < this.nClauses; ++j)
      {
        this.clauseOutput[j] = 1;
        for (int k = 0; k < this.nFeatures; ++k)
        {
          int actionInclude;
          int actionExclude;
          if (this.taState[j][k][0] <= this.nStates)
            actionInclude = 0;
          else
            actionInclude = 1;
          if (this.taState[j][k][1] <= this.nStates)
            actionExclude = 0;
          else
            actionExclude = 1;

          if ((actionInclude == 1 && x[k] == 0) ||
            (actionExclude == 1 && x[k] == 1))
          {
            this.clauseOutput[j] = 0;
            break;
          }
        }
      }
    }

    // ------------------------------------------------------

    public int Predict(int[] x)
    {
      this.CalculateClauseOutput(x);
      int outputSum = this.SumClauseVotes();
      if (outputSum >= 0)
        return 1;
      else
        return 0;
    }

    // ------------------------------------------------------

    //private int Action(int state)
    //{
    //  if (state <= this.nStates)
    //    return 0;
    //  else
    //    return 1;
    //}

    // ------------------------------------------------------

    private int SumClauseVotes()
    {
      // result clipped between -thresh and +thresh
      int outputSum = 0;
      for (int j = 0; j < this.nClauses; ++j)
      {
        outputSum += this.clauseOutput[j] *
          this.clauseSign[j];
      }
      if (outputSum > this.threshold)
        outputSum = this.threshold;
      else if (outputSum < -this.threshold)
        outputSum = -this.threshold;
      return outputSum;
    }

    // ------------------------------------------------------

    public double Accuracy(int[][] X, int[] y)
    {
      int nCorrect = 0; int nWrong = 0;
      for (int i = 0; i < X.Length; ++i)
      {
        int[] xi = X[i];
        int actualY = y[i];
        int predY = this.Predict(xi);
        if (actualY == predY)
          ++nCorrect;
        else
          ++nWrong;
      }
      return (nCorrect * 1.0) / (nCorrect + nWrong);
    }

    // ------------------------------------------------------

    private void Update(int[] x, int y)
    {
      // helper for Train(). does most of the work.
      // updates this.taState
      this.CalculateClauseOutput(x);
      int outputSum = this.SumClauseVotes();
      for (int j = 0; j < this.nClauses; ++j)
        this.feedbackToClauses[j] = 0; // feedback is -1 or +1

      // step 1: compute feedback to clauses
      if (y == 1)
      {
        for (int j = 0; j < this.nClauses; ++j)
        {
          if (this.rnd.NextDouble() > 1.0 *
            (this.threshold - outputSum) /
            (2 * this.threshold))
            continue;
          if (this.clauseSign[j] == +1)
            this.feedbackToClauses[j] = 1; // Type I
          else
            this.feedbackToClauses[j] = -1; // Type II
        } // j

      }
      else if (y == 0)
      {
        for (int j = 0; j < this.nClauses; ++j)
        {
          if (this.rnd.NextDouble() > 1.0 *
            (this.threshold + outputSum) /
            (2 * this.threshold))
            continue;
          if (this.clauseSign[j] == +1)
            this.feedbackToClauses[j] = -1; // Type II
          else
            this.feedbackToClauses[j] = 1; // Type I
        } // j

      }

      // step 2: main processing loop over all clauses
      for (int j = 0; j < this.nClauses; ++j)
      {
        if (this.feedbackToClauses[j] == 1)
        {
          if (this.clauseOutput[j] == 0)
          {
            for (int k = 0; k < this.nFeatures; ++k)
            {
              if (this.rnd.NextDouble() <= 1.0 / this.s)
              {
                if (this.taState[j][k][0] > 1)
                  --this.taState[j][k][0];
              }
              if (this.rnd.NextDouble() <= 1.0 / this.s)
              {
                if (this.taState[j][k][1] > 1)
                  --this.taState[j][k][1];
              }
            } // k
          } // clauseOutput[j] == 0

          else if (this.clauseOutput[j] == 1)
          {
            for (int k = 0; k < this.nFeatures; ++k)
            {
              if (x[k] == 1)
              {
                if (this.rnd.NextDouble() <= 1.0 *
                  (this.s - 1) / this.s)
                {
                  if (this.taState[j][k][0] <
                    this.nStates * 2)
                    ++this.taState[j][k][0];
                }
                if (this.rnd.NextDouble() <= 1.0 / this.s)
                {
                  if (this.taState[j][k][1] > 1)
                    --this.taState[j][k][1];
                }
              }
              else if (x[k] == 0)
              {
                if (this.rnd.NextDouble() <= 1.0 *
                  (this.s - 1) / this.s)
                {
                  if (this.taState[j][k][1] <
                    this.nStates * 2)
                    ++this.taState[j][k][1];
                }
                if (this.rnd.NextDouble() <= 1.0 / this.s)
                {
                  if (this.taState[j][k][0] > 1)
                    --this.taState[j][k][0];
                }
              }
            } // k
          } // clauseOutput[j] == 1

        } // feedbackToClauses[j] > 0

        else if (this.feedbackToClauses[j] == -1)
        {
          if (this.clauseOutput[j] == 1)
          {
            for (int k = 0; k < this.nFeatures; ++k)
            {
              int actionInclude;
              int actionExclude;
              if (this.taState[j][k][0] <= this.nStates)
                actionInclude = 0;
              else
                actionInclude = 1;
              if (this.taState[j][k][1] <= this.nStates)
                actionExclude = 0;
              else
                actionExclude = 1;

              if (x[k] == 0)
              {
                if (actionInclude == 0 &&
                  this.taState[j][k][0] <
                  this.nStates * 2)
                  ++this.taState[j][k][0];
              }
              else if (x[k] == 1)
              {
                if (actionExclude == 0 &&
                  this.taState[j][k][1] <
                  this.nStates * 2)
                  ++this.taState[j][k][1];
              }
            } // k

          } // clauseOutput[j] == 1
        } // feedbackToClauses[j] < 0
      } // main loop j

      return;

    } // Update()

    // ------------------------------------------------------

    public void Train(int[][] X, int[] y, int maxEpochs)
    {
      int nExamples = X.Length;
      int[] xi = new int[this.nFeatures];
      int[] indices = new int[nExamples];
      for (int i = 0; i < nExamples; ++i)
        indices[i] = i;

      for (int epoch = 0; epoch < maxEpochs; ++epoch)
      {
        // if (epoch % 10 == 0) Console.Write(". ");
        if (epoch % 20 == 0)
        {
          double acc = this.Accuracy(X, y);
          string s1 = "Epoch " + epoch.ToString().PadLeft(4);
          string s2 = " |  Accuracy = " + acc.ToString("F4");
          Console.WriteLine(s1 + s2);
        }
        this.Shuffle(indices);
        for (int i = 0; i < nExamples; ++i)
        {
          int exampleID = indices[i];
          int targetY = y[exampleID];
          for (int j = 0; j < this.nFeatures; ++j)
          {
            xi[j] = X[exampleID][j];
            this.Update(xi, targetY);
          }
        }
      } // epoch

      return;
    } // Train()

    // ------------------------------------------------------

    private void Shuffle(int[] sequence)
    {
      // Fisher-Yates
      for (int i = 0; i < sequence.Length; ++i)
      {
        int ri = this.rnd.Next(i, sequence.Length);
        int tmp = sequence[ri];
        sequence[ri] = sequence[i];
        sequence[i] = tmp;
      }
    }

    // ------------------------------------------------------

  } // class TsetlinMachine

  // ========================================================

  public class MatUtils
  {
    // ------------------------------------------------------

    public static int[][] MatLoad(string fn,
      int[] usecols, char sep, string comment)
    {
      List<int[]> result = new List<int[]>();
      string line = "";
      FileStream ifs = new FileStream(fn, FileMode.Open);
      StreamReader sr = new StreamReader(ifs);
      while ((line = sr.ReadLine()) != null)
      {
        if (line.StartsWith(comment) == true)
          continue;
        string[] tokens = line.Split(sep);
        List<int> lst = new List<int>();
        for (int j = 0; j < usecols.Length; ++j)
          lst.Add(int.Parse(tokens[usecols[j]]));
        int[] row = lst.ToArray();
        result.Add(row);
      }
      sr.Close(); ifs.Close();
      return result.ToArray();
    }

    // ------------------------------------------------------

    public static int[] MatToVec(int[][] A)
    {
      int nRows = A.Length;
      int nCols = A[0].Length;
      int[] result = new int[nRows * nCols];
      int k = 0;
      for (int i = 0; i < nRows; ++i)
        for (int j = 0; j < nCols; ++j)
          result[k++] = A[i][j];
      return result;
    }

    // ------------------------------------------------------

    public static void VecShow(int[] vec, int wid)
    {
      for (int i = 0; i < vec.Length; ++i)
        Console.Write(vec[i].ToString().PadLeft(wid) + ", ");
      Console.WriteLine("");
    }

    // ------------------------------------------------------

    public static int[][][] MakeCuboid(int n1, int n2, int n3)
    {
      int[][][] result = new int[n1][][];
      for (int i = 0; i < n1; ++i)
      {
        result[i] = new int[n2][];
        for (int j = 0; j < n2; ++j)
        {
          result[i][j] = new int[n3];
        }
      }
      return result;
    }

    public static void ShowCuboid(int[][][] cuboid, int wid)
    {
      for (int i = 0; i < cuboid.Length; ++i)
      {
        for (int j = 0; j < cuboid[i].Length; ++j)
        {
          for (int k = 0; k < cuboid[i][j].Length; ++k)
          {
            Console.Write(cuboid[i][j][k].
              ToString().PadLeft(wid) + " ");
          }
          Console.WriteLine("");
        }
        Console.WriteLine("");
      }
    }

  } // class MatUtils

  // ========================================================

} // ns

/*
# iris_two_classes_train_80.txt
#
# sepal length (cols 0,1,2,3)
# sepal width (cols 4,5,6,7)
# petal length (cols 8,9,10,11)
# petal width (cols 12,13,14,15)
# species (col 16) setosa = 0, versicolor = 1
# (no virginica)
#
# raw source data: archive.ics.uci.edu/dataset/53/iris
# encoded data: github.com/cair/TsetlinMachine
#
# apparent feature encoding:
# [0.0, 1.5] = 0000
# [1.6, 3.1] = 0001
# [3.2, 4.7] = 0010
# [4.8, 6.3] = 0011
# [6.4, 7.9] = 0100
#
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1
0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1
0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
*/

/*
# iris_two_classes_test_20.txt
#
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1
0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1
*/

