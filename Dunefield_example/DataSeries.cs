using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DunefieldModel {
  public class DataSeries {
    public int[] Data;

    public DataSeries(ref int[] Data) {
      this.Data = Data;
    }

    public int Max() {
      int max = Data[0];
      for (int i = 1; i < Data.Length; i++)
        if (Data[i] > max)
          max = Data[i];
      return max;
    }

    public int Min() {
      int min = Data[0];
      for (int i = 1; i < Data.Length; i++)
        if (Data[i] < min)
          min = Data[i];
      return min;
    }

    public Range GetRange() {
      return new Range(Min(), Max());
    }

  }
}
