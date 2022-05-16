using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Werner1995 {
  public class Model {
    public int[,] Lattice;
    public int WidthAcross = 0;    // across wind
    public int LengthDownwind = 0;
    public int HopLength = 5;
    public double pSand = 0.6;
    public double pNoSand = 0.4;
    public int ticks = 0;
    private int[,] flux;
    private int[,] upslopeNeighbourOffset =
        new int[8, 2] { { -1, 0 }, { -1, -1 }, { -1, 1 }, { 0, -1 }, { 0, 1 }, { 1, -1 }, { 1, 0 }, { 1, 1 } };
    private int[,] downslopeNeighbourOffset =
        new int[8, 2] { { 1, 0 }, { 1, -1 }, { 1, 1 }, { 0, -1 }, { 0, 1 }, { -1, -1 }, { -1, 0 }, { -1, 1 } };
    private Random rnd = new Random(123);

    public Model(int WidthAcross, int LengthDownwind) {
      this.WidthAcross = WidthAcross;
      this.LengthDownwind = LengthDownwind;
      Lattice = new int[LengthDownwind, WidthAcross];
      for (int i = 0; i < LengthDownwind; i++)
        for (int j = 0; j < WidthAcross; j++)
          Lattice[i, j] = 0;
      flux = new int[LengthDownwind, WidthAcross];
    }

    public void InitRandom(int AverageSandDepth) {
      for (int i = AverageSandDepth * LengthDownwind * WidthAcross; i > 0; i--)
        Lattice[rnd.Next(0, LengthDownwind), rnd.Next(0, WidthAcross)]++;
    }

    public void InitUniform(int SandDepth) {
      for (int i = 0; i < LengthDownwind; i++)
        for (int j = 0; j < WidthAcross; j++)
          Lattice[i, j] = SandDepth;
    }

    public void InitSquare(int SandDepth) {
      InitUniform(0);
      for (int i = LengthDownwind / 4; i < 3 * LengthDownwind / 4; i++)
        for (int j = 0; j < WidthAcross; j++)
          Lattice[i, j] = SandDepth;
    }

    public void InitLinear(int MaxDepth) {
      for (int i = 0; i < LengthDownwind; i++)
        for (int j = 0; j < WidthAcross; j++)
          Lattice[i, j] = i * MaxDepth / LengthDownwind;
    }

    public int MaxValue() {
      int maxV = 0;
      for (int i = 0; i < LengthDownwind; i++)
        for (int j = 0; j < WidthAcross; j++)
          if (Lattice[i, j] > maxV)
            maxV = Lattice[i, j];
      return maxV;
    }

    public int Count() {
      int sum = 0;
      for (int i = 0; i < LengthDownwind; i++)
        for (int j = 0; j < WidthAcross; j++)
          sum += Lattice[i, j];
      return sum;
    }

    private int wrapCheck(int i, int maximum) {
      if (i >= maximum)
        i -= maximum;
      else if (i < 0)
        i += maximum;
      return i;
    }

    private int recedeUpslope(int iCenter, int jCenter, out int iSteep, out int jSteep) {
      // find iSteep,jSteep and the max level of neighbours minus center
      int maxHeight = 0;
      iSteep = 0;
      jSteep = 0;
      for (int n = 0; n < 8; n++) {
        int i = wrapCheck(iCenter + upslopeNeighbourOffset[n, 0], LengthDownwind);
        int j = wrapCheck(jCenter + upslopeNeighbourOffset[n, 1], WidthAcross);
        if (Lattice[i, j] > maxHeight) {
          iSteep = i;
          jSteep = j;
          maxHeight = Lattice[i, j];
        }
      }
      return maxHeight - Lattice[iCenter, jCenter];
    }

    private int tumbleDownslope(int iCenter, int jCenter, out int iSteep, out int jSteep) {
      int minHeight = 1000000000;
      iSteep = 0;
      jSteep = 0;
      for (int n = 0; n < 8; n++) {
        int i = wrapCheck(iCenter + downslopeNeighbourOffset[n, 0], LengthDownwind);
        int j = wrapCheck(jCenter + downslopeNeighbourOffset[n, 1], WidthAcross);
        if (Lattice[i, j] < minHeight) {
          iSteep = i;
          jSteep = j;
          minHeight = Lattice[i, j];
        }
      }
      return Lattice[iCenter, jCenter] - minHeight;
    }

    private void erodeGrain(int i, int j) {
      Lattice[i, j]--;
      int iSteep, jSteep;
      while (recedeUpslope(i, j, out iSteep, out jSteep) > 1) {
        Lattice[i, j]++;
        Lattice[iSteep, jSteep]--;
        i = iSteep;
        j = jSteep;
      }
    }

    private void depositGrain(int i, int j) {
      Lattice[i, j]++;
      int iSteep, jSteep;
      while (tumbleDownslope(i, j, out iSteep, out jSteep) > 1) {
        Lattice[i, j]--;
        Lattice[iSteep, jSteep]++;
        i = iSteep;
        j = jSteep;
      }
    }

    private bool inShadow(int i, int j) {
      int baseDepth = Lattice[i, j];
      int iUpwind = i;
      for (int n = 0; n <= HopLength; n++) {
        if (--iUpwind < 0)
          iUpwind = LengthDownwind - 1;
        if (Lattice[iUpwind, j] > (baseDepth + n))
          return true;
      }
      return false;
    }

    private double pHighFlux(int i, int j) {
      int sum = 0;
      int iUpwind = i;
      for (int n = 0; n < HopLength; n++) {
        if (--iUpwind < 0)
          iUpwind = LengthDownwind - 1;
        sum += flux[iUpwind, j];
      }
      return 0.0; //  Math.Min(0.8, ((double)sum) / 5);
    }

    public void Tick(int cycles) {
      ticks += cycles;
      for (int i = 0; i < LengthDownwind; i++)
        for (int j = 0; j < WidthAcross; j++)
          flux[i, j] = 0;
      for (int n = 0; n < cycles; n++) {
        int i = rnd.Next(0, LengthDownwind);
        int j = rnd.Next(0, WidthAcross);
        if (Lattice[i, j] == 0) continue;
        if (inShadow(i, j)) continue;
        if (rnd.NextDouble() < pHighFlux(i, j)) continue;
        erodeGrain(i, j);
        while (true) {
          i += HopLength;
          if (i >= LengthDownwind)
            i -= LengthDownwind;
          flux[i, j]++;
          if (inShadow(i, j) ||
              (rnd.NextDouble() < ((Lattice[i, j] > 0 ? pSand : pNoSand) * (1.0 - pHighFlux(i, j))))) {
            depositGrain(i, j);
            break;
          }
        }
      }
    }

  }
}



      //int maxHeight = Lattice[--iCenter, jCenter];  // offset after this: -1, 0
      //iSteep = iCenter; jSteep = jCenter;
      //if (Lattice[iCenter, jCenter - 1] > maxHeight) {
      //  maxHeight = Lattice[iCenter, jCenter - 1];
      //  iSteep = iCenter; jSteep = jCenter - 1;
      //}
      //if (Lattice[iCenter, ++jCenter] > maxHeight) {  // offset after this: -1, 1
      //  maxHeight = Lattice[iCenter, jCenter];
      //  iSteep = iCenter; jSteep = jCenter;
      //}
      //if (Lattice[++iCenter, jCenter] > maxHeight) {  // offset after this:  0, 1
      //  maxHeight = Lattice[iCenter, jCenter];
      //  iSteep = iCenter; jSteep = jCenter;
      //}
      //if (Lattice[iCenter, jCenter - 2] > maxHeight) {
      //  maxHeight = Lattice[iCenter, jCenter - 2];
      //  iSteep = iCenter; jSteep = jCenter - 2;
      //}
      //if (Lattice[++iCenter, jCenter] > maxHeight) {  // offset after this:  1, 1
      //  maxHeight = Lattice[iCenter, jCenter];
      //  iSteep = iCenter; jSteep = jCenter;
      //}
      //if (Lattice[iCenter, --jCenter] > maxHeight) {  // offset after this:  1, 0
      //  maxHeight = Lattice[iCenter, jCenter];
      //  iSteep = iCenter; jSteep = jCenter;
      //}
      //if (Lattice[iCenter, jCenter - 1] > maxHeight) {
      //  maxHeight = Lattice[iCenter, jCenter - 1];
      //  iSteep = iCenter; jSteep = jCenter - 1;
      //}
      //return maxHeight - Lattice[iCenter - 1, jCenter];
