using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DunefieldModel {
  public class Model {
    public int[,] Elev;
    public float[,] Shadow;
    public int WidthAcross = 0;    // across wind
    public int LengthDownwind = 0;
    public int HopLength = 5;
    public float AverageHeight;
    public float pSand = 0.6f;
    public float pNoSand = 0.4f;
    public IFindSlope FindSlope;
    protected int mWidth, mLength;
    protected Random rnd = new Random(123);
    private Form1 parentForm;
    protected bool openEnded = false;
    public const float SHADOW_SLOPE = 0.803847577f;  //  3 * tan(15 degrees)

    public Model() {
    }

    public Model(Form1 ParentForm, IFindSlope SlopeFinder, int WidthAcross, int LengthDownwind) {
      parentForm = ParentForm;
      FindSlope = SlopeFinder;
      this.WidthAcross = (int)Math.Pow(2, (int)Math.Log(WidthAcross, 2));
      this.LengthDownwind = (int)Math.Pow(2, (int)Math.Log(LengthDownwind, 2));
      mWidth = this.WidthAcross - 1;
      mLength = this.LengthDownwind - 1;
      Elev = new int[WidthAcross, LengthDownwind];
      Array.Clear(Elev, 0, LengthDownwind * WidthAcross);
      Shadow = new float[WidthAcross, LengthDownwind];
      Array.Clear(Shadow, 0, LengthDownwind * WidthAcross);
      FindSlope.Init(ref Elev, WidthAcross, LengthDownwind);
      FindSlope.SetOpenEnded(openEnded);
    }

    public virtual bool UsesHopLength() {
      return true;  // does this model use the user-provided value of hop length?
    }

    public virtual bool UsesSandProbabilities() {
      return true;  // does this model use the user-provided values of sand depositing probabilities?
    }

    public void SetOpenEnded(bool NewState) {  // 'true' means dunefield is open-ended (no wrapping)
      openEnded = NewState;
      FindSlope.SetOpenEnded(openEnded);
    }

    public void InitRandom(int AverageSandDepth) {
      bool saveOpenEnded = openEnded;
      openEnded = false;
      for (int i = AverageSandDepth * LengthDownwind * WidthAcross; i > 0; i--)
        depositGrain(rnd.Next(0, WidthAcross), rnd.Next(0, LengthDownwind));
      openEnded = saveOpenEnded;
      shadowInit();
      AverageHeight = AveHeight();
      parentForm.ChartAxis_Height.ScaleRange = new Range(Math.Max(0, AverageSandDepth -5), AverageSandDepth + 10);
    }

    public void InitUniform(int SandDepth) {
      for (int x = 0; x < LengthDownwind; x++)
        for (int w = 0; w < WidthAcross; w++)
          Elev[w, x] = SandDepth; // -(int)((float)x * ((float)SandDepth / 2.0) / ((float)LengthDownwind)); ;
      shadowInit();
      AverageHeight = AveHeight();
      parentForm.ChartAxis_Height.ScaleRange = new Range(Math.Max(0, SandDepth - 5), SandDepth + 10);
    }

    public void InitSquare(int SandDepth) {
      for (int x = 0; x < LengthDownwind; x++)
        for (int w = 0; w < WidthAcross; w++)
          Elev[w, x] = ((w > WidthAcross / 8) && (w < WidthAcross / 2)) ? SandDepth : 1;

      //InitUniform(0);
      //for (int x = LengthDownwind / 10; x < LengthDownwind / 2; x++)
      //  for (int w = 0; w < WidthAcross; w++)
      //    Elev[w, x] = SandDepth;
      shadowInit();
      AverageHeight = AveHeight();
      parentForm.ChartAxis_Height.ScaleRange = new Range(Math.Max(0, SandDepth - 5), SandDepth + 10);
    }

    public void InitDune(int SandDepth, int Width) {
      InitUniform(SandDepth);
      int h = 0;
      int x = LengthDownwind / 4;
      for (int n = 0; n < 60; n++) {
        for (int w = (WidthAcross - Width) / 2; w < ((WidthAcross + Width) / 2); w++)
          Elev[w, x + n] += h / 2;
        h++;
      }
      h /= 2;
      for (int n = 60; n < 75; n++) {
        for (int w = (WidthAcross - Width) / 2; w < ((WidthAcross + Width) / 2); w++)
          Elev[w, x + n] += h;
        h -= 2;
      }
      shadowInit();
      AverageHeight = AveHeight();
      parentForm.ChartAxis_Height.ScaleRange = new Range(Math.Max(0, SandDepth - 5), SandDepth + 10);
    }

    public void InitLinear(int MaxDepth) {
      for (int x = 0; x < LengthDownwind; x++)
        for (int w = 0; w < WidthAcross; w++)
          Elev[w, x] = x * MaxDepth / LengthDownwind;
      shadowInit();
      AverageHeight = AveHeight();
      parentForm.ChartAxis_Height.ScaleRange = new Range(0, MaxDepth + 10);
    }

    public void InitCompletion() {
      shadowInit();
      AverageHeight = AveHeight();
      Range df = MinMaxHeight();
      parentForm.ChartAxis_Height.ScaleRange = new Range(Math.Max(0, df.Min - 5), df.Max + 10);
    }

    public int MaxHeight() {
      int maxH = 0;
      for (int x = 0; x < LengthDownwind; x++)
        for (int w = 0; w < WidthAcross; w++)
          if (Elev[w, x] > maxH)
            maxH = Elev[w, x];
      return maxH;
    }

    public Range MinMaxHeight() {
      int maxH = 0;
      int minH = 100000;
      for (int x = 0; x < LengthDownwind; x++)
        for (int w = 0; w < WidthAcross; w++) {
          if (Elev[w, x] > maxH)
            maxH = Elev[w, x];
          if (Elev[w, x] < minH)
            minH = Elev[w, x];
        }
      Range r = new Range();
      r.Min = minH;
      r.Max = maxH;
      return r;
    }

    public float AveHeight() {
      float sum = 0;
      for (int x = 0; x < LengthDownwind; x++)
        for (int w = 0; w < WidthAcross; w++)
          sum += Elev[w, x];
      return sum / ((float)(LengthDownwind) * (float)WidthAcross);
    }

    public int Count() {
      int sum = 0;
      for (int x = 0; x < LengthDownwind; x++)
        for (int w = 0; w < WidthAcross; w++)
          sum += Elev[w, x];
      return sum;
    }

    public int[] Profile(int WidthPosition) {
      int[] prof = new int[LengthDownwind];
      for (int x = 0; x < LengthDownwind; x++)
        prof[x] = Elev[WidthPosition, x];
      return prof;
    }

    protected void shadowInit() {
      shadowCheck(false);
    }

    protected int shadowCheck(bool ReportErrors) {  // returns num of fixes
      // Rules:
      // - Shadows start from the downwind edge of slab
      // - A slab is in shadow if its edge is in shadow
      // - If the top slab of a stack not in shadow, that stack is a new peak
      // - If a stack has no accommodation space, it is zero; otherwise it is height of shadow
      float[,] newShadow = new float[Shadow.GetLength(0), Shadow.GetLength(1)];
      Array.Clear(newShadow, 0, newShadow.Length);
      int h, xs;
      float hs;
      for (int w = 0; w < WidthAcross; w++)
        for (int x = 0; x < LengthDownwind; x++) {
          h = Elev[w, x];
          if (h == 0) continue;
          hs = Math.Max(((float)h), newShadow[w, (x - 1) & mLength] - SHADOW_SLOPE);
          xs = x;
          while (hs >= ((float)Elev[w, xs])) {
            newShadow[w, xs] = hs;
            hs -= SHADOW_SLOPE;
            xs = (xs + 1) & mLength;
          }
        }
      for (int x = 0; x < LengthDownwind; x++)
        for (int w = 0; w < WidthAcross; w++)
          if (newShadow[w, x] == ((float)Elev[w, x]))
            newShadow[w, x] = 0;
      int errors = 0;
      for (int x = 0; x < LengthDownwind; x++)
        for (int w = 0; w < WidthAcross; w++)
          if (newShadow[w, x] != Shadow[w, x])
            errors++;
      if (errors > 0) {
        if (ReportErrors)
          Console.WriteLine("shadowCheck error count: " + errors);
        Array.Copy(newShadow, Shadow, Shadow.Length);
      }
      for (int x = 0; x < LengthDownwind; x++)
        for (int w = 0; w < WidthAcross; w++)
          if ((Shadow[w, x] > 0) && (Shadow[w, x] < Elev[w, x]))
            continue;  // bug -- should never get here
      return errors;
    }

    public virtual void erodeGrain(int w, int x) {
      int wSteep, xSteep;
      while (FindSlope.Upslope(w, x, out wSteep, out xSteep) >= 2) {
        if (openEnded && (((xSteep == mLength) && (x == 0)) || ((xSteep == 0) && (x == mLength))))
          return;  // erosion happens off-field
        w = wSteep;
        x = xSteep;
      }
      float h = --Elev[w, x];
      float hs;
      if (openEnded && (x == 0))
        hs = h;
      else {
        int xs = (x - 1) & mLength;
        hs = Math.Max(h, Math.Max(Elev[w, xs], Shadow[w, xs]) - SHADOW_SLOPE);
      }
      while (hs >= (h = ((float)Elev[w, x]))) {
        Shadow[w, x] = (hs == h) ? 0 : hs;
        hs -= SHADOW_SLOPE;
        x = (x + 1) & mLength;
        if (openEnded && (x == 0))
          return;
      }
      while (Shadow[w, x] > 0) {
        Shadow[w, x] = 0;
        x = (x + 1) & mLength;
        if (openEnded && (x == 0))
          return;
        hs = h - SHADOW_SLOPE;
        if (Shadow[w, x] > hs)
          while (hs >= (h = ((float)Elev[w, x]))) {
            Shadow[w, x] = (hs == h) ? 0 : hs;
            hs -= SHADOW_SLOPE;
            x = (x + 1) & mLength;
            if (openEnded && (x == 0))
              return;
          }
      }
    }

    public virtual void depositGrain(int w, int x) {
      int xSteep, wSteep;
      while (FindSlope.Downslope(w, x, out wSteep, out xSteep) >= 2) {
        if (openEnded && (((xSteep == mLength) && (x == 0)) || ((xSteep == 0) && (x == mLength))))
          break;  // deposit happens at boundary, to keep grains from rolling off
        w = wSteep;
        x = xSteep;
      }
      float h = ++Elev[w, x];
      float hs;
      if (openEnded && (x == 0))
        hs = h;
      else {
        int xs = (x - 1) & mLength;
        hs = Math.Max(h, Math.Max(Elev[w, xs], Shadow[w, xs]) - SHADOW_SLOPE);
      }
      while (hs >= (h = ((float)Elev[w, x]))) {
        Shadow[w, x] = (hs == h) ? 0 : hs;
        hs -= SHADOW_SLOPE;
        x = (x + 1) & mLength;
        if (openEnded && (x == 0))
          return;
      }
    }

    // this is WernerEnhanced
    public virtual void Tick() {
      for (int subticks = LengthDownwind * WidthAcross; subticks > 0; subticks--) {
        int x = rnd.Next(0, LengthDownwind);
        int w = rnd.Next(0, WidthAcross);
        if (Elev[w, x] == 0) continue;
        if (Shadow[w, x] > 0) continue;
        erodeGrain(w, x);
        int i = HopLength;
        while (true) {
          if (++x >= LengthDownwind) {
            if (openEnded)
              break;
            x &= mLength;
          }
          if (Shadow[w, x] > 0) {
            depositGrain(w, x);
            break;
          }
          if (--i <= 0) {
            if (rnd.NextDouble() < (Elev[w, x] > 0 ? pSand : pNoSand)) {
              depositGrain(w, x);
              break;
            }
            i = HopLength;
          }
        }
      }
      shadowCheck(true);
    }

    public virtual int SaltationLength(int w, int x) {
      return HopLength;
    }

    public virtual int SpecialField(int w, int x) {
      return 0;
    }

    #region testRoutines
    //// Test routines:

    private void testInit() {
      Array.Clear(Elev, 0, Elev.Length);
      for (int x = 0; x < 10; x++)
        for (int w = 0; w < WidthAcross; w++)
          Elev[w, x] = (x <= 5) ? x : (10 - x);
      shadowInit();
    }

    private void writeResults(int wOrg, int xOrg, int n) {
      string s = "";
      int x = xOrg;
      for (int i = 0; i < n; i++) {
        s += string.Format("{0,4}", Elev[wOrg, x]);
        x = (x + 1) & mLength;
      }
      Console.WriteLine("[" + wOrg.ToString() + "," + xOrg.ToString() + "] Elev:" + s);
      s = "";
      x = xOrg;
      for (int i = 0; i < n; i++) {
        s += string.Format("{0,4}", Shadow[wOrg, x].ToString("0.0"));
        x = (x + 1) & mLength;
      }
      Console.WriteLine("[" + wOrg.ToString() + "," + xOrg.ToString() + "] Shad:" + s);
    }

    private void showResults() {
      writeResults(0, 0, 16);
      shadowCheck(true);
    }

    private void testCode() {
      testInit(); showResults();
      //bool save = OpenEnded;
      //OpenEnded = true;
      //depositGrain(0, mLength); showResults();
      //depositGrain(0, mLength); showResults();
      //depositGrain(0, 0); showResults();
      //depositGrain(0, 0); showResults();
      //erodeGrain(0, 0); showResults();
      //OpenEnded = save;
      //testInit();
      depositGrain(0, 9); showResults();
      erodeGrain(0, 9); showResults();
      depositGrain(0, 10); showResults();
      erodeGrain(0, 10); showResults();
      erodeGrain(0, 3); showResults();
      depositGrain(0, 6); showResults();
      depositGrain(0, 6); showResults();
      erodeGrain(0, 8); showResults();
      erodeGrain(0, 8); showResults();
      testInit(); showResults();
      erodeGrain(0, 6); showResults();
      erodeGrain(0, 7); showResults();
      depositGrain(0, 6); showResults();
      erodeGrain(0, 6); showResults();
      depositGrain(0, 7); showResults();
      depositGrain(0, 5); showResults();
      testInit();
      erodeGrain(0, 6); showResults();
      erodeGrain(0, 7); showResults();
      erodeGrain(0, 8); showResults();
      depositGrain(0, 9); showResults();
      testInit();
      erodeGrain(0, 5); showResults();
      erodeGrain(0, 5); showResults();
      Array.Clear(Elev, 0, Elev.Length);
      Array.Clear(Shadow, 0, Shadow.Length);
    }

    //    private int[,] offset =
    //    new int[8, 2] { { -1, 0 }, { -1, -1 }, { -1, 1 }, { 0, -1 }, { 0, 1 }, { 1, -1 }, { 1, 0 }, { 1, 1 } };

    //public void testSlopeDetection() {
    //  ProtectedZone = 5;
    //  // should be three 'bad' per test set
    //  for (int center = 1; center < 15; center += 10) {
    //    for (int i = 0; i < 8; i++) {
    //      for (int j = 0; j < 3; j++)
    //        for (int k = 0; k < 3; k++)
    //          Lattice[(center - 1) + j, (center - 1) + k] = 0;
    //      Lattice[center + offset[i, 0], center + offset[i, 1]] = 1;
    //      int iSteep, jSteep;
    //      int v = findUpslope(center, center, out iSteep, out jSteep);
    //      Console.WriteLine(i + "(" + offset[i, 0] + "," + offset[i, 1] + "): [" +
    //          (iSteep - center) + "," + (center - 10) + "]=" + Lattice[iSteep, jSteep] + " v=" + v +
    //          ((Lattice[iSteep, jSteep] == 1) ? " OK" : " Bad"));
    //    }
    //  }
    //  for (int center = 1; center < 15; center += 10) {
    //    for (int i = 0; i < 8; i++) {
    //      for (int j = 0; j < 3; j++)
    //        for (int k = 0; k < 3; k++)
    //          Lattice[(center - 1) + j, (center - 1) + k] = 1;
    //      Lattice[center + offset[i, 0], center + offset[i, 1]] = 0;
    //      int iSteep, jSteep;
    //      int v = findDownslope(center, center, out iSteep, out jSteep);
    //      Console.WriteLine(i + "(" + offset[i, 0] + "," + offset[i, 1] + "): [" +
    //          (iSteep - center) + "," + (center - 10) + "]=" + Lattice[iSteep, jSteep] + " v=" + v +
    //          ((Lattice[iSteep, jSteep] == 0) ? " OK" : " Bad"));
    //    }
    //  }
    //}
    #endregion

  }
}