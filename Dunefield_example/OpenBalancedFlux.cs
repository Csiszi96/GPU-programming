using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DunefieldModel {
  class OpenBalancedFlux : Model {
    private float hRef;
    private const float WindSpeedUpFactor = 0.4f;
    private const float NonlinearFactor = 0.002f;
    private float[] influxRates = new float[7];
    private float influxInterval;
    private float influxCounter;

    public OpenBalancedFlux(Form1 ParentForm, IFindSlope SlopeFinder, int WidthAcross, int LengthDownwind) :
      base(ParentForm, SlopeFinder, WidthAcross, LengthDownwind) {
      FindSlope = new FindSlopeLateral();
      FindSlope.Init(ref Elev, WidthAcross, LengthDownwind);
      float sum = 0;
      for (int i = 0; i < influxRates.Length; i++) {
        influxRates[i] = (1 - sum) * pSand;
        sum += (float)Math.Pow((1 - pSand), i) * pSand;
      }
      for (int i = 1; i < influxRates.Length; i++)
        influxRates[i] += influxRates[i - 1];
      influxInterval = ((float)(LengthDownwind * WidthAcross)) / ((1 / pSand) * ((float)HopLength * WidthAcross)) + 0;
      influxCounter = influxInterval;
    }

    public override void erodeGrain(int w, int x) {
      int wSteep, xSteep;
      while (FindSlope.Upslope(w, x, out wSteep, out xSteep) > 2) {
        if (openEnded && (((xSteep == mLength) && (x == 0)) || ((xSteep == 0) && (x == mLength))))
          break;
        w = wSteep;
        x = xSteep;
      }
      Elev[w, x]--;
    }

    public override void depositGrain(int w, int x) {
      int xSteep, wSteep;
      while (FindSlope.Downslope(w, x, out wSteep, out xSteep) > 2) {
        if (openEnded && (((xSteep == mLength) && (x == 0)) || ((xSteep == 0) && (x == mLength))))
          return;
        w = wSteep;
        x = xSteep;
      }
      Elev[w, x]++;
    }

    private void depositGrainX(int w, int x) {
      int xSteep, wSteep;
      while (FindSlope.Downslope(w, x, out wSteep, out xSteep) > 2) {
        if (xSteep == mLength)
          return;
        w = wSteep;
        x = xSteep;
      }
      Elev[w, x]++;
    }

    private float hRefCalc() {
      float sum = 0;
      int startAt = 0;
      if (openEnded)
        AverageHeight = AveHeight();
      for (int x = startAt; x < LengthDownwind; x++)
        for (int w = 0; w < WidthAcross; w++)
          sum += Math.Abs(Elev[w, x] - AverageHeight);
      return AverageHeight - sum / (2 * ((float)(LengthDownwind - startAt)) * ((float)WidthAcross));
    }

    public override void Tick() {
      hRef = hRefCalc();
      int saltationLeap = HopLength;
      for (int subticks = LengthDownwind * WidthAcross; subticks > 0; subticks--) {
        int x = rnd.Next(0, LengthDownwind);
        int w = rnd.Next(0, WidthAcross);
        int h = Elev[w, x];
        if (h == 0) continue;
        //if (Shadow[w, x] > 0) continue;
        //if ((x == 0) && (w == 0))
        //  x = x;
        if (openEnded && (--influxCounter < 0)) {
          double p = rnd.NextDouble();   // zero inclusive to one exclusive
          int i;
          for (i = 0; i < influxRates.Length; i++)
            if (p < influxRates[i])
              break;
          depositGrainX(w, i * HopLength + (x % HopLength));
          influxCounter += influxInterval;
        }
        erodeGrain(w, x);
        while (true) {
          float dh = ((float)h) - hRef;
          if (dh > 0)
            saltationLeap = HopLength + (int)Math.Round(WindSpeedUpFactor * dh + NonlinearFactor * dh * dh);
          else  //    ******** If changing these, also change SaltationLength routine below *********
            saltationLeap = HopLength + (int)Math.Round(WindSpeedUpFactor * dh);
          saltationLeap = HopLength;  // DEBUG
          x += saltationLeap;
          if (x >= LengthDownwind) {
            if (openEnded)
              break;
            x &= mLength;
          }
          //if ((Shadow[w, x] > 0) || (rnd.NextDouble() < (h > 0 ? pSand : pNoSand))) {
          if ((rnd.NextDouble() < (h > 0 ? pSand : pNoSand))) {
            depositGrain(w, x);
            break;
          }
          h = Elev[w, x];
        }
      }
    }

    public override int SaltationLength(int w, int x) {
      // go with most-recent; float hRef = hRefCalc();
      int saltationLeap;
      float dh = ((float)Elev[w, x]) - hRef;
      if (dh > 0)
        saltationLeap = HopLength + (int)Math.Round(WindSpeedUpFactor * dh + NonlinearFactor * dh * dh);
      else
        saltationLeap = HopLength + (int)Math.Round(WindSpeedUpFactor * dh);
      return saltationLeap;
    }

  }
}