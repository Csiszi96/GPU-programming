using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DunefieldModel {
  class Momiji2000 : Model {
    private float hRef;
    private const float WindSpeedUpFactor = 0.4f;
    private const float NonlinearFactor = 0.002f;

    public Momiji2000(Form1 ParentForm, IFindSlope SlopeFinder, int WidthAcross, int LengthDownwind) :
      base(ParentForm, SlopeFinder, WidthAcross, LengthDownwind) { }

    public override bool UsesHopLength() {
      return false;
    }

    private float hRefCalc() {
      float sum = 0;
      if (openEnded)
        AverageHeight = AveHeight();
      for (int x = 0; x < LengthDownwind; x++)
        for (int w = 0; w < WidthAcross; w++)
          sum += Math.Abs(Elev[w, x] - AverageHeight);
      return AverageHeight - sum / (2 * ((float)(LengthDownwind)) * ((float)WidthAcross));
    }

    public override void Tick() {
      int saltationLeap;
      float dh;
      hRef = AverageHeight; // hRefCalc();
      for (int subticks = LengthDownwind * WidthAcross; subticks > 0; subticks--) {
        int x = rnd.Next(0, LengthDownwind);
        int w = rnd.Next(0, WidthAcross);
        int h = Elev[w, x];
        if (h == 0) continue;
        if (Shadow[w, x] > 0) continue;
        erodeGrain(w, x);
        while (true) {
          dh = h - hRef;
          if (dh > 0)
            saltationLeap = HopLength + (int)Math.Round(WindSpeedUpFactor * dh + NonlinearFactor * dh * dh);
          else  //    ******** If changing these, also change SaltationLength routine below *********
            saltationLeap = HopLength + (int)Math.Round(WindSpeedUpFactor * dh);
          x += saltationLeap;
          if (x >= LengthDownwind) {
            if (openEnded)
              break;
            x &= mLength;
          }
          if ((Shadow[w, x] > 0) || (rnd.NextDouble() < (h > 0 ? pSand : pNoSand))) {
            depositGrain(w, x);
            break;
          }
          h = Elev[w, x];
        }
      }
    }

    public override int SaltationLength(int w, int x) {
      // go with most-recent; double hRef = hRefCalc();
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