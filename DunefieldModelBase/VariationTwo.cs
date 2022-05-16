using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DunefieldModel {
  class VariationTwo : Model {
    private float hRef;
    private const float WindSpeedUpFactor = 0.4f;
    private const float NonlinearFactor = 0.002f;

    public VariationTwo(Form1 ParentForm, IFindSlope SlopeFinder, int WidthAcross, int LengthDownwind) :
      base(ParentForm, SlopeFinder, WidthAcross, LengthDownwind) { }

    public override bool UsesHopLength() {
      return false;
    }

    private float hRefCalc() {   // this routine seems to come up with a lower value for href than AverageHeight
      float sum = 0;
      if (openEnded)
        AverageHeight = AveHeight();
      for (int x = 0; x < LengthDownwind; x++)
        for (int w = 0; w < WidthAcross; w++)
          sum += Math.Abs(Elev[w, x] - AverageHeight);
      return AverageHeight - sum / (2 * ((float)(LengthDownwind)) * ((float)WidthAcross));
    }

    public override void Tick() {
      hRef = AverageHeight;                   // hRef is the average sand height on the dunefield
      int i;
      for (int subticks = LengthDownwind * WidthAcross; subticks > 0; subticks--) {
        int x = rnd.Next(0, LengthDownwind);  // get coordinates [w, x] of a random cell
        int w = rnd.Next(0, WidthAcross);
        int h = Elev[w, x];                   // sand height at chosen cell
        if (h == 0) continue;                 // if the cell is bare, get another cell
        if (Shadow[w, x] > 0) continue;       // if the cell is in shadow, get another cell
        erodeGrain(w, x);                     // remove slab from this cell (also do any needed avalanching)
        float dh = ((float)h) - hRef;
        if (dh > 0)                           // calc effective saltation length, based on sand height
          i = HopLength + (int)Math.Round(WindSpeedUpFactor * dh + NonlinearFactor * dh * dh);
        else
          i = HopLength + (int)Math.Round(WindSpeedUpFactor * dh);
        //if (i < 0)                           // make sure the hop is never backwards
        //  i = 0;
        while (true) {                       // repeat until slab is deposited or lost
          if (++x >= LengthDownwind) {       // Move one cell downwind.  If past end of grid...
            if (openEnded)                   // exit loop if open ended (discard slab)
              break;
            x &= mLength;                    // otherwise, wrap to start of grid (which is a power of 2 in length
          }
          if (Shadow[w, x] > 0) {            // if new location is in shadow
            depositGrain(w, x);
            break;
          }
          if (--i <= 0) {                    // if we have moved the desired hop length, try a deposit
            if (rnd.NextDouble() < (Elev[w, x] > 0 ? pSand : pNoSand)) {  // if slab sticks (no bounce)
              depositGrain(w, x);            // then deposit slab (also do any needed avalanching)
              break;
            }
            h = Elev[w, x];                  // didn't deposit; prepare for another hop
            dh = ((float)h) - hRef;          // calc wind speed-up factor at this new spot
            if (dh > 0)
              i = HopLength + (int)Math.Round(WindSpeedUpFactor * dh + NonlinearFactor * dh * dh);
            else
              i = HopLength + (int)Math.Round(WindSpeedUpFactor * dh);
            //if (i < 0)
            //  i = 0;                         // make sure the effective saltation length is not negative
          }
        }
      }
    }

    public override int SaltationLength(int w, int x) {
      // go with most-recent; double hRef = hRefCalc();
      int saltationLeap;
      float dh = ((float)Elev[w, x]) - hRef;
      if (dh > 0)
        saltationLeap = HopLength + (int)Math.Round(WindSpeedUpFactor * dh + NonlinearFactor * dh * dh);
      else {
        saltationLeap = HopLength + (int)Math.Round(WindSpeedUpFactor * dh);
        if (saltationLeap < 0)
          saltationLeap = 0;
      }
      return saltationLeap;
    }

  }
}