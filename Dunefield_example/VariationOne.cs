using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DunefieldModel {
  class VariationOne : Model {
    private const float WindSpeedUpFactor = 0.4f;
    private const float NonlinearFactor = 0.002f;

    public VariationOne(Form1 ParentForm, IFindSlope SlopeFinder, int WidthAcross, int LengthDownwind) :
      base(ParentForm, SlopeFinder, WidthAcross, LengthDownwind) { }

    public override bool UsesHopLength() {
      return true;
    }

    public override void Tick() {
      int i;
      for (int subticks = LengthDownwind * WidthAcross; subticks > 0; subticks--) {
        int x = rnd.Next(0, LengthDownwind);  // get coordinates [w, x] of a random cell
        int w = rnd.Next(0, WidthAcross);
        int h = Elev[w, x];                   // sand height at chosen cell
        if (h == 0) continue;                 // if the cell is bare, get another cell
        if (Shadow[w, x] > 0) continue;       // if the cell is in shadow, get another cell
        erodeGrain(w, x);                     // remove slab from this cell (also do any needed avalanching)
        i = HopLength;
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
            i = HopLength;
          }
        }
      }
    }

  }
}