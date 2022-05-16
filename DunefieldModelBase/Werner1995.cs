﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DunefieldModel {
  public class Werner1995 : Model {

    public Werner1995(Form1 ParentForm, IFindSlope SlopeFinder, int WidthAcross, int LengthDownwind) :
      base(ParentForm, SlopeFinder, WidthAcross, LengthDownwind) { }

    public override void Tick() {
      for (int subticks = LengthDownwind * WidthAcross; subticks > 0; subticks--) {
        int x = rnd.Next(0, LengthDownwind);  // get coordinates [w, x] for a random cell
        int w = rnd.Next(0, WidthAcross);
        if (Elev[w, x] == 0) continue;    // if the cell is bare, get another cell
        //if (Shadow[w, x] > 0) continue; // if the cell is in shadow, get another cell (not in Werner(1995))
        erodeGrain(w, x);                 // remove slab from this cell (also do any needed avalanching)
        while (true) {                    // repeat until slab is deposited or lost
          x += HopLength;                 // jump downwind by saltation hop length
          if (x >= LengthDownwind) {      // if past end of grid...
            if (openEnded)                // exit loop if open ended (discard slab)
              break;
            x &= mLength;                 // otherwise, wrap to start of grid (grid length is a power of 2)
          }
          if ((Shadow[w, x] > 0) ||       // if new location is in shadow, or 
              (rnd.NextDouble() < (Elev[w, x] > 0 ? pSand : pNoSand))) {  // if slab sticks (not a bounce)
            depositGrain(w, x);           // then deposit slab (also do any needed avalanching)
            break;
          }
        }
      }
    }

  }
}