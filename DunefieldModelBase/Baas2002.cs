using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DunefieldModel {
  public class Baas2002 : Model {

    public Baas2002(Form1 ParentForm, IFindSlope SlopeFinder, int WidthAcross, int LengthDownwind) :
        base(ParentForm, SlopeFinder, WidthAcross, LengthDownwind) {
      HopLength = 1;
    }

    public override bool UsesHopLength() {
      return false;
    }

    public override void Tick() {
      int subticks = WidthAcross * LengthDownwind;
      // each grain is to be 'polled' only once, but in random order (see Nield 2008)
      int[] order = new int[subticks];
      for (int i = 0; i < subticks; i++)
        order[i] = i;
      do {
        int i = rnd.Next(0, subticks);
        int w = order[i] / LengthDownwind;
        int x = order[i] % LengthDownwind;
        subticks--;
        order[i] = order[subticks];
        if (Elev[w, x] == 0) continue;
        if (Shadow[w, x] > 0) continue;
        erodeGrain(w, x);
        while (true) {
          if (++x >= LengthDownwind) {
            if (openEnded)
              break;
            x &= mLength;
          }
          if ((Shadow[w, x] > 0) || (rnd.NextDouble() < (Elev[w, x] > 0 ? pSand : pNoSand))) {
            depositGrain(w, x);
            break;
          }
        }
      } while (subticks > 0);
    }

  }
}