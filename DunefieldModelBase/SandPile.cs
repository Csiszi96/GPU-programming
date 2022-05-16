using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DunefieldModel {
  class SandPile : Model {

    public SandPile(Form1 ParentForm, IFindSlope SlopeFinder, int WidthAcross, int LengthDownwind) :
      base(ParentForm, SlopeFinder, WidthAcross, LengthDownwind) { }

    public override bool UsesHopLength() {
      return false;
    }

    public override bool UsesSandProbabilities() {
      return false;
    }

    public override void Tick() {
      if (Elev[0, 0] == 0)
        depositGrain(WidthAcross / 2, LengthDownwind / 2);
      else
        erodeGrain(WidthAcross / 2, LengthDownwind / 2);
    }

  }
}