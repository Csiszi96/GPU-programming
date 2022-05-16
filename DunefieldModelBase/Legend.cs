using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Windows.Forms;

namespace DunefieldModel {
  public class LegendDiscrete {
    public double Minimum {
      get { return min; }
      set {
        min = value;
        range = max - min;
      }
    }
    public double Maximum {
      get { return max; }
      set {
        max = value;
        range = max - min;
      }
    }
    private double min;
    private double max;
    private double range;
    private Color[] colourTable;

    public LegendDiscrete() {
      List<Color> c = new List<Color>();
      c.Add(Color.Black);
      //c.Add(Color.FromArgb(120, 120, 120));
      //c.Add(Color.FromArgb(193, 193, 193));
      //c.Add(Color.FromArgb(102, 0, 153));
      c.Add(Color.FromArgb(186, 0, 255));
      c.Add(Color.FromArgb(0, 0, 255));
      c.Add(Color.FromArgb(45, 129, 255));
      c.Add(Color.FromArgb(0, 198, 255));
      c.Add(Color.FromArgb(0, 153, 0));
      c.Add(Color.FromArgb(25, 219, 0));
      c.Add(Color.FromArgb(180, 255, 0));
      c.Add(Color.FromArgb(255, 255, 0));
      c.Add(Color.FromArgb(255, 162, 0));
      //c.Add(Color.FromArgb(204, 0, 0));
      c.Add(Color.FromArgb(255, 0, 0));
      //c.Add(Color.FromArgb(255, 182, 182));
      colourTable = c.ToArray<Color>();
    }

    public void MinMax(double Minimum, double Maximum) {
      min = Minimum;
      max = Maximum;
      range = max - min;
    }

    public void Render(PictureBox pic) {
      Bitmap bm = (Bitmap)pic.Image;
      if (pic.Width > pic.Height) {
        for (int w = 0; w < pic.ClientRectangle.Width; w++) {
          Color c = Colour(min + ((double)w) / ((double)pic.ClientRectangle.Width) * range);
          for (int h = 0; h < pic.ClientRectangle.Height; h++)
            bm.SetPixel(w, h, c);
        }
      } else {
        for (int h = 0; h < pic.ClientRectangle.Height; h++) {
          Color c = Colour(min + ((double)h) / ((double)pic.ClientRectangle.Width) * range);
          for (int w = 0; w < pic.ClientRectangle.Width; w++)
            bm.SetPixel(w, h, c);
        }
      }
      pic.Refresh();
    }

    public Color Colour(double value) {
      // each colour is [lower, upper), with special case of max included in top category
      // thus [n, n + 1) is assigned to n
      if ((range == 0) || (value < min) || (value == 0))
        return colourTable[0];
      else if (value >= max)
        return colourTable[colourTable.Length - 1];
      return colourTable[((int)((value - min) / range * ((double)colourTable.Length - 1))) + 1];
    }

    public struct ColorRGB {
      public byte R;
      public byte G;
      public byte B;

      public ColorRGB(Color value) {
        this.R = value.R;
        this.G = value.G;
        this.B = value.B;
      }

      public static implicit operator Color(ColorRGB rgb) {
        Color c = Color.FromArgb(rgb.R, rgb.G, rgb.B);
        return c;
      }

      public static explicit operator ColorRGB(Color c) {
        return new ColorRGB(c);
      }
    }

    // Given H,S,L in range of 0-1
    // Returns a Color (RGB struct) in range of 0-255
    public static ColorRGB HSL2RGB(double h, double sl, double l, bool useAlt) {
      double v;
      double r, g, b;
      r = l;   // default to gray
      g = l;
      b = l;
      v = (l <= 0.5) ? (l * (1.0 + sl)) : (l + sl - l * sl);
      if (v > 0) {
        double m;
        double sv;
        int sextant;
        double fract, vsf, mid1, mid2;
        m = l + l - v;
        sv = (v - m) / v;
        h *= 6.0;
        sextant = (int)h;
        fract = h - sextant;
        vsf = v * sv * fract;
        mid1 = m + vsf;
        mid2 = v - vsf;
        switch (sextant) {
          case 0:
            r = v;
            g = useAlt ? Math.Acos(1 - mid1) / 1.5708 : mid1;
            b = m;
            break;
          case 1:
            r = useAlt ? Math.Acos(1 - mid2) / 1.5708 : mid2;
            g = v;
            b = m;
            break;
          case 2:
            r = m;
            g = v;
            b = useAlt ? Math.Acos(1 - mid1) / 1.5708 : mid1;
            break;
          case 3:
            r = m;
            g = useAlt ? Math.Acos(1 - mid2) / 1.5708 : mid2;
            b = v;
            break;
          case 4:
            r = mid1;
            g = m;
            b = v;
            break;
          case 5:
            r = v;
            g = m;
            b = mid2;
            break;
        }
      }
      ColorRGB rgb;
      rgb.R = Convert.ToByte(r * 255.0f);
      rgb.G = Convert.ToByte(g * 255.0f);
      rgb.B = Convert.ToByte(b * 255.0f);
      return rgb;
    }

  }

  public class LegendContinuous {
    public double Minimum {
      get { return min; }
      set {
        min = value;
        range = max - min;
      }
    }
    public double Maximum {
      get { return max; }
      set {
        max = value;
        range = max - min;
      }
    }
    private double min;
    private double max;
    private double range;
    private Color[] colourTable = new Color[1001];

    public LegendContinuous() {
      colourTable[1000] = Color.FromArgb(255, 0, 0);
      #region ctable
      colourTable[999] = Color.FromArgb(255, 1, 0);
      colourTable[998] = Color.FromArgb(255, 2, 0);
      colourTable[997] = Color.FromArgb(255, 3, 0);
      colourTable[996] = Color.FromArgb(255, 4, 0);
      colourTable[995] = Color.FromArgb(255, 5, 0);
      colourTable[994] = Color.FromArgb(255, 7, 0);
      colourTable[993] = Color.FromArgb(255, 8, 0);
      colourTable[992] = Color.FromArgb(255, 9, 0);
      colourTable[991] = Color.FromArgb(255, 10, 0);
      colourTable[990] = Color.FromArgb(255, 11, 0);
      colourTable[989] = Color.FromArgb(255, 12, 0);
      colourTable[988] = Color.FromArgb(255, 13, 0);
      colourTable[987] = Color.FromArgb(255, 14, 0);
      colourTable[986] = Color.FromArgb(255, 15, 0);
      colourTable[985] = Color.FromArgb(255, 17, 0);
      colourTable[984] = Color.FromArgb(255, 18, 0);
      colourTable[983] = Color.FromArgb(255, 19, 0);
      colourTable[982] = Color.FromArgb(255, 20, 0);
      colourTable[981] = Color.FromArgb(255, 21, 0);
      colourTable[980] = Color.FromArgb(255, 22, 0);
      colourTable[979] = Color.FromArgb(255, 23, 0);
      colourTable[978] = Color.FromArgb(255, 24, 0);
      colourTable[977] = Color.FromArgb(255, 25, 0);
      colourTable[976] = Color.FromArgb(255, 26, 0);
      colourTable[975] = Color.FromArgb(255, 27, 0);
      colourTable[974] = Color.FromArgb(255, 29, 0);
      colourTable[973] = Color.FromArgb(255, 30, 0);
      colourTable[972] = Color.FromArgb(255, 31, 0);
      colourTable[971] = Color.FromArgb(255, 32, 0);
      colourTable[970] = Color.FromArgb(255, 33, 0);
      colourTable[969] = Color.FromArgb(255, 34, 0);
      colourTable[968] = Color.FromArgb(255, 35, 0);
      colourTable[967] = Color.FromArgb(255, 36, 0);
      colourTable[966] = Color.FromArgb(255, 37, 0);
      colourTable[965] = Color.FromArgb(255, 38, 0);
      colourTable[964] = Color.FromArgb(255, 39, 0);
      colourTable[963] = Color.FromArgb(255, 40, 0);
      colourTable[962] = Color.FromArgb(255, 40, 0);
      colourTable[961] = Color.FromArgb(255, 42, 0);
      colourTable[960] = Color.FromArgb(255, 43, 0);
      colourTable[959] = Color.FromArgb(255, 44, 0);
      colourTable[958] = Color.FromArgb(255, 45, 0);
      colourTable[957] = Color.FromArgb(255, 46, 0);
      colourTable[956] = Color.FromArgb(255, 47, 0);
      colourTable[955] = Color.FromArgb(255, 47, 0);
      colourTable[954] = Color.FromArgb(255, 48, 0);
      colourTable[953] = Color.FromArgb(255, 49, 0);
      colourTable[952] = Color.FromArgb(255, 50, 0);
      colourTable[951] = Color.FromArgb(255, 51, 0);
      colourTable[950] = Color.FromArgb(255, 51, 0);
      colourTable[949] = Color.FromArgb(255, 52, 0);
      colourTable[948] = Color.FromArgb(255, 53, 0);
      colourTable[947] = Color.FromArgb(255, 54, 0);
      colourTable[946] = Color.FromArgb(255, 55, 0);
      colourTable[945] = Color.FromArgb(255, 55, 0);
      colourTable[944] = Color.FromArgb(255, 56, 0);
      colourTable[943] = Color.FromArgb(255, 57, 0);
      colourTable[942] = Color.FromArgb(255, 58, 0);
      colourTable[941] = Color.FromArgb(255, 59, 0);
      colourTable[940] = Color.FromArgb(255, 60, 0);
      colourTable[939] = Color.FromArgb(255, 61, 0);
      colourTable[938] = Color.FromArgb(255, 62, 0);
      colourTable[937] = Color.FromArgb(255, 63, 0);
      colourTable[936] = Color.FromArgb(255, 64, 0);
      colourTable[935] = Color.FromArgb(255, 65, 0);
      colourTable[934] = Color.FromArgb(255, 67, 0);
      colourTable[933] = Color.FromArgb(255, 68, 0);
      colourTable[932] = Color.FromArgb(255, 69, 0);
      colourTable[931] = Color.FromArgb(255, 70, 0);
      colourTable[930] = Color.FromArgb(255, 72, 0);
      colourTable[929] = Color.FromArgb(255, 73, 0);
      colourTable[928] = Color.FromArgb(255, 74, 0);
      colourTable[927] = Color.FromArgb(255, 75, 0);
      colourTable[926] = Color.FromArgb(255, 76, 0);
      colourTable[925] = Color.FromArgb(255, 77, 0);
      colourTable[924] = Color.FromArgb(255, 79, 0);
      colourTable[923] = Color.FromArgb(255, 80, 0);
      colourTable[922] = Color.FromArgb(255, 81, 0);
      colourTable[921] = Color.FromArgb(255, 82, 0);
      colourTable[920] = Color.FromArgb(255, 84, 0);
      colourTable[919] = Color.FromArgb(255, 85, 0);
      colourTable[918] = Color.FromArgb(255, 87, 0);
      colourTable[917] = Color.FromArgb(255, 88, 0);
      colourTable[916] = Color.FromArgb(255, 89, 0);
      colourTable[915] = Color.FromArgb(255, 91, 0);
      colourTable[914] = Color.FromArgb(255, 92, 0);
      colourTable[913] = Color.FromArgb(255, 94, 0);
      colourTable[912] = Color.FromArgb(255, 95, 0);
      colourTable[911] = Color.FromArgb(255, 97, 0);
      colourTable[910] = Color.FromArgb(255, 98, 0);
      colourTable[909] = Color.FromArgb(255, 99, 0);
      colourTable[908] = Color.FromArgb(255, 101, 0);
      colourTable[907] = Color.FromArgb(255, 102, 0);
      colourTable[906] = Color.FromArgb(255, 104, 0);
      colourTable[905] = Color.FromArgb(255, 105, 0);
      colourTable[904] = Color.FromArgb(255, 107, 0);
      colourTable[903] = Color.FromArgb(255, 108, 0);
      colourTable[902] = Color.FromArgb(255, 110, 0);
      colourTable[901] = Color.FromArgb(255, 111, 0);
      colourTable[900] = Color.FromArgb(255, 112, 0);
      colourTable[899] = Color.FromArgb(255, 114, 0);
      colourTable[898] = Color.FromArgb(255, 115, 0);
      colourTable[897] = Color.FromArgb(255, 117, 0);
      colourTable[896] = Color.FromArgb(255, 118, 0);
      colourTable[895] = Color.FromArgb(255, 119, 0);
      colourTable[894] = Color.FromArgb(255, 121, 0);
      colourTable[893] = Color.FromArgb(255, 122, 0);
      colourTable[892] = Color.FromArgb(255, 123, 0);
      colourTable[891] = Color.FromArgb(255, 125, 0);
      colourTable[890] = Color.FromArgb(255, 126, 0);
      colourTable[889] = Color.FromArgb(255, 127, 0);
      colourTable[888] = Color.FromArgb(255, 128, 0);
      colourTable[887] = Color.FromArgb(255, 129, 0);
      colourTable[886] = Color.FromArgb(255, 130, 0);
      colourTable[885] = Color.FromArgb(255, 130, 0);
      colourTable[884] = Color.FromArgb(255, 131, 0);
      colourTable[883] = Color.FromArgb(255, 132, 0);
      colourTable[882] = Color.FromArgb(255, 133, 0);
      colourTable[881] = Color.FromArgb(255, 133, 0);
      colourTable[880] = Color.FromArgb(255, 134, 0);
      colourTable[879] = Color.FromArgb(255, 135, 0);
      colourTable[878] = Color.FromArgb(255, 135, 0);
      colourTable[877] = Color.FromArgb(255, 136, 0);
      colourTable[876] = Color.FromArgb(255, 137, 0);
      colourTable[875] = Color.FromArgb(255, 137, 0);
      colourTable[874] = Color.FromArgb(255, 138, 0);
      colourTable[873] = Color.FromArgb(255, 139, 0);
      colourTable[872] = Color.FromArgb(255, 139, 0);
      colourTable[871] = Color.FromArgb(255, 140, 0);
      colourTable[870] = Color.FromArgb(255, 141, 0);
      colourTable[869] = Color.FromArgb(255, 141, 0);
      colourTable[868] = Color.FromArgb(255, 142, 0);
      colourTable[867] = Color.FromArgb(255, 143, 0);
      colourTable[866] = Color.FromArgb(255, 143, 0);
      colourTable[865] = Color.FromArgb(255, 144, 0);
      colourTable[864] = Color.FromArgb(255, 145, 0);
      colourTable[863] = Color.FromArgb(255, 145, 0);
      colourTable[862] = Color.FromArgb(255, 146, 0);
      colourTable[861] = Color.FromArgb(255, 146, 0);
      colourTable[860] = Color.FromArgb(255, 147, 0);
      colourTable[859] = Color.FromArgb(254, 148, 0);
      colourTable[858] = Color.FromArgb(254, 148, 0);
      colourTable[857] = Color.FromArgb(254, 149, 0);
      colourTable[856] = Color.FromArgb(254, 149, 0);
      colourTable[855] = Color.FromArgb(254, 150, 0);
      colourTable[854] = Color.FromArgb(254, 151, 0);
      colourTable[853] = Color.FromArgb(254, 151, 0);
      colourTable[852] = Color.FromArgb(254, 152, 0);
      colourTable[851] = Color.FromArgb(254, 152, 0);
      colourTable[850] = Color.FromArgb(254, 153, 0);
      colourTable[849] = Color.FromArgb(254, 154, 0);
      colourTable[848] = Color.FromArgb(254, 154, 0);
      colourTable[847] = Color.FromArgb(254, 155, 0);
      colourTable[846] = Color.FromArgb(254, 155, 0);
      colourTable[845] = Color.FromArgb(254, 156, 0);
      colourTable[844] = Color.FromArgb(254, 157, 0);
      colourTable[843] = Color.FromArgb(254, 157, 0);
      colourTable[842] = Color.FromArgb(254, 158, 0);
      colourTable[841] = Color.FromArgb(254, 158, 0);
      colourTable[840] = Color.FromArgb(254, 159, 0);
      colourTable[839] = Color.FromArgb(254, 160, 0);
      colourTable[838] = Color.FromArgb(254, 160, 0);
      colourTable[837] = Color.FromArgb(254, 161, 0);
      colourTable[836] = Color.FromArgb(254, 161, 0);
      colourTable[835] = Color.FromArgb(254, 162, 0);
      colourTable[834] = Color.FromArgb(254, 163, 0);
      colourTable[833] = Color.FromArgb(254, 163, 0);
      colourTable[832] = Color.FromArgb(254, 164, 0);
      colourTable[831] = Color.FromArgb(254, 164, 0);
      colourTable[830] = Color.FromArgb(254, 165, 0);
      colourTable[829] = Color.FromArgb(254, 166, 0);
      colourTable[828] = Color.FromArgb(254, 166, 0);
      colourTable[827] = Color.FromArgb(254, 167, 0);
      colourTable[826] = Color.FromArgb(254, 168, 0);
      colourTable[825] = Color.FromArgb(254, 168, 0);
      colourTable[824] = Color.FromArgb(254, 169, 0);
      colourTable[823] = Color.FromArgb(254, 170, 0);
      colourTable[822] = Color.FromArgb(254, 170, 0);
      colourTable[821] = Color.FromArgb(254, 171, 0);
      colourTable[820] = Color.FromArgb(254, 172, 0);
      colourTable[819] = Color.FromArgb(254, 173, 0);
      colourTable[818] = Color.FromArgb(254, 174, 0);
      colourTable[817] = Color.FromArgb(254, 175, 0);
      colourTable[816] = Color.FromArgb(254, 175, 0);
      colourTable[815] = Color.FromArgb(254, 176, 0);
      colourTable[814] = Color.FromArgb(254, 177, 0);
      colourTable[813] = Color.FromArgb(254, 178, 0);
      colourTable[812] = Color.FromArgb(254, 180, 0);
      colourTable[811] = Color.FromArgb(254, 181, 0);
      colourTable[810] = Color.FromArgb(254, 182, 0);
      colourTable[809] = Color.FromArgb(254, 183, 0);
      colourTable[808] = Color.FromArgb(254, 184, 0);
      colourTable[807] = Color.FromArgb(254, 185, 0);
      colourTable[806] = Color.FromArgb(254, 186, 0);
      colourTable[805] = Color.FromArgb(254, 187, 0);
      colourTable[804] = Color.FromArgb(254, 188, 0);
      colourTable[803] = Color.FromArgb(254, 189, 0);
      colourTable[802] = Color.FromArgb(254, 190, 0);
      colourTable[801] = Color.FromArgb(254, 191, 0);
      colourTable[800] = Color.FromArgb(254, 192, 0);
      colourTable[799] = Color.FromArgb(254, 193, 0);
      colourTable[798] = Color.FromArgb(254, 194, 0);
      colourTable[797] = Color.FromArgb(254, 195, 0);
      colourTable[796] = Color.FromArgb(254, 196, 0);
      colourTable[795] = Color.FromArgb(254, 197, 0);
      colourTable[794] = Color.FromArgb(254, 198, 0);
      colourTable[793] = Color.FromArgb(254, 199, 0);
      colourTable[792] = Color.FromArgb(254, 200, 0);
      colourTable[791] = Color.FromArgb(254, 201, 0);
      colourTable[790] = Color.FromArgb(254, 202, 0);
      colourTable[789] = Color.FromArgb(254, 203, 0);
      colourTable[788] = Color.FromArgb(254, 204, 0);
      colourTable[787] = Color.FromArgb(254, 205, 0);
      colourTable[786] = Color.FromArgb(254, 206, 0);
      colourTable[785] = Color.FromArgb(254, 207, 0);
      colourTable[784] = Color.FromArgb(254, 208, 0);
      colourTable[783] = Color.FromArgb(254, 209, 0);
      colourTable[782] = Color.FromArgb(254, 210, 0);
      colourTable[781] = Color.FromArgb(254, 211, 0);
      colourTable[780] = Color.FromArgb(254, 212, 0);
      colourTable[779] = Color.FromArgb(254, 213, 0);
      colourTable[778] = Color.FromArgb(254, 214, 0);
      colourTable[777] = Color.FromArgb(254, 215, 0);
      colourTable[776] = Color.FromArgb(254, 216, 0);
      colourTable[775] = Color.FromArgb(254, 217, 0);
      colourTable[774] = Color.FromArgb(254, 218, 0);
      colourTable[773] = Color.FromArgb(254, 218, 0);
      colourTable[772] = Color.FromArgb(254, 219, 0);
      colourTable[771] = Color.FromArgb(254, 220, 0);
      colourTable[770] = Color.FromArgb(254, 221, 0);
      colourTable[769] = Color.FromArgb(254, 222, 0);
      colourTable[768] = Color.FromArgb(254, 222, 0);
      colourTable[767] = Color.FromArgb(254, 223, 0);
      colourTable[766] = Color.FromArgb(254, 224, 0);
      colourTable[765] = Color.FromArgb(254, 225, 0);
      colourTable[764] = Color.FromArgb(254, 225, 0);
      colourTable[763] = Color.FromArgb(254, 226, 0);
      colourTable[762] = Color.FromArgb(254, 227, 0);
      colourTable[761] = Color.FromArgb(254, 228, 0);
      colourTable[760] = Color.FromArgb(254, 229, 0);
      colourTable[759] = Color.FromArgb(254, 229, 0);
      colourTable[758] = Color.FromArgb(254, 230, 0);
      colourTable[757] = Color.FromArgb(254, 231, 0);
      colourTable[756] = Color.FromArgb(254, 231, 0);
      colourTable[755] = Color.FromArgb(254, 232, 0);
      colourTable[754] = Color.FromArgb(254, 233, 0);
      colourTable[753] = Color.FromArgb(254, 234, 0);
      colourTable[752] = Color.FromArgb(254, 234, 0);
      colourTable[751] = Color.FromArgb(254, 235, 0);
      colourTable[750] = Color.FromArgb(254, 236, 0);
      colourTable[749] = Color.FromArgb(254, 236, 0);
      colourTable[748] = Color.FromArgb(254, 237, 0);
      colourTable[747] = Color.FromArgb(254, 238, 0);
      colourTable[746] = Color.FromArgb(254, 238, 0);
      colourTable[745] = Color.FromArgb(254, 239, 0);
      colourTable[744] = Color.FromArgb(254, 240, 0);
      colourTable[743] = Color.FromArgb(254, 240, 0);
      colourTable[742] = Color.FromArgb(254, 241, 0);
      colourTable[741] = Color.FromArgb(253, 242, 0);
      colourTable[740] = Color.FromArgb(253, 242, 0);
      colourTable[739] = Color.FromArgb(253, 243, 0);
      colourTable[738] = Color.FromArgb(253, 243, 0);
      colourTable[737] = Color.FromArgb(253, 244, 0);
      colourTable[736] = Color.FromArgb(253, 245, 0);
      colourTable[735] = Color.FromArgb(253, 245, 0);
      colourTable[734] = Color.FromArgb(252, 246, 0);
      colourTable[733] = Color.FromArgb(252, 246, 0);
      colourTable[732] = Color.FromArgb(252, 247, 0);
      colourTable[731] = Color.FromArgb(252, 247, 0);
      colourTable[730] = Color.FromArgb(252, 248, 0);
      colourTable[729] = Color.FromArgb(252, 248, 0);
      colourTable[728] = Color.FromArgb(252, 249, 0);
      colourTable[727] = Color.FromArgb(251, 249, 0);
      colourTable[726] = Color.FromArgb(251, 250, 0);
      colourTable[725] = Color.FromArgb(251, 250, 0);
      colourTable[724] = Color.FromArgb(251, 250, 0);
      colourTable[723] = Color.FromArgb(251, 251, 0);
      colourTable[722] = Color.FromArgb(250, 251, 0);
      colourTable[721] = Color.FromArgb(250, 251, 0);
      colourTable[720] = Color.FromArgb(250, 252, 0);
      colourTable[719] = Color.FromArgb(250, 252, 0);
      colourTable[718] = Color.FromArgb(250, 252, 0);
      colourTable[717] = Color.FromArgb(249, 252, 0);
      colourTable[716] = Color.FromArgb(249, 253, 0);
      colourTable[715] = Color.FromArgb(249, 253, 0);
      colourTable[714] = Color.FromArgb(249, 253, 0);
      colourTable[713] = Color.FromArgb(249, 253, 0);
      colourTable[712] = Color.FromArgb(248, 253, 0);
      colourTable[711] = Color.FromArgb(248, 254, 0);
      colourTable[710] = Color.FromArgb(248, 254, 0);
      colourTable[709] = Color.FromArgb(248, 254, 0);
      colourTable[708] = Color.FromArgb(248, 254, 0);
      colourTable[707] = Color.FromArgb(247, 254, 0);
      colourTable[706] = Color.FromArgb(247, 254, 0);
      colourTable[705] = Color.FromArgb(247, 254, 0);
      colourTable[704] = Color.FromArgb(247, 254, 0);
      colourTable[703] = Color.FromArgb(246, 254, 0);
      colourTable[702] = Color.FromArgb(246, 254, 0);
      colourTable[701] = Color.FromArgb(246, 255, 0);
      colourTable[700] = Color.FromArgb(246, 255, 0);
      colourTable[699] = Color.FromArgb(245, 255, 0);
      colourTable[698] = Color.FromArgb(245, 255, 0);
      colourTable[697] = Color.FromArgb(245, 255, 0);
      colourTable[696] = Color.FromArgb(245, 255, 0);
      colourTable[695] = Color.FromArgb(244, 255, 0);
      colourTable[694] = Color.FromArgb(244, 255, 0);
      colourTable[693] = Color.FromArgb(244, 255, 0);
      colourTable[692] = Color.FromArgb(244, 255, 0);
      colourTable[691] = Color.FromArgb(243, 255, 0);
      colourTable[690] = Color.FromArgb(243, 255, 0);
      colourTable[689] = Color.FromArgb(243, 255, 0);
      colourTable[688] = Color.FromArgb(243, 255, 0);
      colourTable[687] = Color.FromArgb(242, 255, 0);
      colourTable[686] = Color.FromArgb(242, 255, 0);
      colourTable[685] = Color.FromArgb(242, 255, 0);
      colourTable[684] = Color.FromArgb(242, 255, 0);
      colourTable[683] = Color.FromArgb(241, 255, 0);
      colourTable[682] = Color.FromArgb(241, 255, 0);
      colourTable[681] = Color.FromArgb(241, 255, 0);
      colourTable[680] = Color.FromArgb(240, 255, 0);
      colourTable[679] = Color.FromArgb(240, 255, 0);
      colourTable[678] = Color.FromArgb(240, 255, 0);
      colourTable[677] = Color.FromArgb(240, 255, 0);
      colourTable[676] = Color.FromArgb(239, 255, 0);
      colourTable[675] = Color.FromArgb(239, 255, 0);
      colourTable[674] = Color.FromArgb(239, 255, 0);
      colourTable[673] = Color.FromArgb(238, 255, 0);
      colourTable[672] = Color.FromArgb(238, 255, 0);
      colourTable[671] = Color.FromArgb(238, 255, 0);
      colourTable[670] = Color.FromArgb(238, 255, 0);
      colourTable[669] = Color.FromArgb(237, 255, 0);
      colourTable[668] = Color.FromArgb(237, 255, 0);
      colourTable[667] = Color.FromArgb(237, 255, 0);
      colourTable[666] = Color.FromArgb(236, 255, 0);
      colourTable[665] = Color.FromArgb(236, 255, 0);
      colourTable[664] = Color.FromArgb(236, 255, 0);
      colourTable[663] = Color.FromArgb(235, 255, 0);
      colourTable[662] = Color.FromArgb(234, 255, 0);
      colourTable[661] = Color.FromArgb(234, 255, 0);
      colourTable[660] = Color.FromArgb(233, 255, 0);
      colourTable[659] = Color.FromArgb(232, 255, 0);
      colourTable[658] = Color.FromArgb(231, 255, 0);
      colourTable[657] = Color.FromArgb(231, 255, 0);
      colourTable[656] = Color.FromArgb(230, 255, 0);
      colourTable[655] = Color.FromArgb(229, 255, 0);
      colourTable[654] = Color.FromArgb(228, 255, 0);
      colourTable[653] = Color.FromArgb(227, 255, 0);
      colourTable[652] = Color.FromArgb(226, 255, 0);
      colourTable[651] = Color.FromArgb(225, 255, 0);
      colourTable[650] = Color.FromArgb(224, 255, 0);
      colourTable[649] = Color.FromArgb(223, 255, 0);
      colourTable[648] = Color.FromArgb(222, 255, 0);
      colourTable[647] = Color.FromArgb(221, 255, 0);
      colourTable[646] = Color.FromArgb(220, 255, 0);
      colourTable[645] = Color.FromArgb(219, 255, 0);
      colourTable[644] = Color.FromArgb(218, 255, 0);
      colourTable[643] = Color.FromArgb(217, 255, 0);
      colourTable[642] = Color.FromArgb(216, 255, 0);
      colourTable[641] = Color.FromArgb(215, 255, 0);
      colourTable[640] = Color.FromArgb(213, 255, 0);
      colourTable[639] = Color.FromArgb(212, 255, 0);
      colourTable[638] = Color.FromArgb(211, 255, 0);
      colourTable[637] = Color.FromArgb(209, 255, 0);
      colourTable[636] = Color.FromArgb(208, 255, 0);
      colourTable[635] = Color.FromArgb(206, 255, 0);
      colourTable[634] = Color.FromArgb(205, 255, 0);
      colourTable[633] = Color.FromArgb(203, 255, 0);
      colourTable[632] = Color.FromArgb(202, 255, 0);
      colourTable[631] = Color.FromArgb(200, 255, 0);
      colourTable[630] = Color.FromArgb(199, 255, 0);
      colourTable[629] = Color.FromArgb(197, 255, 0);
      colourTable[628] = Color.FromArgb(196, 255, 0);
      colourTable[627] = Color.FromArgb(194, 255, 0);
      colourTable[626] = Color.FromArgb(193, 255, 0);
      colourTable[625] = Color.FromArgb(192, 255, 0);
      colourTable[624] = Color.FromArgb(190, 255, 0);
      colourTable[623] = Color.FromArgb(189, 255, 0);
      colourTable[622] = Color.FromArgb(187, 255, 0);
      colourTable[621] = Color.FromArgb(186, 255, 0);
      colourTable[620] = Color.FromArgb(184, 255, 0);
      colourTable[619] = Color.FromArgb(182, 255, 0);
      colourTable[618] = Color.FromArgb(181, 255, 0);
      colourTable[617] = Color.FromArgb(179, 255, 0);
      colourTable[616] = Color.FromArgb(177, 255, 0);
      colourTable[615] = Color.FromArgb(176, 255, 0);
      colourTable[614] = Color.FromArgb(174, 255, 0);
      colourTable[613] = Color.FromArgb(172, 255, 0);
      colourTable[612] = Color.FromArgb(170, 255, 0);
      colourTable[611] = Color.FromArgb(168, 255, 0);
      colourTable[610] = Color.FromArgb(166, 255, 0);
      colourTable[609] = Color.FromArgb(164, 255, 0);
      colourTable[608] = Color.FromArgb(162, 255, 0);
      colourTable[607] = Color.FromArgb(160, 255, 0);
      colourTable[606] = Color.FromArgb(158, 255, 0);
      colourTable[605] = Color.FromArgb(156, 255, 0);
      colourTable[604] = Color.FromArgb(154, 255, 0);
      colourTable[603] = Color.FromArgb(152, 255, 0);
      colourTable[602] = Color.FromArgb(150, 255, 0);
      colourTable[601] = Color.FromArgb(148, 255, 0);
      colourTable[600] = Color.FromArgb(146, 255, 0);
      colourTable[599] = Color.FromArgb(144, 255, 0);
      colourTable[598] = Color.FromArgb(142, 255, 0);
      colourTable[597] = Color.FromArgb(140, 255, 0);
      colourTable[596] = Color.FromArgb(138, 255, 0);
      colourTable[595] = Color.FromArgb(136, 255, 0);
      colourTable[594] = Color.FromArgb(133, 255, 0);
      colourTable[593] = Color.FromArgb(131, 255, 0);
      colourTable[592] = Color.FromArgb(129, 255, 0);
      colourTable[591] = Color.FromArgb(127, 255, 0);
      colourTable[590] = Color.FromArgb(125, 255, 0);
      colourTable[589] = Color.FromArgb(123, 255, 0);
      colourTable[588] = Color.FromArgb(121, 255, 0);
      colourTable[587] = Color.FromArgb(119, 255, 0);
      colourTable[586] = Color.FromArgb(117, 255, 0);
      colourTable[585] = Color.FromArgb(114, 255, 0);
      colourTable[584] = Color.FromArgb(112, 255, 0);
      colourTable[583] = Color.FromArgb(110, 255, 0);
      colourTable[582] = Color.FromArgb(108, 255, 0);
      colourTable[581] = Color.FromArgb(106, 255, 0);
      colourTable[580] = Color.FromArgb(104, 255, 0);
      colourTable[579] = Color.FromArgb(102, 255, 0);
      colourTable[578] = Color.FromArgb(100, 255, 0);
      colourTable[577] = Color.FromArgb(98, 255, 0);
      colourTable[576] = Color.FromArgb(97, 255, 0);
      colourTable[575] = Color.FromArgb(95, 255, 0);
      colourTable[574] = Color.FromArgb(93, 255, 0);
      colourTable[573] = Color.FromArgb(91, 255, 0);
      colourTable[572] = Color.FromArgb(89, 255, 0);
      colourTable[571] = Color.FromArgb(87, 255, 0);
      colourTable[570] = Color.FromArgb(86, 255, 0);
      colourTable[569] = Color.FromArgb(84, 255, 0);
      colourTable[568] = Color.FromArgb(82, 255, 0);
      colourTable[567] = Color.FromArgb(80, 255, 0);
      colourTable[566] = Color.FromArgb(79, 255, 0);
      colourTable[565] = Color.FromArgb(77, 255, 0);
      colourTable[564] = Color.FromArgb(75, 255, 0);
      colourTable[563] = Color.FromArgb(73, 255, 0);
      colourTable[562] = Color.FromArgb(70, 255, 0);
      colourTable[561] = Color.FromArgb(68, 255, 0);
      colourTable[560] = Color.FromArgb(65, 255, 0);
      colourTable[559] = Color.FromArgb(63, 255, 0);
      colourTable[558] = Color.FromArgb(60, 255, 0);
      colourTable[557] = Color.FromArgb(58, 255, 0);
      colourTable[556] = Color.FromArgb(55, 255, 0);
      colourTable[555] = Color.FromArgb(53, 255, 0);
      colourTable[554] = Color.FromArgb(50, 255, 0);
      colourTable[553] = Color.FromArgb(48, 255, 0);
      colourTable[552] = Color.FromArgb(45, 255, 0);
      colourTable[551] = Color.FromArgb(43, 255, 0);
      colourTable[550] = Color.FromArgb(41, 255, 0);
      colourTable[549] = Color.FromArgb(39, 255, 0);
      colourTable[548] = Color.FromArgb(36, 255, 0);
      colourTable[547] = Color.FromArgb(34, 255, 0);
      colourTable[546] = Color.FromArgb(32, 255, 0);
      colourTable[545] = Color.FromArgb(30, 255, 0);
      colourTable[544] = Color.FromArgb(28, 255, 0);
      colourTable[543] = Color.FromArgb(26, 255, 0);
      colourTable[542] = Color.FromArgb(24, 255, 0);
      colourTable[541] = Color.FromArgb(22, 255, 0);
      colourTable[540] = Color.FromArgb(20, 255, 0);
      colourTable[539] = Color.FromArgb(18, 255, 0);
      colourTable[538] = Color.FromArgb(16, 255, 0);
      colourTable[537] = Color.FromArgb(15, 255, 0);
      colourTable[536] = Color.FromArgb(13, 255, 0);
      colourTable[535] = Color.FromArgb(11, 255, 0);
      colourTable[534] = Color.FromArgb(10, 255, 0);
      colourTable[533] = Color.FromArgb(8, 255, 0);
      colourTable[532] = Color.FromArgb(7, 255, 0);
      colourTable[531] = Color.FromArgb(5, 255, 0);
      colourTable[530] = Color.FromArgb(4, 255, 0);
      colourTable[529] = Color.FromArgb(3, 255, 0);
      colourTable[528] = Color.FromArgb(2, 255, 0);
      colourTable[527] = Color.FromArgb(1, 255, 4);
      colourTable[526] = Color.FromArgb(0, 255, 9);
      colourTable[525] = Color.FromArgb(0, 255, 13);
      colourTable[524] = Color.FromArgb(0, 255, 17);
      colourTable[523] = Color.FromArgb(0, 255, 21);
      colourTable[522] = Color.FromArgb(0, 255, 25);
      colourTable[521] = Color.FromArgb(0, 255, 30);
      colourTable[520] = Color.FromArgb(0, 255, 34);
      colourTable[519] = Color.FromArgb(0, 255, 38);
      colourTable[518] = Color.FromArgb(0, 255, 42);
      colourTable[517] = Color.FromArgb(0, 255, 46);
      colourTable[516] = Color.FromArgb(0, 255, 50);
      colourTable[515] = Color.FromArgb(0, 255, 55);
      colourTable[514] = Color.FromArgb(0, 255, 59);
      colourTable[513] = Color.FromArgb(0, 255, 63);
      colourTable[512] = Color.FromArgb(0, 255, 67);
      colourTable[511] = Color.FromArgb(0, 255, 71);
      colourTable[510] = Color.FromArgb(0, 255, 76);
      colourTable[509] = Color.FromArgb(0, 255, 80);
      colourTable[508] = Color.FromArgb(0, 255, 84);
      colourTable[507] = Color.FromArgb(0, 255, 88);
      colourTable[506] = Color.FromArgb(0, 255, 92);
      colourTable[505] = Color.FromArgb(0, 255, 97);
      colourTable[504] = Color.FromArgb(0, 255, 101);
      colourTable[503] = Color.FromArgb(0, 255, 105);
      colourTable[502] = Color.FromArgb(0, 255, 109);
      colourTable[501] = Color.FromArgb(0, 255, 113);
      colourTable[500] = Color.FromArgb(0, 255, 118);
      colourTable[499] = Color.FromArgb(0, 255, 122);
      colourTable[498] = Color.FromArgb(0, 255, 126);
      colourTable[497] = Color.FromArgb(0, 255, 130);
      colourTable[496] = Color.FromArgb(0, 255, 134);
      colourTable[495] = Color.FromArgb(0, 255, 138);
      colourTable[494] = Color.FromArgb(0, 255, 142);
      colourTable[493] = Color.FromArgb(0, 255, 145);
      colourTable[492] = Color.FromArgb(0, 255, 148);
      colourTable[491] = Color.FromArgb(0, 255, 152);
      colourTable[490] = Color.FromArgb(0, 255, 155);
      colourTable[489] = Color.FromArgb(0, 255, 158);
      colourTable[488] = Color.FromArgb(0, 255, 160);
      colourTable[487] = Color.FromArgb(0, 255, 163);
      colourTable[486] = Color.FromArgb(0, 255, 166);
      colourTable[485] = Color.FromArgb(0, 255, 169);
      colourTable[484] = Color.FromArgb(0, 255, 172);
      colourTable[483] = Color.FromArgb(0, 255, 174);
      colourTable[482] = Color.FromArgb(1, 255, 177);
      colourTable[481] = Color.FromArgb(1, 255, 179);
      colourTable[480] = Color.FromArgb(2, 255, 182);
      colourTable[479] = Color.FromArgb(2, 255, 184);
      colourTable[478] = Color.FromArgb(3, 255, 186);
      colourTable[477] = Color.FromArgb(3, 255, 189);
      colourTable[476] = Color.FromArgb(4, 255, 191);
      colourTable[475] = Color.FromArgb(4, 255, 193);
      colourTable[474] = Color.FromArgb(5, 255, 195);
      colourTable[473] = Color.FromArgb(6, 254, 197);
      colourTable[472] = Color.FromArgb(6, 254, 199);
      colourTable[471] = Color.FromArgb(7, 254, 201);
      colourTable[470] = Color.FromArgb(8, 254, 203);
      colourTable[469] = Color.FromArgb(9, 253, 205);
      colourTable[468] = Color.FromArgb(9, 253, 207);
      colourTable[467] = Color.FromArgb(10, 253, 209);
      colourTable[466] = Color.FromArgb(11, 252, 211);
      colourTable[465] = Color.FromArgb(12, 252, 213);
      colourTable[464] = Color.FromArgb(13, 252, 215);
      colourTable[463] = Color.FromArgb(14, 251, 216);
      colourTable[462] = Color.FromArgb(14, 251, 218);
      colourTable[461] = Color.FromArgb(15, 251, 219);
      colourTable[460] = Color.FromArgb(16, 250, 220);
      colourTable[459] = Color.FromArgb(17, 250, 222);
      colourTable[458] = Color.FromArgb(18, 249, 223);
      colourTable[457] = Color.FromArgb(19, 249, 224);
      colourTable[456] = Color.FromArgb(20, 249, 225);
      colourTable[455] = Color.FromArgb(21, 248, 227);
      colourTable[454] = Color.FromArgb(22, 248, 228);
      colourTable[453] = Color.FromArgb(23, 247, 229);
      colourTable[452] = Color.FromArgb(24, 247, 230);
      colourTable[451] = Color.FromArgb(24, 246, 231);
      colourTable[450] = Color.FromArgb(25, 246, 232);
      colourTable[449] = Color.FromArgb(26, 246, 233);
      colourTable[448] = Color.FromArgb(27, 245, 234);
      colourTable[447] = Color.FromArgb(28, 245, 235);
      colourTable[446] = Color.FromArgb(29, 244, 236);
      colourTable[445] = Color.FromArgb(30, 244, 237);
      colourTable[444] = Color.FromArgb(31, 243, 238);
      colourTable[443] = Color.FromArgb(32, 243, 239);
      colourTable[442] = Color.FromArgb(33, 242, 240);
      colourTable[441] = Color.FromArgb(34, 242, 241);
      colourTable[440] = Color.FromArgb(35, 241, 242);
      colourTable[439] = Color.FromArgb(35, 241, 243);
      colourTable[438] = Color.FromArgb(36, 240, 244);
      colourTable[437] = Color.FromArgb(37, 240, 245);
      colourTable[436] = Color.FromArgb(38, 239, 245);
      colourTable[435] = Color.FromArgb(39, 239, 246);
      colourTable[434] = Color.FromArgb(40, 238, 247);
      colourTable[433] = Color.FromArgb(40, 238, 248);
      colourTable[432] = Color.FromArgb(41, 237, 248);
      colourTable[431] = Color.FromArgb(42, 237, 249);
      colourTable[430] = Color.FromArgb(43, 236, 250);
      colourTable[429] = Color.FromArgb(44, 235, 250);
      colourTable[428] = Color.FromArgb(46, 235, 251);
      colourTable[427] = Color.FromArgb(47, 234, 252);
      colourTable[426] = Color.FromArgb(48, 234, 252);
      colourTable[425] = Color.FromArgb(49, 233, 253);
      colourTable[424] = Color.FromArgb(51, 233, 253);
      colourTable[423] = Color.FromArgb(52, 232, 254);
      colourTable[422] = Color.FromArgb(53, 232, 254);
      colourTable[421] = Color.FromArgb(55, 231, 255);
      colourTable[420] = Color.FromArgb(56, 231, 255);
      colourTable[419] = Color.FromArgb(57, 230, 255);
      colourTable[418] = Color.FromArgb(58, 230, 255);
      colourTable[417] = Color.FromArgb(60, 229, 255);
      colourTable[416] = Color.FromArgb(61, 229, 255);
      colourTable[415] = Color.FromArgb(62, 228, 255);
      colourTable[414] = Color.FromArgb(63, 228, 255);
      colourTable[413] = Color.FromArgb(65, 227, 255);
      colourTable[412] = Color.FromArgb(66, 226, 255);
      colourTable[411] = Color.FromArgb(67, 226, 255);
      colourTable[410] = Color.FromArgb(68, 225, 255);
      colourTable[409] = Color.FromArgb(69, 225, 255);
      colourTable[408] = Color.FromArgb(70, 224, 255);
      colourTable[407] = Color.FromArgb(72, 224, 255);
      colourTable[406] = Color.FromArgb(73, 223, 255);
      colourTable[405] = Color.FromArgb(74, 223, 255);
      colourTable[404] = Color.FromArgb(75, 222, 255);
      colourTable[403] = Color.FromArgb(76, 222, 255);
      colourTable[402] = Color.FromArgb(77, 221, 255);
      colourTable[401] = Color.FromArgb(78, 220, 255);
      colourTable[400] = Color.FromArgb(78, 219, 255);
      colourTable[399] = Color.FromArgb(79, 218, 255);
      colourTable[398] = Color.FromArgb(80, 217, 255);
      colourTable[397] = Color.FromArgb(81, 216, 255);
      colourTable[396] = Color.FromArgb(81, 215, 255);
      colourTable[395] = Color.FromArgb(82, 214, 255);
      colourTable[394] = Color.FromArgb(83, 213, 255);
      colourTable[393] = Color.FromArgb(83, 212, 255);
      colourTable[392] = Color.FromArgb(84, 211, 255);
      colourTable[391] = Color.FromArgb(85, 210, 255);
      colourTable[390] = Color.FromArgb(85, 209, 255);
      colourTable[389] = Color.FromArgb(86, 208, 255);
      colourTable[388] = Color.FromArgb(87, 207, 255);
      colourTable[387] = Color.FromArgb(87, 206, 255);
      colourTable[386] = Color.FromArgb(88, 205, 255);
      colourTable[385] = Color.FromArgb(88, 204, 255);
      colourTable[384] = Color.FromArgb(89, 203, 255);
      colourTable[383] = Color.FromArgb(89, 202, 255);
      colourTable[382] = Color.FromArgb(90, 201, 255);
      colourTable[381] = Color.FromArgb(90, 200, 255);
      colourTable[380] = Color.FromArgb(91, 199, 255);
      colourTable[379] = Color.FromArgb(91, 198, 255);
      colourTable[378] = Color.FromArgb(92, 197, 255);
      colourTable[377] = Color.FromArgb(92, 196, 255);
      colourTable[376] = Color.FromArgb(93, 195, 255);
      colourTable[375] = Color.FromArgb(93, 194, 255);
      colourTable[374] = Color.FromArgb(93, 193, 255);
      colourTable[373] = Color.FromArgb(94, 192, 255);
      colourTable[372] = Color.FromArgb(94, 191, 255);
      colourTable[371] = Color.FromArgb(94, 190, 255);
      colourTable[370] = Color.FromArgb(95, 189, 255);
      colourTable[369] = Color.FromArgb(95, 188, 255);
      colourTable[368] = Color.FromArgb(95, 187, 255);
      colourTable[367] = Color.FromArgb(95, 186, 255);
      colourTable[366] = Color.FromArgb(96, 185, 255);
      colourTable[365] = Color.FromArgb(96, 183, 255);
      colourTable[364] = Color.FromArgb(96, 182, 255);
      colourTable[363] = Color.FromArgb(96, 181, 255);
      colourTable[362] = Color.FromArgb(96, 180, 255);
      colourTable[361] = Color.FromArgb(96, 179, 255);
      colourTable[360] = Color.FromArgb(95, 178, 255);
      colourTable[359] = Color.FromArgb(95, 177, 255);
      colourTable[358] = Color.FromArgb(95, 176, 255);
      colourTable[357] = Color.FromArgb(95, 175, 255);
      colourTable[356] = Color.FromArgb(94, 174, 255);
      colourTable[355] = Color.FromArgb(94, 173, 255);
      colourTable[354] = Color.FromArgb(94, 172, 255);
      colourTable[353] = Color.FromArgb(94, 171, 255);
      colourTable[352] = Color.FromArgb(94, 170, 255);
      colourTable[351] = Color.FromArgb(93, 169, 255);
      colourTable[350] = Color.FromArgb(93, 168, 255);
      colourTable[349] = Color.FromArgb(93, 167, 255);
      colourTable[348] = Color.FromArgb(93, 166, 255);
      colourTable[347] = Color.FromArgb(92, 164, 255);
      colourTable[346] = Color.FromArgb(92, 163, 255);
      colourTable[345] = Color.FromArgb(92, 162, 255);
      colourTable[344] = Color.FromArgb(92, 161, 255);
      colourTable[343] = Color.FromArgb(91, 159, 255);
      colourTable[342] = Color.FromArgb(91, 158, 255);
      colourTable[341] = Color.FromArgb(91, 157, 255);
      colourTable[340] = Color.FromArgb(91, 155, 255);
      colourTable[339] = Color.FromArgb(90, 154, 255);
      colourTable[338] = Color.FromArgb(90, 152, 255);
      colourTable[337] = Color.FromArgb(90, 151, 255);
      colourTable[336] = Color.FromArgb(90, 150, 255);
      colourTable[335] = Color.FromArgb(90, 149, 255);
      colourTable[334] = Color.FromArgb(89, 148, 255);
      colourTable[333] = Color.FromArgb(89, 148, 255);
      colourTable[332] = Color.FromArgb(89, 147, 255);
      colourTable[331] = Color.FromArgb(89, 147, 255);
      colourTable[330] = Color.FromArgb(88, 146, 255);
      colourTable[329] = Color.FromArgb(88, 146, 255);
      colourTable[328] = Color.FromArgb(88, 145, 255);
      colourTable[327] = Color.FromArgb(87, 144, 255);
      colourTable[326] = Color.FromArgb(87, 144, 255);
      colourTable[325] = Color.FromArgb(86, 143, 255);
      colourTable[324] = Color.FromArgb(85, 143, 255);
      colourTable[323] = Color.FromArgb(85, 142, 255);
      colourTable[322] = Color.FromArgb(84, 141, 255);
      colourTable[321] = Color.FromArgb(83, 141, 255);
      colourTable[320] = Color.FromArgb(82, 140, 255);
      colourTable[319] = Color.FromArgb(82, 140, 255);
      colourTable[318] = Color.FromArgb(81, 139, 255);
      colourTable[317] = Color.FromArgb(80, 138, 255);
      colourTable[316] = Color.FromArgb(79, 138, 255);
      colourTable[315] = Color.FromArgb(78, 137, 255);
      colourTable[314] = Color.FromArgb(77, 136, 255);
      colourTable[313] = Color.FromArgb(76, 136, 255);
      colourTable[312] = Color.FromArgb(75, 135, 255);
      colourTable[311] = Color.FromArgb(75, 134, 255);
      colourTable[310] = Color.FromArgb(74, 134, 255);
      colourTable[309] = Color.FromArgb(73, 133, 255);
      colourTable[308] = Color.FromArgb(72, 132, 255);
      colourTable[307] = Color.FromArgb(71, 132, 255);
      colourTable[306] = Color.FromArgb(70, 131, 255);
      colourTable[305] = Color.FromArgb(69, 130, 255);
      colourTable[304] = Color.FromArgb(68, 130, 255);
      colourTable[303] = Color.FromArgb(67, 129, 255);
      colourTable[302] = Color.FromArgb(66, 128, 255);
      colourTable[301] = Color.FromArgb(64, 127, 255);
      colourTable[300] = Color.FromArgb(63, 127, 255);
      colourTable[299] = Color.FromArgb(62, 126, 255);
      colourTable[298] = Color.FromArgb(61, 125, 255);
      colourTable[297] = Color.FromArgb(60, 125, 255);
      colourTable[296] = Color.FromArgb(59, 124, 255);
      colourTable[295] = Color.FromArgb(58, 123, 255);
      colourTable[294] = Color.FromArgb(57, 122, 255);
      colourTable[293] = Color.FromArgb(56, 122, 255);
      colourTable[292] = Color.FromArgb(55, 121, 255);
      colourTable[291] = Color.FromArgb(53, 120, 255);
      colourTable[290] = Color.FromArgb(51, 120, 255);
      colourTable[289] = Color.FromArgb(49, 119, 255);
      colourTable[288] = Color.FromArgb(47, 118, 255);
      colourTable[287] = Color.FromArgb(45, 117, 255);
      colourTable[286] = Color.FromArgb(42, 117, 255);
      colourTable[285] = Color.FromArgb(40, 116, 255);
      colourTable[284] = Color.FromArgb(38, 115, 255);
      colourTable[283] = Color.FromArgb(35, 114, 255);
      colourTable[282] = Color.FromArgb(33, 114, 255);
      colourTable[281] = Color.FromArgb(30, 113, 255);
      colourTable[280] = Color.FromArgb(28, 112, 255);
      colourTable[279] = Color.FromArgb(26, 111, 255);
      colourTable[278] = Color.FromArgb(24, 111, 255);
      colourTable[277] = Color.FromArgb(21, 110, 255);
      colourTable[276] = Color.FromArgb(19, 109, 255);
      colourTable[275] = Color.FromArgb(17, 108, 255);
      colourTable[274] = Color.FromArgb(15, 107, 255);
      colourTable[273] = Color.FromArgb(14, 107, 255);
      colourTable[272] = Color.FromArgb(12, 106, 255);
      colourTable[271] = Color.FromArgb(11, 105, 255);
      colourTable[270] = Color.FromArgb(9, 104, 255);
      colourTable[269] = Color.FromArgb(8, 104, 255);
      colourTable[268] = Color.FromArgb(8, 103, 255);
      colourTable[267] = Color.FromArgb(8, 102, 255);
      colourTable[266] = Color.FromArgb(8, 101, 255);
      colourTable[265] = Color.FromArgb(8, 100, 255);
      colourTable[264] = Color.FromArgb(8, 100, 255);
      colourTable[263] = Color.FromArgb(8, 99, 255);
      colourTable[262] = Color.FromArgb(8, 98, 255);
      colourTable[261] = Color.FromArgb(8, 97, 255);
      colourTable[260] = Color.FromArgb(8, 96, 255);
      colourTable[259] = Color.FromArgb(8, 96, 255);
      colourTable[258] = Color.FromArgb(8, 95, 255);
      colourTable[257] = Color.FromArgb(8, 94, 255);
      colourTable[256] = Color.FromArgb(8, 93, 255);
      colourTable[255] = Color.FromArgb(8, 93, 255);
      colourTable[254] = Color.FromArgb(8, 92, 255);
      colourTable[253] = Color.FromArgb(8, 91, 255);
      colourTable[252] = Color.FromArgb(8, 90, 255);
      colourTable[251] = Color.FromArgb(8, 89, 255);
      colourTable[250] = Color.FromArgb(8, 89, 255);
      colourTable[249] = Color.FromArgb(8, 88, 255);
      colourTable[248] = Color.FromArgb(8, 87, 255);
      colourTable[247] = Color.FromArgb(8, 86, 255);
      colourTable[246] = Color.FromArgb(8, 85, 255);
      colourTable[245] = Color.FromArgb(8, 85, 255);
      colourTable[244] = Color.FromArgb(8, 84, 255);
      colourTable[243] = Color.FromArgb(8, 83, 255);
      colourTable[242] = Color.FromArgb(8, 82, 255);
      colourTable[241] = Color.FromArgb(8, 80, 255);
      colourTable[240] = Color.FromArgb(8, 78, 255);
      colourTable[239] = Color.FromArgb(8, 75, 255);
      colourTable[238] = Color.FromArgb(8, 73, 255);
      colourTable[237] = Color.FromArgb(8, 71, 255);
      colourTable[236] = Color.FromArgb(8, 69, 255);
      colourTable[235] = Color.FromArgb(8, 67, 255);
      colourTable[234] = Color.FromArgb(8, 64, 255);
      colourTable[233] = Color.FromArgb(8, 62, 255);
      colourTable[232] = Color.FromArgb(8, 59, 255);
      colourTable[231] = Color.FromArgb(8, 57, 255);
      colourTable[230] = Color.FromArgb(8, 55, 255);
      colourTable[229] = Color.FromArgb(8, 52, 255);
      colourTable[228] = Color.FromArgb(8, 50, 255);
      colourTable[227] = Color.FromArgb(8, 47, 255);
      colourTable[226] = Color.FromArgb(8, 45, 255);
      colourTable[225] = Color.FromArgb(8, 42, 255);
      colourTable[224] = Color.FromArgb(8, 40, 255);
      colourTable[223] = Color.FromArgb(8, 38, 255);
      colourTable[222] = Color.FromArgb(8, 35, 255);
      colourTable[221] = Color.FromArgb(8, 33, 255);
      colourTable[220] = Color.FromArgb(8, 31, 255);
      colourTable[219] = Color.FromArgb(8, 28, 255);
      colourTable[218] = Color.FromArgb(8, 26, 255);
      colourTable[217] = Color.FromArgb(8, 24, 255);
      colourTable[216] = Color.FromArgb(8, 22, 255);
      colourTable[215] = Color.FromArgb(8, 20, 255);
      colourTable[214] = Color.FromArgb(8, 17, 255);
      colourTable[213] = Color.FromArgb(8, 15, 255);
      colourTable[212] = Color.FromArgb(9, 14, 255);
      colourTable[211] = Color.FromArgb(9, 12, 255);
      colourTable[210] = Color.FromArgb(10, 10, 255);
      colourTable[209] = Color.FromArgb(11, 8, 255);
      colourTable[208] = Color.FromArgb(12, 7, 255);
      colourTable[207] = Color.FromArgb(13, 5, 255);
      colourTable[206] = Color.FromArgb(14, 4, 255);
      colourTable[205] = Color.FromArgb(15, 2, 255);
      colourTable[204] = Color.FromArgb(16, 1, 255);
      colourTable[203] = Color.FromArgb(17, 0, 255);
      colourTable[202] = Color.FromArgb(18, 0, 255);
      colourTable[201] = Color.FromArgb(19, 0, 255);
      colourTable[200] = Color.FromArgb(21, 0, 255);
      colourTable[199] = Color.FromArgb(22, 0, 255);
      colourTable[198] = Color.FromArgb(23, 0, 255);
      colourTable[197] = Color.FromArgb(24, 0, 255);
      colourTable[196] = Color.FromArgb(26, 0, 255);
      colourTable[195] = Color.FromArgb(27, 0, 255);
      colourTable[194] = Color.FromArgb(29, 0, 255);
      colourTable[193] = Color.FromArgb(30, 0, 255);
      colourTable[192] = Color.FromArgb(31, 0, 255);
      colourTable[191] = Color.FromArgb(33, 0, 255);
      colourTable[190] = Color.FromArgb(34, 0, 255);
      colourTable[189] = Color.FromArgb(35, 0, 255);
      colourTable[188] = Color.FromArgb(37, 0, 254);
      colourTable[187] = Color.FromArgb(38, 0, 254);
      colourTable[186] = Color.FromArgb(40, 0, 254);
      colourTable[185] = Color.FromArgb(41, 0, 253);
      colourTable[184] = Color.FromArgb(42, 0, 253);
      colourTable[183] = Color.FromArgb(44, 0, 253);
      colourTable[182] = Color.FromArgb(45, 0, 252);
      colourTable[181] = Color.FromArgb(46, 0, 252);
      colourTable[180] = Color.FromArgb(47, 0, 252);
      colourTable[179] = Color.FromArgb(49, 0, 251);
      colourTable[178] = Color.FromArgb(50, 0, 251);
      colourTable[177] = Color.FromArgb(51, 0, 250);
      colourTable[176] = Color.FromArgb(52, 0, 250);
      colourTable[175] = Color.FromArgb(53, 0, 249);
      colourTable[174] = Color.FromArgb(54, 0, 249);
      colourTable[173] = Color.FromArgb(55, 0, 248);
      colourTable[172] = Color.FromArgb(56, 0, 248);
      colourTable[171] = Color.FromArgb(57, 0, 247);
      colourTable[170] = Color.FromArgb(58, 0, 246);
      colourTable[169] = Color.FromArgb(59, 0, 246);
      colourTable[168] = Color.FromArgb(60, 0, 245);
      colourTable[167] = Color.FromArgb(61, 0, 245);
      colourTable[166] = Color.FromArgb(62, 0, 244);
      colourTable[165] = Color.FromArgb(63, 0, 244);
      colourTable[164] = Color.FromArgb(63, 0, 243);
      colourTable[163] = Color.FromArgb(64, 0, 242);
      colourTable[162] = Color.FromArgb(65, 0, 242);
      colourTable[161] = Color.FromArgb(66, 0, 241);
      colourTable[160] = Color.FromArgb(67, 0, 240);
      colourTable[159] = Color.FromArgb(68, 0, 240);
      colourTable[158] = Color.FromArgb(68, 0, 239);
      colourTable[157] = Color.FromArgb(69, 0, 238);
      colourTable[156] = Color.FromArgb(70, 0, 238);
      colourTable[155] = Color.FromArgb(71, 0, 237);
      colourTable[154] = Color.FromArgb(71, 0, 236);
      colourTable[153] = Color.FromArgb(72, 0, 236);
      colourTable[152] = Color.FromArgb(73, 0, 235);
      colourTable[151] = Color.FromArgb(74, 0, 234);
      colourTable[150] = Color.FromArgb(75, 0, 234);
      colourTable[149] = Color.FromArgb(75, 0, 233);
      colourTable[148] = Color.FromArgb(76, 0, 232);
      colourTable[147] = Color.FromArgb(77, 0, 232);
      colourTable[146] = Color.FromArgb(78, 0, 231);
      colourTable[145] = Color.FromArgb(79, 0, 230);
      colourTable[144] = Color.FromArgb(79, 0, 229);
      colourTable[143] = Color.FromArgb(80, 0, 229);
      colourTable[142] = Color.FromArgb(81, 0, 228);
      colourTable[141] = Color.FromArgb(82, 0, 227);
      colourTable[140] = Color.FromArgb(83, 0, 227);
      colourTable[139] = Color.FromArgb(84, 0, 226);
      colourTable[138] = Color.FromArgb(85, 0, 225);
      colourTable[137] = Color.FromArgb(86, 0, 225);
      colourTable[136] = Color.FromArgb(87, 0, 224);
      colourTable[135] = Color.FromArgb(88, 0, 223);
      colourTable[134] = Color.FromArgb(89, 0, 222);
      colourTable[133] = Color.FromArgb(90, 0, 222);
      colourTable[132] = Color.FromArgb(91, 0, 221);
      colourTable[131] = Color.FromArgb(92, 0, 220);
      colourTable[130] = Color.FromArgb(93, 0, 220);
      colourTable[129] = Color.FromArgb(94, 0, 219);
      colourTable[128] = Color.FromArgb(95, 0, 218);
      colourTable[127] = Color.FromArgb(96, 0, 218);
      colourTable[126] = Color.FromArgb(97, 0, 217);
      colourTable[125] = Color.FromArgb(98, 0, 217);
      colourTable[124] = Color.FromArgb(99, 0, 216);
      colourTable[123] = Color.FromArgb(100, 0, 215);
      colourTable[122] = Color.FromArgb(101, 0, 215);
      colourTable[121] = Color.FromArgb(102, 0, 214);
      colourTable[120] = Color.FromArgb(103, 0, 214);
      colourTable[119] = Color.FromArgb(104, 0, 213);
      colourTable[118] = Color.FromArgb(106, 0, 212);
      colourTable[117] = Color.FromArgb(107, 0, 212);
      colourTable[116] = Color.FromArgb(108, 0, 211);
      colourTable[115] = Color.FromArgb(109, 0, 211);
      colourTable[114] = Color.FromArgb(110, 0, 210);
      colourTable[113] = Color.FromArgb(111, 0, 210);
      colourTable[112] = Color.FromArgb(112, 0, 209);
      colourTable[111] = Color.FromArgb(114, 0, 209);
      colourTable[110] = Color.FromArgb(115, 0, 208);
      colourTable[109] = Color.FromArgb(116, 0, 208);
      colourTable[108] = Color.FromArgb(117, 0, 207);
      colourTable[107] = Color.FromArgb(118, 0, 207);
      colourTable[106] = Color.FromArgb(120, 0, 207);
      colourTable[105] = Color.FromArgb(121, 0, 206);
      colourTable[104] = Color.FromArgb(122, 0, 206);
      colourTable[103] = Color.FromArgb(124, 0, 205);
      colourTable[102] = Color.FromArgb(125, 0, 205);
      colourTable[101] = Color.FromArgb(127, 0, 205);
      colourTable[100] = Color.FromArgb(128, 0, 204);
      colourTable[99] = Color.FromArgb(129, 0, 204);
      colourTable[98] = Color.FromArgb(131, 0, 204);
      colourTable[97] = Color.FromArgb(132, 0, 204);
      colourTable[96] = Color.FromArgb(134, 0, 203);
      colourTable[95] = Color.FromArgb(135, 0, 203);
      colourTable[94] = Color.FromArgb(136, 0, 203);
      colourTable[93] = Color.FromArgb(138, 0, 203);
      colourTable[92] = Color.FromArgb(139, 0, 203);
      colourTable[91] = Color.FromArgb(141, 0, 203);
      colourTable[90] = Color.FromArgb(142, 0, 204);
      colourTable[89] = Color.FromArgb(143, 0, 204);
      colourTable[88] = Color.FromArgb(145, 0, 204);
      colourTable[87] = Color.FromArgb(146, 0, 204);
      colourTable[86] = Color.FromArgb(147, 0, 204);
      colourTable[85] = Color.FromArgb(149, 0, 204);
      colourTable[84] = Color.FromArgb(150, 0, 205);
      colourTable[83] = Color.FromArgb(151, 0, 205);
      colourTable[82] = Color.FromArgb(152, 0, 205);
      colourTable[81] = Color.FromArgb(153, 0, 205);
      colourTable[80] = Color.FromArgb(154, 0, 205);
      colourTable[79] = Color.FromArgb(155, 0, 206);
      colourTable[78] = Color.FromArgb(156, 0, 206);
      colourTable[77] = Color.FromArgb(157, 0, 206);
      colourTable[76] = Color.FromArgb(158, 0, 206);
      colourTable[75] = Color.FromArgb(159, 0, 206);
      colourTable[74] = Color.FromArgb(160, 0, 206);
      colourTable[73] = Color.FromArgb(161, 0, 207);
      colourTable[72] = Color.FromArgb(162, 0, 207);
      colourTable[71] = Color.FromArgb(162, 0, 207);
      colourTable[70] = Color.FromArgb(163, 0, 207);
      colourTable[69] = Color.FromArgb(164, 0, 207);
      colourTable[68] = Color.FromArgb(165, 0, 207);
      colourTable[67] = Color.FromArgb(166, 0, 208);
      colourTable[66] = Color.FromArgb(166, 0, 208);
      colourTable[65] = Color.FromArgb(167, 0, 208);
      colourTable[64] = Color.FromArgb(168, 0, 208);
      colourTable[63] = Color.FromArgb(169, 0, 208);
      colourTable[62] = Color.FromArgb(170, 0, 208);
      colourTable[61] = Color.FromArgb(170, 0, 209);
      colourTable[60] = Color.FromArgb(171, 0, 209);
      colourTable[59] = Color.FromArgb(172, 0, 209);
      colourTable[58] = Color.FromArgb(173, 0, 209);
      colourTable[57] = Color.FromArgb(173, 0, 209);
      colourTable[56] = Color.FromArgb(174, 0, 209);
      colourTable[55] = Color.FromArgb(175, 0, 210);
      colourTable[54] = Color.FromArgb(175, 0, 210);
      colourTable[53] = Color.FromArgb(176, 0, 210);
      colourTable[52] = Color.FromArgb(177, 0, 210);
      colourTable[51] = Color.FromArgb(177, 0, 210);
      colourTable[50] = Color.FromArgb(178, 0, 210);
      colourTable[49] = Color.FromArgb(179, 0, 211);
      colourTable[48] = Color.FromArgb(179, 0, 211);
      colourTable[47] = Color.FromArgb(180, 0, 211);
      colourTable[46] = Color.FromArgb(181, 0, 211);
      colourTable[45] = Color.FromArgb(181, 0, 211);
      colourTable[44] = Color.FromArgb(182, 0, 212);
      colourTable[43] = Color.FromArgb(182, 0, 212);
      colourTable[42] = Color.FromArgb(182, 0, 212);
      colourTable[41] = Color.FromArgb(183, 0, 212);
      colourTable[40] = Color.FromArgb(183, 0, 212);
      colourTable[39] = Color.FromArgb(184, 0, 212);
      colourTable[38] = Color.FromArgb(184, 0, 213);
      colourTable[37] = Color.FromArgb(185, 0, 213);
      colourTable[36] = Color.FromArgb(185, 0, 213);
      colourTable[35] = Color.FromArgb(185, 0, 213);
      colourTable[34] = Color.FromArgb(186, 0, 213);
      colourTable[33] = Color.FromArgb(186, 0, 213);
      colourTable[32] = Color.FromArgb(186, 0, 214);
      colourTable[31] = Color.FromArgb(187, 0, 214);
      colourTable[30] = Color.FromArgb(187, 0, 214);
      colourTable[29] = Color.FromArgb(188, 0, 214);
      colourTable[28] = Color.FromArgb(188, 0, 214);
      colourTable[27] = Color.FromArgb(188, 0, 214);
      colourTable[26] = Color.FromArgb(189, 0, 215);
      colourTable[25] = Color.FromArgb(189, 0, 215);
      colourTable[24] = Color.FromArgb(189, 0, 215);
      colourTable[23] = Color.FromArgb(190, 0, 215);
      colourTable[22] = Color.FromArgb(190, 0, 215);
      colourTable[21] = Color.FromArgb(190, 0, 215);
      colourTable[20] = Color.FromArgb(191, 0, 216);
      colourTable[19] = Color.FromArgb(191, 0, 216);
      colourTable[18] = Color.FromArgb(191, 0, 216);
      colourTable[17] = Color.FromArgb(192, 0, 216);
      colourTable[16] = Color.FromArgb(192, 0, 216);
      colourTable[15] = Color.FromArgb(192, 0, 217);
      colourTable[14] = Color.FromArgb(193, 0, 217);
      colourTable[13] = Color.FromArgb(193, 0, 217);
      colourTable[12] = Color.FromArgb(193, 0, 217);
      colourTable[11] = Color.FromArgb(194, 0, 217);
      colourTable[10] = Color.FromArgb(194, 0, 217);
      colourTable[9] = Color.FromArgb(194, 0, 218);
      colourTable[8] = Color.FromArgb(195, 0, 218);
      colourTable[7] = Color.FromArgb(195, 0, 218);
      colourTable[6] = Color.FromArgb(195, 0, 218);
      colourTable[5] = Color.FromArgb(196, 0, 218);
      colourTable[4] = Color.FromArgb(196, 0, 218);
      colourTable[3] = Color.FromArgb(196, 0, 219);
      colourTable[2] = Color.FromArgb(197, 0, 219);
      colourTable[1] = Color.FromArgb(197, 0, 219);
      colourTable[0] = Color.FromArgb(198, 0, 219);
      #endregion

      //for (int i = 0; i < colourTable.Length; i++) {
      //  double hue = ((double)i) / ((double)(colourTable.Length - 1));
      //  hue = 0.15 + hue * 0.88;
      //  if (hue > 1.0)
      //    hue -= 1.0;
      //  colourTable[i] = HSL2RGB(1.0 - hue, 1, .5, true);
      //}
    }

    public void MinMax(double Minimum, double Maximum) {
      min = Minimum;
      max = Maximum;
      range = max - min;
    }

    public Color Colour(double Value) {
      if (Value <= min)
        return Color.Black;
      if (Value >= max)
        return colourTable[colourTable.Length - 1];
      Value -= min;
      double hue = Math.Min(1.0, Value / range);  // 0 to 1
      //hue = 0.15 + hue * 0.93;
      //if (hue > 1.0)
      //  hue -= 1.0;
      // c = HSL2RGB(1.0 - hue, 1, .5, useAlt);
      return colourTable[(int)(hue * (colourTable.Length - 1))];
    }

    public void Render(PictureBox pic) {
      Graphics g = Graphics.FromImage(pic.Image);
      if (pic.Width > pic.Height) {
        double w = ((double)pic.ClientRectangle.Width) / ((double)colourTable.Length);
        for (int i = 0; i < colourTable.Length; i++)
          g.FillRectangle(new SolidBrush(colourTable[i]), (int)(((double)i) * w), 0,
            (int)w + 1, pic.ClientRectangle.Height);
      } else {
        double h = ((double)pic.ClientRectangle.Height) / ((double)colourTable.Length);
        for (int i = 0; i < colourTable.Length; i++)
          g.FillRectangle(new SolidBrush(colourTable[i]), 0, (int)(((double)i) * h),
            pic.ClientRectangle.Height, (int)h + 1);
      }
      pic.Refresh();
    }

    public struct ColorRGB {
      public byte R;
      public byte G;
      public byte B;

      public ColorRGB(Color value) {
        this.R = value.R;
        this.G = value.G;
        this.B = value.B;
      }

      public static implicit operator Color(ColorRGB rgb) {
        Color c = Color.FromArgb(rgb.R, rgb.G, rgb.B);
        return c;
      }

      public static explicit operator ColorRGB(Color c) {
        return new ColorRGB(c);
      }
    }

    // Given H,S,L in range of 0-1
    // Returns a Color (RGB struct) in range of 0-255
    public static ColorRGB HSL2RGB(double h, double sl, double l, bool useAlt) {
      double v;
      double r, g, b;
      r = l;   // default to gray
      g = l;
      b = l;
      v = (l <= 0.5) ? (l * (1.0 + sl)) : (l + sl - l * sl);
      if (v > 0) {
        double m;
        double sv;
        int sextant;
        double fract, vsf, mid1, mid2;
        m = l + l - v;
        sv = (v - m) / v;
        h *= 6.0;
        sextant = (int)h;
        fract = h - sextant;
        vsf = v * sv * fract;
        mid1 = m + vsf;
        mid2 = v - vsf;
        switch (sextant) {
          case 0:
            r = v;
            g = useAlt ? Math.Acos(1 - mid1) / 1.5708 : mid1;
            b = m;
            break;
          case 1:
            r = useAlt ? Math.Acos(1 - mid2) / 1.5708 : mid2;
            g = v;
            b = m;
            break;
          case 2:
            r = m;
            g = v;
            b = useAlt ? Math.Acos(1 - mid1) / 1.5708 : mid1;
            break;
          case 3:
            r = m;
            g = useAlt ? Math.Acos(1 - mid2) / 1.5708 : mid2;
            b = v;
            break;
          case 4:
            r = mid1;
            g = m;
            b = v;
            break;
          case 5:
            r = v;
            g = m;
            b = mid2;
            break;
        }
      }
      ColorRGB rgb;
      rgb.R = Convert.ToByte(r * 255.0f);
      rgb.G = Convert.ToByte(g * 255.0f);
      rgb.B = Convert.ToByte(b * 255.0f);
      return rgb;
    }

  }

  public class Damper {
    public enum OrientationType {MaxValue, MinValue, AverageValue};
    public OrientationType Orientation;
    private int[] history;
    private int writeNextHere = 0;
    private int num = 0;

    public Damper (int Size, OrientationType DamperType) {
      history = new int[Size];
      this.Orientation = DamperType;
    }

    public void Add(int NewValue) {
      history[writeNextHere] = NewValue;
      if (++writeNextHere >= history.Length)
        writeNextHere = 0;
      if (num < history.Length)
        num++;
    }

    public double Value() {
      if (num == 0)
        return 0;
        //throw new Exception("Undefined -- no values provided yet");
      else {
        int i = writeNextHere - num;
        if (i < 0)
          i += history.Length;
        int rc = history[i];
        for (int n = 1; n < num; n++) {
          if (++i >= history.Length)
            i = 0;
          switch (Orientation) {
            case OrientationType.MinValue: rc = Math.Min(rc, history[i]); break;
            case OrientationType.MaxValue: rc = Math.Max(rc, history[i]); break;
            case OrientationType.AverageValue: rc += history[i]; break;
          }
        }
        if (Orientation == OrientationType.AverageValue)
          return ((double)rc) / ((double)num);
        return rc;
      }
    }

    public void Reset() {
      num = 0;
    }

  }
}