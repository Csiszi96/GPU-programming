using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace DunefieldModel {
  public partial class Chart : UserControl {
    public int ChartWidth;
    public int ChartHeight;
    public struct Dataset {
      public DataSeries Data;
      public ChartAxis Axis;
      public Color LineColour;
      public Dataset(DataSeries Data, ChartAxis Axis, Color LineColour) {
        this.Data = Data;
        this.Axis = Axis;
        this.LineColour = LineColour;
      }
    }
    public List<Dataset> Datasets = new List<Dataset>();

    public Chart() {
      InitializeComponent();
    }

    private void Chart_Load(object sender, EventArgs e) {
    }

    public void Init(int Width, int Height) {
      ChartWidth = (Width < 0) ? pictureBox1.ClientRectangle.Width : Width;
      ChartHeight = (Height < 0) ? pictureBox1.ClientRectangle.Height : Height;
      this.Width = this.Width - pictureBox1.ClientRectangle.Width + ChartWidth;
      this.Height = this.Height - pictureBox1.ClientRectangle.Height + ChartHeight;
      Reset();
    }

    public void Reset() {
      pictureBox1.Image = new Bitmap(Width, Height);
      Datasets.Clear();
      Datasets.Add(new Dataset(null, null, Color.Black));  // reserve item zero for background
    }

    public void AddBackgroundSeries(DataSeries Data) {
      Datasets.RemoveAt(0);
      Datasets.Insert(0, new Dataset(Data, null, Color.Black));
    }

    public int AddSeries(DataSeries Data, ChartAxis Axis, Color LineColor) {
      Datasets.Add(new Dataset(Data, Axis, LineColor));
      return Datasets.Count - 1;
    }

    private void drawSeries(Graphics g, DataSeries ds, ChartAxis ca, Pen pen) {
      int min = ca.ScaleMin;
      float scale = ((float)(ChartHeight - 1)) / ((float)(ca.ScaleMax - min));
      if (float.IsInfinity(scale))
        scale = 1;
      float maxY = (float)(ChartHeight - 1);
      //// this draws steps between points
      //PointF[] pts = new PointF[ds.Data.Length * 2];
      //for (int i = 0; i < ds.Data.Length; i++) {
      //  int j = i * 2;
      //  pts[j].X = i;
      //  pts[j].Y = maxY - ((float)(ds.Data[i] - min)) * scale;
      //  pts[j + 1].X = i;
      //  pts[j + 1].Y = pts[j].Y;
      //}
      // this draws diagonals between points
      PointF[] pts = new PointF[ds.Data.Length];
      for (int i = 0; i < ds.Data.Length; i++) {
        pts[i].X = i;
        pts[i].Y = maxY - ((float)(ds.Data[i] - min)) * scale;
      }
      g.DrawLines(pen, pts);
    }

    public void Render() {
      Graphics g = Graphics.FromImage(pictureBox1.Image);
      g.Clear(Color.FromKnownColor(KnownColor.Control));
      DataSeries dsBg = Datasets[0].Data;
      if (dsBg != null) {
        SolidBrush dark = new SolidBrush(Color.Silver);
        int startDark = -1;
        for (int x = 0; x <= dsBg.Data.Length; x++) {
          if ((x < dsBg.Data.Length) && (dsBg.Data[x] > 0)) {
            if (startDark < 0)
              startDark = x;
          } else if (startDark >= 0) {
            g.FillRectangle(dark, startDark, 0, x - startDark, pictureBox1.Height);
            startDark = -1;
          }
        }
      }
      for (int i = 1; i < Datasets.Count; i++) {
        Dataset ds = Datasets[i];
        drawSeries(g, ds.Data, ds.Axis, new Pen(ds.LineColour));
      }
      pictureBox1.Refresh();
    }

  }
}
