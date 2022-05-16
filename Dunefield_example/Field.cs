using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace DunefieldModel {
  public partial class Field : UserControl {
    public int[,] Data;
    public Rectangle VisibleArea;  // Width is dunefieldLength
    public LegendDiscrete FieldLegend = new LegendDiscrete();
    public string CaptionText {
      set { label_Caption.Text = value; }
      get { return label_Caption.Text; }
    }
    public Image FieldImage { get { return pictureBox_Field.Image; } }

    public event MovementHandler SliderMove;
    public delegate void MovementHandler(object sender, int newPosition);
    public event MagnifierHandler Magnifier;
    public delegate void MagnifierHandler(object sender, MouseEventArgs e, Point ScreenLocation);

    public Field() {
      InitializeComponent();
    }

    private void Field_Load(object sender, EventArgs e) {
      VisibleArea = pictureBox_Field.ClientRectangle;
      pictureBox_Legend.Image =
          new Bitmap(pictureBox_Legend.ClientRectangle.Width, pictureBox_Legend.ClientRectangle.Height);
      FieldLegend.MinMax(0, 5);
      paintLegend();
    }

    public void Init(Rectangle VisibleArea) {
      this.VisibleArea = VisibleArea;
      this.Width = VisibleArea.Width + (this.Width - pictureBox_Field.Width);
      this.Height = VisibleArea.Height + (this.Height - pictureBox_Field.Height);
      pictureBox_Field.Image = new Bitmap(VisibleArea.Width, VisibleArea.Height);
      slider1.Init();
    }

    public Rectangle GetVisibleArea() {
      return VisibleArea;
    }

    public void Repaint(int Min, int Max) {
      if ((Min != FieldLegend.Minimum) || (Max != FieldLegend.Maximum)) {
        FieldLegend.MinMax(Min, Max);
        // paintLegend();
      }
      paintField();
    }

    public void Repaint(Range NewRange) {
      Repaint(NewRange.Min, NewRange.Max);
    }

    private void paintLegend() {
      FieldLegend.Render(pictureBox_Legend);
    }

    private void paintField() {
      Bitmap bm = (Bitmap)pictureBox_Field.Image;
      //Console.WriteLine("Field Paint" + ((bm == null) ? ", image is null" : "") + "legendScale [" +
      //    FieldLegend.Minimum.ToString() + "," + FieldLegend.Maximum.ToString() + "]");
      if ((bm != null) && (Data != null)) {
        int xEnd = Math.Min(VisibleArea.X + VisibleArea.Width, Data.GetLength(1));
        int wEnd = Math.Min(VisibleArea.Y + VisibleArea.Height, Data.GetLength(0));
        for (int x = VisibleArea.X; x < xEnd; x++)
          for (int w = VisibleArea.Y; w < wEnd; w++)
            bm.SetPixel(x, pictureBox_Field.ClientSize.Height - w - 1, FieldLegend.Colour(Data[w, x]));
      }
      pictureBox_Field.Refresh();
    }

    private void magnifier(MouseEventArgs e) {
      if (Magnifier != null)
        Magnifier(this, e, pictureBox_Field.PointToScreen(e.Location));
    }

    private void pictureBox_Field_MouseDown(object sender, MouseEventArgs e) {
      magnifier(e);
    }

    private void pictureBox_Field_MouseMove(object sender, MouseEventArgs e) {
      magnifier(e);
    }

    private void pictureBox_Field_MouseUp(object sender, MouseEventArgs e) {
      magnifier(e);
    }

    private void slider1_SliderMove(object sender, int newPosition) {
      if (SliderMove != null)
        SliderMove(this, newPosition);
    }

  }
}
