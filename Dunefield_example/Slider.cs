using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace DunefieldModel {
  public partial class Slider : UserControl {
    private int originalSliderY;
    private int originalCursorY;

    public event MovementHandler SliderMove;
    public delegate void MovementHandler(object sender, int newPosition);

    public Slider() {
      InitializeComponent();
    }

    private void Slider_Load(object sender, EventArgs e) {
      int w = pictureBox1.Width;
      pictureBox1.Height = pictureBox1.Width;
      Bitmap bm = new Bitmap(w, w);
      Graphics g = Graphics.FromImage(bm);
      Point[] pts = new Point[3];
      pts[0] = new Point(0, 0);
      pts[1] = new Point(15, 7);
      pts[2] = new Point(0, 15);
      g.FillPolygon(new SolidBrush(Color.Gray), pts);
      pictureBox1.Image = bm;
      pictureBox1.Location = new Point(0, 0);
    }

    public void Init() {
      pictureBox1.Top = 0;
    }

    private void pictureBox1_MouseDown(object sender, MouseEventArgs e) {
      originalSliderY = pictureBox1.Top;
      originalCursorY = Cursor.Position.Y;
    }

    private void pictureBox1_MouseMove(object sender, MouseEventArgs e) {
      if (e.Button == MouseButtons.Left) {
        int middleOfSlider = pictureBox1.Height / 2;
        int y = Math.Max(0, originalSliderY + (Cursor.Position.Y - originalCursorY));
        y = Math.Min(y, this.Height - pictureBox1.Width);
        pictureBox1.Top = y;
        if (SliderMove != null)
          SliderMove(this, this.Height - pictureBox1.Width - y);
      }
    }

  }
}
