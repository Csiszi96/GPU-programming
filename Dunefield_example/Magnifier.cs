using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace DunefieldModel {
  public partial class Magnifier : Form {
    private const int displayOffset = 20;

    public Magnifier() {
      InitializeComponent();
    }

    public Magnifier(int Width, int Height) {
      InitializeComponent();
      this.Width = Width + (this.Width - pictureBox1.Width);
      this.Height = Height + (this.Height - pictureBox1.Height);
    }

    public void UpdateImage(Point ScreenLocation, Point FieldLocation,
        int ValueAtLocation, Bitmap NewImage) {
      this.Show();
      this.Text = "Magnifier [" + FieldLocation.Y.ToString() + "," +
          FieldLocation.X.ToString() + "] " + ValueAtLocation.ToString();
      pictureBox1.Image = NewImage;
      Rectangle scr = Screen.GetWorkingArea(ScreenLocation);
      if ((ScreenLocation.X + this.Width + displayOffset) > (scr.Left + scr.Width))
        ScreenLocation.X -= this.Width + displayOffset * 2;
      if ((ScreenLocation.Y + this.Height + displayOffset) > (scr.Top + scr.Height))
        ScreenLocation.Y -= this.Height + displayOffset * 2;
      ScreenLocation.X += displayOffset;
      ScreenLocation.Y += displayOffset;
      // this.Location = ScreenLocation;
    }

    private void Magnifier_FormClosing(object sender, FormClosingEventArgs e) {
      if (e.CloseReason == CloseReason.UserClosing) {
        e.Cancel = true;
        this.Hide();
      }
    }
  }
}
