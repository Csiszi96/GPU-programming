using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace DunefieldModel {
  public partial class ChartAxis : UserControl {
    public int ScaleMax;
    public int ScaleMin;
    public string Title;
    public Color LineColour;
    public Range ScaleRange {
      get { return new Range(ScaleMin, ScaleMax); }
      set {
        ScaleMin = value.Min;
        ScaleMax = value.Max;
        textBox_Min.Text = ScaleMin.ToString();
        textBox_Max.Text = ScaleMax.ToString();
      }
    }
    public Range ActualRange {
      get {
        int min = 0;
        int max = 1;
        int.TryParse(label_Min.Text, out min);
        int.TryParse(label_Max.Text, out max);
        return new Range(min, max);
      }
      set {
        label_Min.Text = value.Min.ToString();
        label_Max.Text = value.Max.ToString();
        if (checkBox_Auto.Checked && ((ScaleMin != value.Min) || (ScaleMax != value.Max))) {
          ScaleRange = value;
          if (RenderChart != null)
            RenderChart(this);
        }
      }
    }

    public event RenderingHandler RenderChart;
    public delegate void RenderingHandler(object sender);

    public ChartAxis() {
      InitializeComponent();
      Reset();
    }

    public void Reset() {
      label_Min.Text = "";
      label_Max.Text = "";
      textBox_Min.Text = "";
      textBox_Max.Text = "";
      checkBox_Auto.Checked = true;
    }

    public void Bind(string Title, Color LineColour) {
      this.Title = Title;
      this.LineColour = LineColour;
      label_Min.ForeColor = LineColour;
      label_Max.ForeColor = LineColour;
      Reset();
      this.Refresh();
    }

    private void ChartAxis_Paint(object sender, PaintEventArgs e) {
      Graphics g = this.CreateGraphics();
      StringFormat sf = new StringFormat();
      sf.FormatFlags = StringFormatFlags.DirectionVertical; // | StringFormatFlags.DirectionRightToLeft;
      g.DrawString(Title, new Font("Arial", 9), new SolidBrush(LineColour), 0, 0, sf);
    }

    private void label_Max_Click(object sender, EventArgs e) {
      ScaleMax = int.Parse(label_Max.Text);
      textBox_Max.Text = ScaleMax.ToString();
      if (RenderChart != null)
        RenderChart(this);
    }

    private void label_Min_Click(object sender, EventArgs e) {
      ScaleMin = int.Parse(label_Min.Text);
      textBox_Min.Text = ScaleMin.ToString();
      if (RenderChart != null)
        RenderChart(this);
    }

    private void textBox_Max_Validated(object sender, EventArgs e) {
      ScaleMax = int.Parse(textBox_Max.Text);
      if (RenderChart != null)
        RenderChart(this);
    }

    private void textBox_Min_Validated(object sender, EventArgs e) {
      ScaleMin = int.Parse(textBox_Min.Text);
      if (RenderChart != null)
        RenderChart(this);
    }

  }
}
