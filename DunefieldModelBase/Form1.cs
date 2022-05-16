using System;
using System.IO;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Forms;
using System.Diagnostics;

namespace DunefieldModel {
  public partial class Form1 : Form {
    public Chart refChart1;
    public ChartAxis refChartAxis1;
    public ChartAxis refChartAxis2;
    public ChartAxis ChartAxis_Field;  // chartAxis controlling field display
    public ChartAxis ChartAxis_Height; // chartAxis controlling height line on chart
    public Model df;
    private AppState appState = new AppState();
    private bool runUntilStopped;
    private bool stop = false;
    private int recordingIndex = 0;
    private Stopwatch et = new Stopwatch();
    private Damper dfMin = new Damper(5, Damper.OrientationType.MinValue);
    private Damper dfMax = new Damper(5, Damper.OrientationType.MaxValue);
    private Damper dfCount;
    private double initialGrainCount;
    private int[] chartBgnd;
    private int[] chartLine1;
    private int crossSectionW = 0;
    private Magnifier magnifierForm;
    private int ticksPerRefresh;
    private int tickStop;
    private int[,] customData;

    public class AppState {
      public int Version = 1;
      public int Ticks = 0;
      public int ModelAcrossWind = 256;
      public int ModelLengthDownwind = 256;
      public bool Recycle = true;
      public int SandDepth = 3;
      public float pSand = 0.6f;
      public float pNoSand = 0.4f;
      public int hopLength = 5;
      public int legendMin = 0;
      public int legendMax = 10;
      public string FolderPath = "";  // @"C:\Users\Jim\Documents\Jim\sand\model";
      public string Filename = "model.dune";
      public string InitializeType = "Random";
      public string RecordingPath = "";  // @"C:\Users\Jim\Documents\Jim\sand\model";
      public int SelectedModelIndex = 1;
      public int SelectedNeighbourhoodIndex = 0;
    }

    private Color[] legend = {
      Color.Yellow, Color.Green, Color.Cyan, Color.Red};

    public Form1() {
      InitializeComponent();
    }

    private void run(int cycles) {
      et.Start();
      appState.Ticks += cycles;
      for (int i = 0; i < cycles; i++) {
        df.Tick();
        Application.DoEvents();
      }
      dfCount.Add(df.Count());
      float c = (float)dfCount.Value();
      label_GrainCount.Text = "Grains: " + c.ToString("#") + " (" +
          (c * 100 / initialGrainCount).ToString("#") + "%)";
      Range r = df.MinMaxHeight();
      dfMin.Add(r.Min);
      dfMax.Add(r.Max);
      chartAxis_Dunefield.ActualRange = r;
      chartAxis1.ActualRange = r;
      displayStatus(r);
      updateDunefieldDisplay();
      updateChart();
      et.Stop();
      if (checkBox_MakeRecording.Checked)
        recordingDoIt();
      label_ElapsedTime.Text = appState.Ticks + " ticks in " +
          (et.ElapsedMilliseconds / 3600000 % 24).ToString("0#") + ":" +
          (et.ElapsedMilliseconds / 60000 % 60).ToString("0#") + ":" +
          (et.ElapsedMilliseconds / 1000 % 60).ToString("0#");
    }

    private void resize() {
      int gap = groupBox1.Left;
      this.Width = field1.Left + field1.Width + gap + 8;
      this.Height = this.Height - this.ClientRectangle.Height +
          groupBox2.Top + groupBox2.Height + gap;
      crossSectionW = 0;
    }

    private void init() {
      if (radioButton_Uniform.Checked)
        appState.InitializeType = "Uniform";
      else if (radioButton_Random.Checked)
        appState.InitializeType = "Random";
      else if (radioButton_Square.Checked)
        appState.InitializeType = "Square";
      else if (radioButton_Dune.Checked)
        appState.InitializeType = "Transverse";
      else if (radioButton_TruncTrans.Checked)
        appState.InitializeType = "TruncTrans";
      else if (radioButton_File.Checked)
        appState.InitializeType = "File";
      appState.ModelLengthDownwind = int.Parse(textBox_Width.Text);
      appState.ModelAcrossWind = int.Parse(textBox_Height.Text);
      field1.Init(new Rectangle(0, 0, appState.ModelLengthDownwind, appState.ModelAcrossWind));
      resize();
      chart1.Init(appState.ModelLengthDownwind, -1);
      chartBgnd = new int[appState.ModelLengthDownwind];
      chart1.Reset();
      chart1.AddBackgroundSeries(new DataSeries(ref chartBgnd));
      chartLine1 = new int[appState.ModelLengthDownwind];
      DataSeries ds = new DataSeries(ref chartLine1);
      int dsNum = chart1.AddSeries(ds, chartAxis1, Color.Red);
      chartAxis1.Bind("Height", chart1.Datasets[dsNum].LineColour);
      appState.Recycle = checkBox_Recycle.Checked;
      appState.SandDepth = int.Parse(textBox_SandDepth.Text);
      appState.pSand = float.Parse(textBox_pSand.Text);
      appState.pNoSand = float.Parse(textBox_pNoSand.Text);
      appState.hopLength = int.Parse(textBox_hopLength.Text);
      IFindSlope fs = null;
      switch (appState.SelectedNeighbourhoodIndex) {
        case 0: fs = new FindSlopeVonNeumannDeterministic(); break;
        case 1: fs = new FindSlopeVonNeumannStochastic(); break;
        case 2: fs = new FindSlopeMooreDeterministic(); break;
        case 3: fs = new FindSlopeMooreStochastic(); break;
        case 4: fs = new FindSlopeMooreDeterministicDownwind(); break;
        case 5: fs = new FindSlopeMooreStochasticDownwind(); break;
      }
      switch (appState.SelectedModelIndex) {  // set in Load, below
        case 0: df = new SandPile(this, fs, appState.ModelAcrossWind, appState.ModelLengthDownwind); break;
        case 1: df = new Werner1995(this, fs, appState.ModelAcrossWind, appState.ModelLengthDownwind); break;
        case 2: df = new WernerModified(this, fs, appState.ModelAcrossWind, appState.ModelLengthDownwind); break;
        case 3: df = new Momiji2000(this, fs, appState.ModelAcrossWind, appState.ModelLengthDownwind); break;
        case 4: df = new Baas2002(this, fs, appState.ModelAcrossWind, appState.ModelLengthDownwind); break;
        case 5: df = new VariationOne(this, fs, appState.ModelAcrossWind, appState.ModelLengthDownwind); break;
        case 6: df = new VariationTwo(this, fs, appState.ModelAcrossWind, appState.ModelLengthDownwind); break;
        case 7: df = new OpenBalancedFlux(this, fs, appState.ModelAcrossWind, appState.ModelLengthDownwind); break;
      }
      df.HopLength = appState.hopLength;
      textBox_hopLength.Enabled = df.UsesHopLength();
      textBox_pNoSand.Enabled = df.UsesSandProbabilities();
      textBox_pSand.Enabled = df.UsesSandProbabilities();
      field1.Data = df.Elev;
      df.SetOpenEnded(!appState.Recycle);
      df.pSand = appState.pSand;
      df.pNoSand = appState.pNoSand;
      switch (appState.InitializeType) {
        case "Random": df.InitRandom(appState.SandDepth); break;
        case "Uniform": df.InitUniform(appState.SandDepth); break;
        case "Square": df.InitSquare(appState.SandDepth); break;
        case "Transverse": df.InitDune(appState.SandDepth, df.WidthAcross); break;
        case "TruncTrans": df.InitDune(appState.SandDepth, Math.Min(100, df.WidthAcross / 2)); break;
        case "File":
          if (InitFromFile(df, openFileDialog1.FileName)) {
            appState.FolderPath = Path.GetDirectoryName(openFileDialog1.FileName);
            appState.Filename = Path.GetFileName(openFileDialog1.FileName);
          }
          break;
      }
      // Above, Model initializers set chartAxis1 scaleRange to reasonable values for desired init
      chartAxis_Dunefield.ScaleRange = chartAxis1.ScaleRange;
      initialGrainCount = df.Count();
      label_InitStatus.Text = "Ready.  " + df.LengthDownwind + " x " + df.WidthAcross +
          " with " + initialGrainCount + " grains";
      Range dfRange = df.MinMaxHeight();
      displayStatus(dfRange);
      dfMin.Reset();
      dfMax.Reset();
      chartAxis_Dunefield.ActualRange = dfRange;
      chartAxis1.ActualRange = dfRange;
      field1.Repaint(chartAxis_Dunefield.ScaleMin, chartAxis_Dunefield.ScaleMax);
      updateChart();
      recordingInit();
      appState.Ticks = 0;
      et.Reset();
      label_ElapsedTime.Text = "";
      label_DateTime.Text = "           " + DateTime.Now.ToString("yyyy-MMM-dd HH:mm");
      dfCount = new Damper(df.LengthDownwind / 2, Damper.OrientationType.AverageValue);
    }

    private void button_Initialize_Click(object sender, EventArgs e) {
      init();
    }

    private void button_Tick_Click(object sender, EventArgs e) {
      runUntilStopped = false;
      timer1.Enabled = true;
    }

    private void button_Run_Click(object sender, EventArgs e) {
      if (button_Run.Text == "Run") {
        ticksPerRefresh = int.Parse(textBox_ticksPerRefresh.Text);
        tickStop = -1;
        if (textBox_tStop.Text.Length > 0)
          tickStop = int.Parse(textBox_tStop.Text);
        runUntilStopped = true;
        timer1.Enabled = true;
        button_Run.Text = "Stop";
      } else {
        stop = true;
        button_Run.Text = "Run";
      }
    }

    private void button_Stop_Click(object sender, EventArgs e) {
      stop = true;
    }

    private void displayStatus(Range r) {
      field1.CaptionText = "Scale [" + chartAxis_Dunefield.ScaleMin + ", " + chartAxis_Dunefield.ScaleMax +
          "]  T=" + appState.Ticks + "  Height range [" +
          r.Min + " " + r.Max + "], relief " + (r.Max - r.Min + 1);
    }

    private void button_Browse_Click(object sender, EventArgs e) {
      if (openFileDialog1.ShowDialog() == DialogResult.OK) {
        label_Filename.Text = Path.GetFileName(openFileDialog1.FileName);
        radioButton_File.Checked = true;
      }
    }

    private void button_Save_Click(object sender, EventArgs e) {
      saveFileDialog1.InitialDirectory = appState.FolderPath;
      saveFileDialog1.FileName = appState.Filename;
      if (saveFileDialog1.ShowDialog() == DialogResult.OK) {
        if (SaveToFile(df, saveFileDialog1.FileName)) {
          appState.FolderPath = Path.GetDirectoryName(saveFileDialog1.FileName);
          appState.Filename = Path.GetFileName(saveFileDialog1.FileName);
        }
      }
    }

    private void updateDunefieldDisplay() {
      if (listBox_Show.SelectedItem.Equals("Saltation")) {
        customData = new int[df.WidthAcross, df.LengthDownwind];
        for (int w = 0; w < df.WidthAcross; w++)
          for (int x = 0; x < df.LengthDownwind; x++)
            df.SaltationLength(w, x);
        field1.Data = customData;
        chartAxis_Dunefield.ScaleRange = new Range(0, 15);
      } else if (listBox_Show.SelectedItem.Equals("Shadow")) {
        customData = new int[df.WidthAcross, df.LengthDownwind];
        for (int w = 0; w < df.WidthAcross; w++)
          for (int x = 0; x < df.LengthDownwind; x++)
            customData[w, x] = (df.Shadow[w, x] > 0) ? 1 : 0;
        field1.Data = customData;
        chartAxis_Dunefield.ScaleRange = new Range(0, 1);
      } else if (listBox_Show.SelectedItem.Equals("Custom A")) {
        customData = new int[df.WidthAcross, df.LengthDownwind];
        for (int w = 0; w < df.WidthAcross; w++)
          for (int x = 0; x < df.LengthDownwind; x++)
            customData[w, x] = 0;
        field1.Data = customData;
        chartAxis_Dunefield.ScaleRange = new Range(0, 1);
      } else {
        field1.Data = df.Elev;
        chartAxis_Dunefield.ActualRange = df.MinMaxHeight();
      }
      field1.Repaint(chartAxis_Dunefield.ScaleMin, chartAxis_Dunefield.ScaleMax);
      updateMagnifier();
    }

    private void listBox_Show_SelectedIndexChanged(object sender, EventArgs e) {
      if (df != null)
        updateDunefieldDisplay();
    }

    private void chartAxis_Dunefield_RenderChart(object sender) {
      field1.Repaint(chartAxis_Dunefield.ScaleMin, chartAxis_Dunefield.ScaleMax);
    }

    private void chartAxis_RenderChart(object sender) {
      chart1.Render();
    }

    private void updateChart() {
      for (int x = 0; x < df.LengthDownwind; x++) {
        chartBgnd[x] = (df.Shadow[crossSectionW, x] > 0) ? 1 : 0;
        chartLine1[x] = df.Elev[crossSectionW, x];
      }
      chart1.Render();
    }

    private void field1_SliderMove(object sender, int newPosition) {
      if (crossSectionW != newPosition) {
        crossSectionW = newPosition;
        updateChart();
      }
    }

    private void comboBox1_SelectedIndexChanged(object sender, EventArgs e) {
      appState.SelectedModelIndex = comboBox_Model.SelectedIndex;
    }

    private void comboBox_Neighbourhood_SelectedIndexChanged(object sender, EventArgs e) {
      appState.SelectedNeighbourhoodIndex = comboBox_Neighbourhood.SelectedIndex;
    }

    private bool InitFromFile(Model df, string PathFilename) {
      bool rc = true;
      try {
        using (StreamReader file = new StreamReader(PathFilename)) {
          string[] buf = file.ReadLine().Split(' ');
          if ((int.Parse(buf[0]) != df.LengthDownwind) || (int.Parse(buf[1]) != df.WidthAcross))
            throw new ArgumentException("Dunefield size in file (" + buf[0] + "," + buf[1] +
                ") does not match current dimensions");
          file.ReadLine();  // skip config comment line
          for (int x = 0; x < df.LengthDownwind; x++) {
            buf = file.ReadLine().Split(' ');
            for (int w = 0; w < df.WidthAcross; w++)
              df.Elev[w, x] = int.Parse(buf[w]);
          }
        }
        df.InitCompletion();
      } catch (Exception ex) {
        MessageBox.Show("Init from file failed: " + ex.Message);
        rc = false;
      }
      return rc;
    }

    private bool SaveToFile(Model df, string PathFilename) {
      bool rc = true;
      try {
        using (StreamWriter file = new StreamWriter(PathFilename)) {
          file.WriteLine(df.LengthDownwind.ToString() + ' ' + df.WidthAcross.ToString());
          file.WriteLine("# " + Universal.ObjToXml(appState));
          for (int x = 0; x < df.LengthDownwind; x++) {
            for (int w = 0; w < df.WidthAcross; w++)
              file.Write(df.Elev[w, x].ToString() + " ");
            file.WriteLine();
          }
        }
      } catch (Exception ex) {
        MessageBox.Show("File could not be written: " + ex.Message);
        rc = false;
      }
      return rc;
    }

    private void recordingInit() {
      recordingIndex = 0;
    }

    private string recordingFilename() {
      return textBox_RecordingFile.Text + "_" + recordingIndex.ToString("00#") + ".png";
    }

    private void recordingSetDir(string newPath) {
      folderBrowserDialog1.SelectedPath = newPath;
      textBox_RecordingDir.Text = newPath;
      appState.RecordingPath = newPath;
      recordingInit();
      label_RecordingStatus.Text = "Next file: " + recordingFilename();
    }

    private void button_RecordingDir_Click(object sender, EventArgs e) {
      if (folderBrowserDialog1.ShowDialog() == DialogResult.OK)
        recordingSetDir(folderBrowserDialog1.SelectedPath);
    }

    private void checkBox_MakeRecording_CheckedChanged(object sender, EventArgs e) {
      if (textBox_RecordingFile.Text.Length == 0)
        MessageBox.Show("Notice: Recording filename is blank");
    }

    private ImageCodecInfo GetImageEncoder(string imageType) {
      imageType = imageType.ToUpperInvariant();
      foreach (ImageCodecInfo info in ImageCodecInfo.GetImageEncoders())
        if (info.FormatDescription == imageType)
          return info;
      return null;
    }

    private void recordingDoIt() {
      label_RecordingStatus.Text = "This file: " + recordingFilename();
      Application.DoEvents();
      string destPathfile = textBox_RecordingDir.Text + Path.DirectorySeparatorChar + recordingFilename();
      Bitmap bm;
      if (radioButton_RecordWindow.Checked) {
        bm = new Bitmap(Width, Height);
        Graphics g = Graphics.FromImage(bm);
        g.CopyFromScreen(Left, Top, 0, 0, new Size(Width, Height));
      } else
        bm = new Bitmap(field1.FieldImage);
      //ImageCodecInfo jpegEncoder = GetImageEncoder("JPEG");
      //EncoderParameters parms = new EncoderParameters(1);
      //parms.Param[0] = new EncoderParameter(Encoder.Compression, 40);
      //bm.Save(destPathfile, jpegEncoder, parms);
      bm.Save(destPathfile, ImageFormat.Png);
      recordingIndex++;
      label_RecordingStatus.Text = "Next file: " + recordingFilename();
    }

    private void button_MakeRecording_Click(object sender, EventArgs e) {
      if (textBox_RecordingFile.Text.Length == 0)
        MessageBox.Show("No recording made", "Recording filename is blank");
      else
        recordingDoIt();
    }

    private void loadFromAppState(string appStateString) {
      if (appStateString.Length > 0) {
        appState = (AppState)Universal.XmlToObj(typeof(AppState), appStateString);
        if (appState == null) {
          Console.WriteLine("AppState deserializer failed");
          appState = new AppState();
        }
      }
      textBox_Width.Text = appState.ModelLengthDownwind.ToString();
      textBox_Height.Text = appState.ModelAcrossWind.ToString();
      checkBox_Recycle.Checked = appState.Recycle;
      textBox_SandDepth.Text = appState.SandDepth.ToString();
      textBox_pSand.Text = appState.pSand.ToString();
      textBox_pNoSand.Text = appState.pNoSand.ToString();
      textBox_hopLength.Text = appState.hopLength.ToString();
      switch (appState.InitializeType) {
        case "Uniform": radioButton_Uniform.Checked = true; break;
        case "Square": radioButton_Square.Checked = true; break;
        case "Dune": radioButton_Dune.Checked = true; break;
        case "TruncTrans": radioButton_TruncTrans.Checked = true; break;
        case "File": radioButton_File.Checked = true; break;
        default: radioButton_Random.Checked = true; break;
      }
      label_Filename.Text = appState.Filename;
      label_InitStatus.Text = "Press 'Initialize'";
      field1.CaptionText = "";
      label_GrainCount.Text = "";
      chartAxis_Dunefield.ScaleRange = new Range(appState.legendMin, appState.legendMax);
      openFileDialog1.InitialDirectory = appState.FolderPath;
      openFileDialog1.FileName = appState.FolderPath + Path.DirectorySeparatorChar + appState.Filename;
      recordingSetDir(appState.RecordingPath);
      label_ElapsedTime.Text = "";
      listBox_Show.SelectedIndex = 0;
      comboBox_Model.SelectedIndex = appState.SelectedModelIndex;
      comboBox_Neighbourhood.SelectedIndex = appState.SelectedNeighbourhoodIndex;
    }

    private void Form1_Load(object sender, EventArgs e) {
      radioButton_RecordWindow.Checked = true;
                                                  // these numbers are used in the case statement above, in 'init'
      comboBox_Model.Items.Add("Sand pile/pit");  // 0
      comboBox_Model.Items.Add("A: Werner (1995)");  // 1
      comboBox_Model.Items.Add("B: Werner (1995) + no erosion in shadow");  // 2
      comboBox_Model.Items.Add("Momiji (2000): B + stoss wind speed-up");  // 3
      comboBox_Model.Items.Add("Baas (2002): B + one event per cell, L=1");  // 4
      comboBox_Model.Items.Add("Variation 1: B + stop hop at first shadow");  // 5
      comboBox_Model.Items.Add("Variation 2: 1 + wind speed-up");  // 6
      comboBox_Model.Items.Add("Open with balanced net flux");  // 7
      comboBox_Neighbourhood.Items.Add("Von Neumann, deterministic");
      comboBox_Neighbourhood.Items.Add("Von Neumann, stochastic");
      comboBox_Neighbourhood.Items.Add("Moore, deterministic");
      comboBox_Neighbourhood.Items.Add("Moore, stochastic");
      comboBox_Neighbourhood.Items.Add("Moore, deterministic, no upwind");
      comboBox_Neighbourhood.Items.Add("Moore, stochastic, no upwind");
      textBox_ticksPerRefresh.Text = "5";
      textBox_tStop.Text = "";
      string buf = Properties.Settings.Default.AppState;
      refChart1 = chart1;
      refChartAxis1 = chartAxis1;
      refChartAxis2 = chartAxis2;
      chartAxis2.Visible = false;
      chartAxis_Dunefield.Bind("Dunefield", Color.Black);
      ChartAxis_Field = chartAxis_Dunefield;  // make chartAxes visible to models
      ChartAxis_Height = chartAxis1;
      loadFromAppState(buf);
      init();
    }

    private void Form1_FormClosing(object sender, FormClosingEventArgs e) {
      stop = true;
      Properties.Settings.Default.AppState = Universal.ObjToXml(appState);
      Properties.Settings.Default.Save();
    }

    private void timer1_Tick(object sender, EventArgs e) {
      timer1.Enabled = false;
      if (runUntilStopped) {
        stop = false;
        while (!stop) {
          run(ticksPerRefresh);
          if ((tickStop >= 0) && (appState.Ticks >= tickStop))
            button_Run_Click(null, null);
          Application.DoEvents();
        }
      } else
        run(1);
    }

    #region Magnifier
    private Point magnifierCenter;
    private const int magnifierPixelSize = 60;
    private const int magnifierPixelCount = 9;
    private const int magnifierGap = 2;

    private void updateMagnifier() {
      if ((magnifierForm != null) && magnifierForm.Visible) {
        Point tmp = magnifierCenter;
        magnifierCenter.X++;
        paintMagnifier(tmp, new Point(0, 0));
      }
    }

    private void paintMagnifier(Point Location, Point ScreenLocation) {
      Point center = Location;
      if (!center.Equals(magnifierCenter) &&
          (center.X >= 0) && (center.X < df.LengthDownwind) &&
          (center.Y >= 0) && (center.Y < df.WidthAcross)) {
        magnifierCenter = center;
        center.X = Math.Max(0, center.X - magnifierPixelCount / 2);
        center.X = Math.Min(field1.GetVisibleArea().Width - magnifierPixelCount, center.X);
        center.Y = df.WidthAcross - center.Y - 1;
        center.Y = Math.Max(0, center.Y - magnifierPixelCount / 2);
        center.Y = Math.Min(field1.GetVisibleArea().Height - magnifierPixelCount, center.Y);
        Bitmap bm = new Bitmap(magnifierPixelCount * magnifierPixelSize + (magnifierPixelCount - 1) * magnifierGap,
            magnifierPixelCount * magnifierPixelSize + (magnifierPixelCount - 1) * magnifierGap);
        Graphics g = Graphics.FromImage(bm);
        //Console.WriteLine(center.Y);
        Font bf = new Font("Arial", 9, FontStyle.Bold);
        for (int x = 0; x < magnifierPixelCount; x++)
          for (int w = 0; w < magnifierPixelCount; w++) {
            int cx = center.X + x;
            int cw = center.Y + w;
            if ((cx < 0) || (cw < 0)) continue;
            Color bc = field1.FieldLegend.Colour(field1.Data[cw, cx]);
            g.FillRectangle(new SolidBrush(bc),
                x * (magnifierPixelSize + magnifierGap),
                (magnifierPixelCount - w - 1) * (magnifierPixelSize + magnifierGap),
                magnifierPixelSize, magnifierPixelSize);
            SolidBrush tb = new SolidBrush((bc.Equals(Color.FromArgb(255, 255, 0)) ||
                bc.Equals(Color.FromArgb(180, 255, 0))) ? Color.Black : Color.White);
            g.DrawString("[" + (cw).ToString() + "," + (cx).ToString() + "]", bf,
                tb, x * (magnifierPixelSize + magnifierGap) + 1,
                (magnifierPixelCount - w - 1) * (magnifierPixelSize + magnifierGap) + 1);
            if (listBox_Show.SelectedItem.Equals("Shadow"))
              g.DrawString("Shd: " + df.Elev[cw, cx].ToString(), bf,
                  tb, x * (magnifierPixelSize + magnifierGap) + 1,
                  (magnifierPixelCount - w - 1) * (magnifierPixelSize + magnifierGap) + 15);
            else if (listBox_Show.SelectedItem.Equals("Custom A"))
              g.DrawString("cA: " + df.Elev[cw, cx].ToString(), bf,
                  tb, x * (magnifierPixelSize + magnifierGap) + 1,
                  (magnifierPixelCount - w - 1) * (magnifierPixelSize + magnifierGap) + 15);
            else
              g.DrawString(field1.Data[cw, cx].ToString(), bf,
                  tb, x * (magnifierPixelSize + magnifierGap) + 1,
                  (magnifierPixelCount - w - 1) * (magnifierPixelSize + magnifierGap) + 15);
            g.DrawString("Shd: " + df.Shadow[cw, cx].ToString("0.0"), bf,
                tb, x * (magnifierPixelSize + magnifierGap) + 1,
                (magnifierPixelCount - w - 1) * (magnifierPixelSize + magnifierGap) + 30);
            g.DrawString("Hop: " + df.SaltationLength(cw, cx).ToString(), bf,
                tb, x * (magnifierPixelSize + magnifierGap) + 1,
                (magnifierPixelCount - w - 1) * (magnifierPixelSize + magnifierGap) + magnifierPixelSize - 15);
            if (df.Shadow[cw, cx] > 0) {
              g.FillEllipse(new SolidBrush(Color.Gray), x * (magnifierPixelSize + magnifierGap) + magnifierPixelSize - 18,
                  (magnifierPixelCount - w - 1) * (magnifierPixelSize + magnifierGap) + magnifierPixelSize - 18,
                  16, 16);
              if (df.Shadow[cw, cx] > 0)
                g.DrawEllipse(new Pen(Color.White, 2), x * (magnifierPixelSize + magnifierGap) + magnifierPixelSize - 17,
                    (magnifierPixelCount - w - 1) * (magnifierPixelSize + magnifierGap) + magnifierPixelSize - 17,
                    15, 15);
            }
          }
        if (magnifierForm == null)
          magnifierForm = new Magnifier(bm.Width, bm.Height);
        magnifierForm.UpdateImage(ScreenLocation, new Point(field1.GetVisibleArea().Height - Location.Y - 1,
            Location.X), field1.Data[field1.GetVisibleArea().Height - Location.Y - 1, Location.X], bm);
      }
    }

    private void field1_Magnifier(object sender, MouseEventArgs e, Point ScreenLocation) {
      if (e.Button == MouseButtons.Left)
        paintMagnifier(e.Location, ScreenLocation);
      // else if (magnifierForm != null)
      // magnifierForm.Hide();
    }
    #endregion

  }

  public struct Range {
    public int Min, Max;
    public Range(int Min, int Max) {
      this.Min = Min; this.Max = Max;
    }
  }

}
