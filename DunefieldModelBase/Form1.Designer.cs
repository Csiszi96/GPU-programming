namespace DunefieldModel {
  partial class Form1 {
    /// <summary>
    /// Required designer variable.
    /// </summary>
    private System.ComponentModel.IContainer components = null;

    /// <summary>
    /// Clean up any resources being used.
    /// </summary>
    /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
    protected override void Dispose(bool disposing) {
      if (disposing && (components != null)) {
        components.Dispose();
      }
      base.Dispose(disposing);
    }

    #region Windows Form Designer generated code

    /// <summary>
    /// Required method for Designer support - do not modify
    /// the contents of this method with the code editor.
    /// </summary>
    private void InitializeComponent() {
      this.components = new System.ComponentModel.Container();
      this.button_Tick = new System.Windows.Forms.Button();
      this.button_Run = new System.Windows.Forms.Button();
      this.groupBox1 = new System.Windows.Forms.GroupBox();
      this.label12 = new System.Windows.Forms.Label();
      this.textBox_hopLength = new System.Windows.Forms.TextBox();
      this.checkBox_Recycle = new System.Windows.Forms.CheckBox();
      this.radioButton_TruncTrans = new System.Windows.Forms.RadioButton();
      this.radioButton_Dune = new System.Windows.Forms.RadioButton();
      this.label2 = new System.Windows.Forms.Label();
      this.textBox_pNoSand = new System.Windows.Forms.TextBox();
      this.label1 = new System.Windows.Forms.Label();
      this.textBox_pSand = new System.Windows.Forms.TextBox();
      this.radioButton_File = new System.Windows.Forms.RadioButton();
      this.label_InitStatus = new System.Windows.Forms.Label();
      this.button_Initialize = new System.Windows.Forms.Button();
      this.label5 = new System.Windows.Forms.Label();
      this.textBox_SandDepth = new System.Windows.Forms.TextBox();
      this.label4 = new System.Windows.Forms.Label();
      this.textBox_Width = new System.Windows.Forms.TextBox();
      this.label3 = new System.Windows.Forms.Label();
      this.textBox_Height = new System.Windows.Forms.TextBox();
      this.label_Filename = new System.Windows.Forms.Label();
      this.button_Browse = new System.Windows.Forms.Button();
      this.radioButton_Square = new System.Windows.Forms.RadioButton();
      this.radioButton_Uniform = new System.Windows.Forms.RadioButton();
      this.radioButton_Random = new System.Windows.Forms.RadioButton();
      this.button_Save = new System.Windows.Forms.Button();
      this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
      this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
      this.groupBox2 = new System.Windows.Forms.GroupBox();
      this.radioButton_RecordDune = new System.Windows.Forms.RadioButton();
      this.radioButton_RecordWindow = new System.Windows.Forms.RadioButton();
      this.label_RecordingStatus = new System.Windows.Forms.Label();
      this.button_MakeRecording = new System.Windows.Forms.Button();
      this.checkBox_MakeRecording = new System.Windows.Forms.CheckBox();
      this.label6 = new System.Windows.Forms.Label();
      this.textBox_RecordingFile = new System.Windows.Forms.TextBox();
      this.textBox_RecordingDir = new System.Windows.Forms.TextBox();
      this.button_RecordingDir = new System.Windows.Forms.Button();
      this.folderBrowserDialog1 = new System.Windows.Forms.FolderBrowserDialog();
      this.label_ElapsedTime = new System.Windows.Forms.Label();
      this.listBox_Show = new System.Windows.Forms.ListBox();
      this.label9 = new System.Windows.Forms.Label();
      this.label_DateTime = new System.Windows.Forms.Label();
      this.label_GrainCount = new System.Windows.Forms.Label();
      this.comboBox_Model = new System.Windows.Forms.ComboBox();
      this.label11 = new System.Windows.Forms.Label();
      this.timer1 = new System.Windows.Forms.Timer(this.components);
      this.comboBox1 = new System.Windows.Forms.ComboBox();
      this.label7 = new System.Windows.Forms.Label();
      this.comboBox_Neighbourhood = new System.Windows.Forms.ComboBox();
      this.textBox_ticksPerRefresh = new System.Windows.Forms.TextBox();
      this.label8 = new System.Windows.Forms.Label();
      this.label10 = new System.Windows.Forms.Label();
      this.textBox_tStop = new System.Windows.Forms.TextBox();
      this.field1 = new DunefieldModel.Field();
      this.chartAxis_Dunefield = new DunefieldModel.ChartAxis();
      this.chartAxis2 = new DunefieldModel.ChartAxis();
      this.chart1 = new DunefieldModel.Chart();
      this.chartAxis1 = new DunefieldModel.ChartAxis();
      this.groupBox1.SuspendLayout();
      this.groupBox2.SuspendLayout();
      this.SuspendLayout();
      // 
      // button_Tick
      // 
      this.button_Tick.Location = new System.Drawing.Point(12, 354);
      this.button_Tick.Name = "button_Tick";
      this.button_Tick.Size = new System.Drawing.Size(43, 26);
      this.button_Tick.TabIndex = 1;
      this.button_Tick.Text = "Tick";
      this.button_Tick.UseVisualStyleBackColor = true;
      this.button_Tick.Click += new System.EventHandler(this.button_Tick_Click);
      // 
      // button_Run
      // 
      this.button_Run.Location = new System.Drawing.Point(81, 354);
      this.button_Run.Name = "button_Run";
      this.button_Run.Size = new System.Drawing.Size(43, 26);
      this.button_Run.TabIndex = 2;
      this.button_Run.Text = "Run";
      this.button_Run.UseVisualStyleBackColor = true;
      this.button_Run.Click += new System.EventHandler(this.button_Run_Click);
      // 
      // groupBox1
      // 
      this.groupBox1.Controls.Add(this.label12);
      this.groupBox1.Controls.Add(this.textBox_hopLength);
      this.groupBox1.Controls.Add(this.checkBox_Recycle);
      this.groupBox1.Controls.Add(this.radioButton_TruncTrans);
      this.groupBox1.Controls.Add(this.radioButton_Dune);
      this.groupBox1.Controls.Add(this.label2);
      this.groupBox1.Controls.Add(this.textBox_pNoSand);
      this.groupBox1.Controls.Add(this.label1);
      this.groupBox1.Controls.Add(this.textBox_pSand);
      this.groupBox1.Controls.Add(this.radioButton_File);
      this.groupBox1.Controls.Add(this.label_InitStatus);
      this.groupBox1.Controls.Add(this.button_Initialize);
      this.groupBox1.Controls.Add(this.label5);
      this.groupBox1.Controls.Add(this.textBox_SandDepth);
      this.groupBox1.Controls.Add(this.label4);
      this.groupBox1.Controls.Add(this.textBox_Width);
      this.groupBox1.Controls.Add(this.label3);
      this.groupBox1.Controls.Add(this.textBox_Height);
      this.groupBox1.Controls.Add(this.label_Filename);
      this.groupBox1.Controls.Add(this.button_Browse);
      this.groupBox1.Controls.Add(this.radioButton_Square);
      this.groupBox1.Controls.Add(this.radioButton_Uniform);
      this.groupBox1.Controls.Add(this.radioButton_Random);
      this.groupBox1.Location = new System.Drawing.Point(12, 162);
      this.groupBox1.Name = "groupBox1";
      this.groupBox1.Size = new System.Drawing.Size(307, 181);
      this.groupBox1.TabIndex = 7;
      this.groupBox1.TabStop = false;
      this.groupBox1.Text = "Initial conditions";
      // 
      // label12
      // 
      this.label12.AutoSize = true;
      this.label12.Location = new System.Drawing.Point(255, 98);
      this.label12.Name = "label12";
      this.label12.Size = new System.Drawing.Size(40, 13);
      this.label12.TabIndex = 22;
      this.label12.Text = "L (hop)";
      // 
      // textBox_hopLength
      // 
      this.textBox_hopLength.Location = new System.Drawing.Point(223, 95);
      this.textBox_hopLength.Name = "textBox_hopLength";
      this.textBox_hopLength.Size = new System.Drawing.Size(30, 20);
      this.textBox_hopLength.TabIndex = 21;
      // 
      // checkBox_Recycle
      // 
      this.checkBox_Recycle.AutoSize = true;
      this.checkBox_Recycle.Location = new System.Drawing.Point(238, 120);
      this.checkBox_Recycle.Name = "checkBox_Recycle";
      this.checkBox_Recycle.Size = new System.Drawing.Size(65, 17);
      this.checkBox_Recycle.TabIndex = 20;
      this.checkBox_Recycle.Text = "Recycle";
      this.checkBox_Recycle.UseVisualStyleBackColor = true;
      // 
      // radioButton_TruncTrans
      // 
      this.radioButton_TruncTrans.AutoSize = true;
      this.radioButton_TruncTrans.Location = new System.Drawing.Point(12, 117);
      this.radioButton_TruncTrans.Name = "radioButton_TruncTrans";
      this.radioButton_TruncTrans.Size = new System.Drawing.Size(106, 17);
      this.radioButton_TruncTrans.TabIndex = 19;
      this.radioButton_TruncTrans.TabStop = true;
      this.radioButton_TruncTrans.Text = "Trunc trans dune";
      this.radioButton_TruncTrans.UseVisualStyleBackColor = true;
      // 
      // radioButton_Dune
      // 
      this.radioButton_Dune.AutoSize = true;
      this.radioButton_Dune.Location = new System.Drawing.Point(12, 99);
      this.radioButton_Dune.Name = "radioButton_Dune";
      this.radioButton_Dune.Size = new System.Drawing.Size(105, 17);
      this.radioButton_Dune.TabIndex = 18;
      this.radioButton_Dune.TabStop = true;
      this.radioButton_Dune.Text = "Transverse dune";
      this.radioButton_Dune.UseVisualStyleBackColor = true;
      // 
      // label2
      // 
      this.label2.AutoSize = true;
      this.label2.Location = new System.Drawing.Point(172, 121);
      this.label2.Name = "label2";
      this.label2.Size = new System.Drawing.Size(58, 13);
      this.label2.TabIndex = 17;
      this.label2.Text = "p(NoSand)";
      // 
      // textBox_pNoSand
      // 
      this.textBox_pNoSand.Location = new System.Drawing.Point(140, 118);
      this.textBox_pNoSand.Name = "textBox_pNoSand";
      this.textBox_pNoSand.Size = new System.Drawing.Size(30, 20);
      this.textBox_pNoSand.TabIndex = 16;
      // 
      // label1
      // 
      this.label1.AutoSize = true;
      this.label1.Location = new System.Drawing.Point(172, 98);
      this.label1.Name = "label1";
      this.label1.Size = new System.Drawing.Size(44, 13);
      this.label1.TabIndex = 15;
      this.label1.Text = "p(Sand)";
      // 
      // textBox_pSand
      // 
      this.textBox_pSand.Location = new System.Drawing.Point(140, 95);
      this.textBox_pSand.Name = "textBox_pSand";
      this.textBox_pSand.Size = new System.Drawing.Size(30, 20);
      this.textBox_pSand.TabIndex = 14;
      // 
      // radioButton_File
      // 
      this.radioButton_File.AutoSize = true;
      this.radioButton_File.Location = new System.Drawing.Point(12, 19);
      this.radioButton_File.Name = "radioButton_File";
      this.radioButton_File.Size = new System.Drawing.Size(41, 17);
      this.radioButton_File.TabIndex = 13;
      this.radioButton_File.TabStop = true;
      this.radioButton_File.Text = "File";
      this.radioButton_File.UseVisualStyleBackColor = true;
      // 
      // label_InitStatus
      // 
      this.label_InitStatus.AutoSize = true;
      this.label_InitStatus.Location = new System.Drawing.Point(78, 154);
      this.label_InitStatus.Name = "label_InitStatus";
      this.label_InitStatus.Size = new System.Drawing.Size(35, 13);
      this.label_InitStatus.TabIndex = 12;
      this.label_InitStatus.Text = "label6";
      // 
      // button_Initialize
      // 
      this.button_Initialize.Location = new System.Drawing.Point(12, 148);
      this.button_Initialize.Name = "button_Initialize";
      this.button_Initialize.Size = new System.Drawing.Size(55, 23);
      this.button_Initialize.TabIndex = 11;
      this.button_Initialize.Text = "Initialize";
      this.button_Initialize.UseVisualStyleBackColor = true;
      this.button_Initialize.Click += new System.EventHandler(this.button_Initialize_Click);
      // 
      // label5
      // 
      this.label5.AutoSize = true;
      this.label5.Location = new System.Drawing.Point(172, 67);
      this.label5.Name = "label5";
      this.label5.Size = new System.Drawing.Size(64, 13);
      this.label5.TabIndex = 10;
      this.label5.Text = "Sand height";
      // 
      // textBox_SandDepth
      // 
      this.textBox_SandDepth.Location = new System.Drawing.Point(140, 64);
      this.textBox_SandDepth.Name = "textBox_SandDepth";
      this.textBox_SandDepth.Size = new System.Drawing.Size(30, 20);
      this.textBox_SandDepth.TabIndex = 9;
      // 
      // label4
      // 
      this.label4.AutoSize = true;
      this.label4.Location = new System.Drawing.Point(220, 44);
      this.label4.Name = "label4";
      this.label4.Size = new System.Drawing.Size(79, 13);
      this.label4.TabIndex = 8;
      this.label4.Text = "Length x Width";
      // 
      // textBox_Width
      // 
      this.textBox_Width.Location = new System.Drawing.Point(140, 39);
      this.textBox_Width.Name = "textBox_Width";
      this.textBox_Width.Size = new System.Drawing.Size(30, 20);
      this.textBox_Width.TabIndex = 5;
      // 
      // label3
      // 
      this.label3.AutoSize = true;
      this.label3.Location = new System.Drawing.Point(172, 44);
      this.label3.Name = "label3";
      this.label3.Size = new System.Drawing.Size(14, 13);
      this.label3.TabIndex = 6;
      this.label3.Text = "X";
      // 
      // textBox_Height
      // 
      this.textBox_Height.Location = new System.Drawing.Point(186, 39);
      this.textBox_Height.Name = "textBox_Height";
      this.textBox_Height.Size = new System.Drawing.Size(30, 20);
      this.textBox_Height.TabIndex = 6;
      // 
      // label_Filename
      // 
      this.label_Filename.AutoSize = true;
      this.label_Filename.Location = new System.Drawing.Point(112, 20);
      this.label_Filename.Name = "label_Filename";
      this.label_Filename.Size = new System.Drawing.Size(35, 13);
      this.label_Filename.TabIndex = 4;
      this.label_Filename.Text = "label2";
      // 
      // button_Browse
      // 
      this.button_Browse.Location = new System.Drawing.Point(56, 17);
      this.button_Browse.Name = "button_Browse";
      this.button_Browse.Size = new System.Drawing.Size(53, 21);
      this.button_Browse.TabIndex = 3;
      this.button_Browse.Text = "Browse";
      this.button_Browse.UseVisualStyleBackColor = true;
      this.button_Browse.Click += new System.EventHandler(this.button_Browse_Click);
      // 
      // radioButton_Square
      // 
      this.radioButton_Square.AutoSize = true;
      this.radioButton_Square.Location = new System.Drawing.Point(12, 79);
      this.radioButton_Square.Name = "radioButton_Square";
      this.radioButton_Square.Size = new System.Drawing.Size(61, 17);
      this.radioButton_Square.TabIndex = 2;
      this.radioButton_Square.TabStop = true;
      this.radioButton_Square.Text = "Plateau";
      this.radioButton_Square.UseVisualStyleBackColor = true;
      // 
      // radioButton_Uniform
      // 
      this.radioButton_Uniform.AutoSize = true;
      this.radioButton_Uniform.Location = new System.Drawing.Point(12, 59);
      this.radioButton_Uniform.Name = "radioButton_Uniform";
      this.radioButton_Uniform.Size = new System.Drawing.Size(61, 17);
      this.radioButton_Uniform.TabIndex = 1;
      this.radioButton_Uniform.TabStop = true;
      this.radioButton_Uniform.Text = "Uniform";
      this.radioButton_Uniform.UseVisualStyleBackColor = true;
      // 
      // radioButton_Random
      // 
      this.radioButton_Random.AutoSize = true;
      this.radioButton_Random.Location = new System.Drawing.Point(12, 39);
      this.radioButton_Random.Name = "radioButton_Random";
      this.radioButton_Random.Size = new System.Drawing.Size(65, 17);
      this.radioButton_Random.TabIndex = 0;
      this.radioButton_Random.TabStop = true;
      this.radioButton_Random.Text = "Random";
      this.radioButton_Random.UseVisualStyleBackColor = true;
      // 
      // button_Save
      // 
      this.button_Save.Location = new System.Drawing.Point(12, 428);
      this.button_Save.Name = "button_Save";
      this.button_Save.Size = new System.Drawing.Size(43, 26);
      this.button_Save.TabIndex = 8;
      this.button_Save.Text = "Save";
      this.button_Save.UseVisualStyleBackColor = true;
      this.button_Save.Click += new System.EventHandler(this.button_Save_Click);
      // 
      // openFileDialog1
      // 
      this.openFileDialog1.FileName = "openFileDialog1";
      // 
      // groupBox2
      // 
      this.groupBox2.Controls.Add(this.radioButton_RecordDune);
      this.groupBox2.Controls.Add(this.radioButton_RecordWindow);
      this.groupBox2.Controls.Add(this.label_RecordingStatus);
      this.groupBox2.Controls.Add(this.button_MakeRecording);
      this.groupBox2.Controls.Add(this.checkBox_MakeRecording);
      this.groupBox2.Controls.Add(this.label6);
      this.groupBox2.Controls.Add(this.textBox_RecordingFile);
      this.groupBox2.Controls.Add(this.textBox_RecordingDir);
      this.groupBox2.Controls.Add(this.button_RecordingDir);
      this.groupBox2.Location = new System.Drawing.Point(12, 467);
      this.groupBox2.Name = "groupBox2";
      this.groupBox2.Size = new System.Drawing.Size(307, 156);
      this.groupBox2.TabIndex = 9;
      this.groupBox2.TabStop = false;
      this.groupBox2.Text = "Recording parameters";
      // 
      // radioButton_RecordDune
      // 
      this.radioButton_RecordDune.AutoSize = true;
      this.radioButton_RecordDune.Location = new System.Drawing.Point(17, 103);
      this.radioButton_RecordDune.Name = "radioButton_RecordDune";
      this.radioButton_RecordDune.Size = new System.Drawing.Size(95, 17);
      this.radioButton_RecordDune.TabIndex = 8;
      this.radioButton_RecordDune.TabStop = true;
      this.radioButton_RecordDune.Text = "Dune field only";
      this.radioButton_RecordDune.UseVisualStyleBackColor = true;
      // 
      // radioButton_RecordWindow
      // 
      this.radioButton_RecordWindow.AutoSize = true;
      this.radioButton_RecordWindow.Location = new System.Drawing.Point(17, 82);
      this.radioButton_RecordWindow.Name = "radioButton_RecordWindow";
      this.radioButton_RecordWindow.Size = new System.Drawing.Size(80, 17);
      this.radioButton_RecordWindow.TabIndex = 7;
      this.radioButton_RecordWindow.TabStop = true;
      this.radioButton_RecordWindow.Text = "Full window";
      this.radioButton_RecordWindow.UseVisualStyleBackColor = true;
      // 
      // label_RecordingStatus
      // 
      this.label_RecordingStatus.Location = new System.Drawing.Point(118, 132);
      this.label_RecordingStatus.Name = "label_RecordingStatus";
      this.label_RecordingStatus.Size = new System.Drawing.Size(179, 13);
      this.label_RecordingStatus.TabIndex = 6;
      this.label_RecordingStatus.Text = "label7";
      this.label_RecordingStatus.TextAlign = System.Drawing.ContentAlignment.TopRight;
      // 
      // button_MakeRecording
      // 
      this.button_MakeRecording.Location = new System.Drawing.Point(210, 79);
      this.button_MakeRecording.Name = "button_MakeRecording";
      this.button_MakeRecording.Size = new System.Drawing.Size(87, 26);
      this.button_MakeRecording.TabIndex = 5;
      this.button_MakeRecording.Text = "Record now";
      this.button_MakeRecording.UseVisualStyleBackColor = true;
      this.button_MakeRecording.Click += new System.EventHandler(this.button_MakeRecording_Click);
      // 
      // checkBox_MakeRecording
      // 
      this.checkBox_MakeRecording.AutoSize = true;
      this.checkBox_MakeRecording.Location = new System.Drawing.Point(210, 108);
      this.checkBox_MakeRecording.Name = "checkBox_MakeRecording";
      this.checkBox_MakeRecording.Size = new System.Drawing.Size(79, 17);
      this.checkBox_MakeRecording.TabIndex = 4;
      this.checkBox_MakeRecording.Text = "Continuous";
      this.checkBox_MakeRecording.UseVisualStyleBackColor = true;
      this.checkBox_MakeRecording.CheckedChanged += new System.EventHandler(this.checkBox_MakeRecording_CheckedChanged);
      // 
      // label6
      // 
      this.label6.AutoSize = true;
      this.label6.Location = new System.Drawing.Point(18, 54);
      this.label6.Name = "label6";
      this.label6.Size = new System.Drawing.Size(76, 13);
      this.label6.TabIndex = 3;
      this.label6.Text = "Base filename:";
      this.label6.TextAlign = System.Drawing.ContentAlignment.TopRight;
      // 
      // textBox_RecordingFile
      // 
      this.textBox_RecordingFile.Location = new System.Drawing.Point(97, 51);
      this.textBox_RecordingFile.Name = "textBox_RecordingFile";
      this.textBox_RecordingFile.Size = new System.Drawing.Size(200, 20);
      this.textBox_RecordingFile.TabIndex = 2;
      // 
      // textBox_RecordingDir
      // 
      this.textBox_RecordingDir.Location = new System.Drawing.Point(97, 25);
      this.textBox_RecordingDir.Name = "textBox_RecordingDir";
      this.textBox_RecordingDir.Size = new System.Drawing.Size(200, 20);
      this.textBox_RecordingDir.TabIndex = 1;
      // 
      // button_RecordingDir
      // 
      this.button_RecordingDir.Location = new System.Drawing.Point(13, 21);
      this.button_RecordingDir.Name = "button_RecordingDir";
      this.button_RecordingDir.Size = new System.Drawing.Size(78, 27);
      this.button_RecordingDir.TabIndex = 0;
      this.button_RecordingDir.Text = "Set directory";
      this.button_RecordingDir.UseVisualStyleBackColor = true;
      this.button_RecordingDir.Click += new System.EventHandler(this.button_RecordingDir_Click);
      // 
      // label_ElapsedTime
      // 
      this.label_ElapsedTime.AutoSize = true;
      this.label_ElapsedTime.Location = new System.Drawing.Point(12, 390);
      this.label_ElapsedTime.Name = "label_ElapsedTime";
      this.label_ElapsedTime.Size = new System.Drawing.Size(111, 13);
      this.label_ElapsedTime.TabIndex = 10;
      this.label_ElapsedTime.Text = "T=0 elapsed 00:00:00";
      // 
      // listBox_Show
      // 
      this.listBox_Show.BackColor = System.Drawing.SystemColors.ActiveBorder;
      this.listBox_Show.FormattingEnabled = true;
      this.listBox_Show.Items.AddRange(new object[] {
            "Elevation",
            "Shadow",
            "Saltation",
            "Custom A"});
      this.listBox_Show.Location = new System.Drawing.Point(257, 362);
      this.listBox_Show.Name = "listBox_Show";
      this.listBox_Show.Size = new System.Drawing.Size(62, 56);
      this.listBox_Show.TabIndex = 19;
      this.listBox_Show.SelectedIndexChanged += new System.EventHandler(this.listBox_Show_SelectedIndexChanged);
      // 
      // label9
      // 
      this.label9.AutoSize = true;
      this.label9.Location = new System.Drawing.Point(258, 346);
      this.label9.Name = "label9";
      this.label9.Size = new System.Drawing.Size(37, 13);
      this.label9.TabIndex = 20;
      this.label9.Text = "Show:";
      // 
      // label_DateTime
      // 
      this.label_DateTime.Location = new System.Drawing.Point(246, 441);
      this.label_DateTime.Name = "label_DateTime";
      this.label_DateTime.Size = new System.Drawing.Size(73, 28);
      this.label_DateTime.TabIndex = 21;
      this.label_DateTime.Text = "label10";
      this.label_DateTime.TextAlign = System.Drawing.ContentAlignment.TopRight;
      // 
      // label_GrainCount
      // 
      this.label_GrainCount.AutoSize = true;
      this.label_GrainCount.Location = new System.Drawing.Point(12, 406);
      this.label_GrainCount.Name = "label_GrainCount";
      this.label_GrainCount.Size = new System.Drawing.Size(69, 13);
      this.label_GrainCount.TabIndex = 22;
      this.label_GrainCount.Text = "Grains: 100%";
      // 
      // comboBox_Model
      // 
      this.comboBox_Model.FormattingEnabled = true;
      this.comboBox_Model.Location = new System.Drawing.Point(47, 112);
      this.comboBox_Model.Name = "comboBox_Model";
      this.comboBox_Model.Size = new System.Drawing.Size(272, 21);
      this.comboBox_Model.TabIndex = 25;
      this.comboBox_Model.SelectedIndexChanged += new System.EventHandler(this.comboBox1_SelectedIndexChanged);
      // 
      // label11
      // 
      this.label11.AutoSize = true;
      this.label11.Location = new System.Drawing.Point(9, 115);
      this.label11.Name = "label11";
      this.label11.Size = new System.Drawing.Size(39, 13);
      this.label11.TabIndex = 26;
      this.label11.Text = "Model:";
      // 
      // timer1
      // 
      this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
      // 
      // comboBox1
      // 
      this.comboBox1.FormattingEnabled = true;
      this.comboBox1.Location = new System.Drawing.Point(53, 112);
      this.comboBox1.Name = "comboBox1";
      this.comboBox1.Size = new System.Drawing.Size(266, 21);
      this.comboBox1.TabIndex = 25;
      // 
      // label7
      // 
      this.label7.AutoSize = true;
      this.label7.Location = new System.Drawing.Point(50, 142);
      this.label7.Name = "label7";
      this.label7.Size = new System.Drawing.Size(83, 13);
      this.label7.TabIndex = 35;
      this.label7.Text = "Neighbourhood:";
      // 
      // comboBox_Neighbourhood
      // 
      this.comboBox_Neighbourhood.FormattingEnabled = true;
      this.comboBox_Neighbourhood.Location = new System.Drawing.Point(133, 139);
      this.comboBox_Neighbourhood.Name = "comboBox_Neighbourhood";
      this.comboBox_Neighbourhood.Size = new System.Drawing.Size(186, 21);
      this.comboBox_Neighbourhood.TabIndex = 34;
      this.comboBox_Neighbourhood.SelectedIndexChanged += new System.EventHandler(this.comboBox_Neighbourhood_SelectedIndexChanged);
      // 
      // textBox_ticksPerRefresh
      // 
      this.textBox_ticksPerRefresh.Location = new System.Drawing.Point(135, 354);
      this.textBox_ticksPerRefresh.Name = "textBox_ticksPerRefresh";
      this.textBox_ticksPerRefresh.Size = new System.Drawing.Size(32, 20);
      this.textBox_ticksPerRefresh.TabIndex = 36;
      // 
      // label8
      // 
      this.label8.AutoSize = true;
      this.label8.Location = new System.Drawing.Point(167, 357);
      this.label8.Name = "label8";
      this.label8.Size = new System.Drawing.Size(70, 13);
      this.label8.TabIndex = 37;
      this.label8.Text = "Ticks/refresh";
      // 
      // label10
      // 
      this.label10.AutoSize = true;
      this.label10.Location = new System.Drawing.Point(167, 381);
      this.label10.Name = "label10";
      this.label10.Size = new System.Drawing.Size(40, 13);
      this.label10.TabIndex = 39;
      this.label10.Text = "T(stop)";
      // 
      // textBox_tStop
      // 
      this.textBox_tStop.Location = new System.Drawing.Point(135, 378);
      this.textBox_tStop.Name = "textBox_tStop";
      this.textBox_tStop.Size = new System.Drawing.Size(32, 20);
      this.textBox_tStop.TabIndex = 38;
      // 
      // field1
      // 
      this.field1.CaptionText = "";
      this.field1.Location = new System.Drawing.Point(332, 95);
      this.field1.Name = "field1";
      this.field1.Size = new System.Drawing.Size(596, 260);
      this.field1.TabIndex = 33;
      this.field1.SliderMove += new DunefieldModel.Field.MovementHandler(this.field1_SliderMove);
      this.field1.Magnifier += new DunefieldModel.Field.MagnifierHandler(this.field1_Magnifier);
      // 
      // chartAxis_Dunefield
      // 
      this.chartAxis_Dunefield.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
      this.chartAxis_Dunefield.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
      this.chartAxis_Dunefield.Location = new System.Drawing.Point(12, 11);
      this.chartAxis_Dunefield.Name = "chartAxis_Dunefield";
      this.chartAxis_Dunefield.Size = new System.Drawing.Size(52, 95);
      this.chartAxis_Dunefield.TabIndex = 29;
      this.chartAxis_Dunefield.RenderChart += new DunefieldModel.ChartAxis.RenderingHandler(this.chartAxis_Dunefield_RenderChart);
      // 
      // chartAxis2
      // 
      this.chartAxis2.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
      this.chartAxis2.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
      this.chartAxis2.Location = new System.Drawing.Point(206, 12);
      this.chartAxis2.Name = "chartAxis2";
      this.chartAxis2.Size = new System.Drawing.Size(52, 95);
      this.chartAxis2.TabIndex = 28;
      this.chartAxis2.RenderChart += new DunefieldModel.ChartAxis.RenderingHandler(this.chartAxis_RenderChart);
      // 
      // chart1
      // 
      this.chart1.Location = new System.Drawing.Point(347, 11);
      this.chart1.Name = "chart1";
      this.chart1.Size = new System.Drawing.Size(581, 78);
      this.chart1.TabIndex = 27;
      // 
      // chartAxis1
      // 
      this.chartAxis1.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
      this.chartAxis1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
      this.chartAxis1.Location = new System.Drawing.Point(267, 11);
      this.chartAxis1.Name = "chartAxis1";
      this.chartAxis1.Size = new System.Drawing.Size(52, 95);
      this.chartAxis1.TabIndex = 24;
      this.chartAxis1.RenderChart += new DunefieldModel.ChartAxis.RenderingHandler(this.chartAxis_RenderChart);
      // 
      // Form1
      // 
      this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
      this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
      this.ClientSize = new System.Drawing.Size(940, 707);
      this.Controls.Add(this.label10);
      this.Controls.Add(this.textBox_tStop);
      this.Controls.Add(this.label8);
      this.Controls.Add(this.textBox_ticksPerRefresh);
      this.Controls.Add(this.label7);
      this.Controls.Add(this.comboBox_Neighbourhood);
      this.Controls.Add(this.field1);
      this.Controls.Add(this.chartAxis_Dunefield);
      this.Controls.Add(this.chartAxis2);
      this.Controls.Add(this.chart1);
      this.Controls.Add(this.label11);
      this.Controls.Add(this.comboBox_Model);
      this.Controls.Add(this.chartAxis1);
      this.Controls.Add(this.label_GrainCount);
      this.Controls.Add(this.label_DateTime);
      this.Controls.Add(this.label9);
      this.Controls.Add(this.listBox_Show);
      this.Controls.Add(this.label_ElapsedTime);
      this.Controls.Add(this.groupBox2);
      this.Controls.Add(this.button_Save);
      this.Controls.Add(this.groupBox1);
      this.Controls.Add(this.button_Run);
      this.Controls.Add(this.button_Tick);
      this.Name = "Form1";
      this.Text = "Dunefield simulator";
      this.Load += new System.EventHandler(this.Form1_Load);
      this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.Form1_FormClosing);
      this.groupBox1.ResumeLayout(false);
      this.groupBox1.PerformLayout();
      this.groupBox2.ResumeLayout(false);
      this.groupBox2.PerformLayout();
      this.ResumeLayout(false);
      this.PerformLayout();

    }

    #endregion

    private System.Windows.Forms.Button button_Tick;
    private System.Windows.Forms.Button button_Run;
    private System.Windows.Forms.GroupBox groupBox1;
    private System.Windows.Forms.Label label_Filename;
    private System.Windows.Forms.Button button_Browse;
    private System.Windows.Forms.RadioButton radioButton_Square;
    private System.Windows.Forms.RadioButton radioButton_Uniform;
    private System.Windows.Forms.RadioButton radioButton_Random;
    private System.Windows.Forms.Label label3;
    private System.Windows.Forms.TextBox textBox_Height;
    private System.Windows.Forms.Button button_Save;
    private System.Windows.Forms.Label label5;
    private System.Windows.Forms.TextBox textBox_SandDepth;
    private System.Windows.Forms.Label label4;
    private System.Windows.Forms.TextBox textBox_Width;
    private System.Windows.Forms.Label label_InitStatus;
    private System.Windows.Forms.Button button_Initialize;
    private System.Windows.Forms.RadioButton radioButton_File;
    private System.Windows.Forms.Label label1;
    private System.Windows.Forms.TextBox textBox_pSand;
    private System.Windows.Forms.Label label2;
    private System.Windows.Forms.TextBox textBox_pNoSand;
    private System.Windows.Forms.SaveFileDialog saveFileDialog1;
    private System.Windows.Forms.OpenFileDialog openFileDialog1;
    private System.Windows.Forms.GroupBox groupBox2;
    private System.Windows.Forms.CheckBox checkBox_MakeRecording;
    private System.Windows.Forms.Label label6;
    private System.Windows.Forms.TextBox textBox_RecordingFile;
    private System.Windows.Forms.TextBox textBox_RecordingDir;
    private System.Windows.Forms.Button button_RecordingDir;
    private System.Windows.Forms.Label label_RecordingStatus;
    private System.Windows.Forms.Button button_MakeRecording;
    private System.Windows.Forms.FolderBrowserDialog folderBrowserDialog1;
    private System.Windows.Forms.Label label_ElapsedTime;
    private System.Windows.Forms.RadioButton radioButton_Dune;
    private System.Windows.Forms.ListBox listBox_Show;
    private System.Windows.Forms.Label label9;
    private System.Windows.Forms.RadioButton radioButton_RecordDune;
    private System.Windows.Forms.RadioButton radioButton_RecordWindow;
    private System.Windows.Forms.RadioButton radioButton_TruncTrans;
    private System.Windows.Forms.Label label_DateTime;
    private System.Windows.Forms.CheckBox checkBox_Recycle;
    private System.Windows.Forms.Label label_GrainCount;
    private ChartAxis chartAxis1;
    private System.Windows.Forms.ComboBox comboBox_Model;
    private System.Windows.Forms.Label label11;
    private Chart chart1;
    private ChartAxis chartAxis2;
    private ChartAxis chartAxis_Dunefield;
    private System.Windows.Forms.Timer timer1;
    private Field field1;
    private System.Windows.Forms.ComboBox comboBox1;
    private System.Windows.Forms.Label label7;
    private System.Windows.Forms.ComboBox comboBox_Neighbourhood;
    private System.Windows.Forms.TextBox textBox_ticksPerRefresh;
    private System.Windows.Forms.Label label8;
    private System.Windows.Forms.Label label10;
    private System.Windows.Forms.TextBox textBox_tStop;
    private System.Windows.Forms.Label label12;
    private System.Windows.Forms.TextBox textBox_hopLength;
  }
}

