namespace DunefieldModel {
  partial class ChartAxis {
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

    #region Component Designer generated code

    /// <summary> 
    /// Required method for Designer support - do not modify 
    /// the contents of this method with the code editor.
    /// </summary>
    private void InitializeComponent() {
      this.textBox_Max = new System.Windows.Forms.TextBox();
      this.textBox_Min = new System.Windows.Forms.TextBox();
      this.label_Max = new System.Windows.Forms.Label();
      this.label_Min = new System.Windows.Forms.Label();
      this.checkBox_Auto = new System.Windows.Forms.CheckBox();
      this.SuspendLayout();
      // 
      // textBox_Max
      // 
      this.textBox_Max.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
      this.textBox_Max.BackColor = System.Drawing.SystemColors.Window;
      this.textBox_Max.Location = new System.Drawing.Point(26, 2);
      this.textBox_Max.Name = "textBox_Max";
      this.textBox_Max.Size = new System.Drawing.Size(25, 20);
      this.textBox_Max.TabIndex = 0;
      this.textBox_Max.Text = "500";
      this.textBox_Max.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
      this.textBox_Max.Validated += new System.EventHandler(this.textBox_Max_Validated);
      // 
      // textBox_Min
      // 
      this.textBox_Min.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
      this.textBox_Min.BackColor = System.Drawing.SystemColors.Window;
      this.textBox_Min.Location = new System.Drawing.Point(26, 61);
      this.textBox_Min.Name = "textBox_Min";
      this.textBox_Min.Size = new System.Drawing.Size(25, 20);
      this.textBox_Min.TabIndex = 1;
      this.textBox_Min.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
      this.textBox_Min.Validated += new System.EventHandler(this.textBox_Min_Validated);
      // 
      // label_Max
      // 
      this.label_Max.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
      this.label_Max.Location = new System.Drawing.Point(26, 24);
      this.label_Max.Name = "label_Max";
      this.label_Max.Size = new System.Drawing.Size(25, 13);
      this.label_Max.TabIndex = 2;
      this.label_Max.Text = "500";
      this.label_Max.TextAlign = System.Drawing.ContentAlignment.TopRight;
      this.label_Max.Click += new System.EventHandler(this.label_Max_Click);
      // 
      // label_Min
      // 
      this.label_Min.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
      this.label_Min.Location = new System.Drawing.Point(26, 47);
      this.label_Min.Name = "label_Min";
      this.label_Min.Size = new System.Drawing.Size(25, 13);
      this.label_Min.TabIndex = 3;
      this.label_Min.Text = "500";
      this.label_Min.TextAlign = System.Drawing.ContentAlignment.TopRight;
      this.label_Min.Click += new System.EventHandler(this.label_Min_Click);
      // 
      // checkBox_Auto
      // 
      this.checkBox_Auto.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
      this.checkBox_Auto.AutoSize = true;
      this.checkBox_Auto.Location = new System.Drawing.Point(6, 83);
      this.checkBox_Auto.Name = "checkBox_Auto";
      this.checkBox_Auto.Size = new System.Drawing.Size(48, 17);
      this.checkBox_Auto.TabIndex = 4;
      this.checkBox_Auto.Text = "Auto";
      this.checkBox_Auto.UseVisualStyleBackColor = true;
      // 
      // ChartAxis
      // 
      this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
      this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
      this.Controls.Add(this.checkBox_Auto);
      this.Controls.Add(this.label_Min);
      this.Controls.Add(this.label_Max);
      this.Controls.Add(this.textBox_Min);
      this.Controls.Add(this.textBox_Max);
      this.Name = "ChartAxis";
      this.Size = new System.Drawing.Size(53, 99);
      this.Paint += new System.Windows.Forms.PaintEventHandler(this.ChartAxis_Paint);
      this.ResumeLayout(false);
      this.PerformLayout();

    }

    #endregion

    private System.Windows.Forms.TextBox textBox_Max;
    private System.Windows.Forms.TextBox textBox_Min;
    private System.Windows.Forms.Label label_Max;
    private System.Windows.Forms.Label label_Min;
    private System.Windows.Forms.CheckBox checkBox_Auto;
  }
}
