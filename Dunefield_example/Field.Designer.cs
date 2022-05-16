namespace DunefieldModel {
  partial class Field {
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
      this.pictureBox_Legend = new System.Windows.Forms.PictureBox();
      this.label_Caption = new System.Windows.Forms.Label();
      this.pictureBox_Field = new System.Windows.Forms.PictureBox();
      this.panel_Side = new System.Windows.Forms.Panel();
      this.slider1 = new DunefieldModel.Slider();
      this.panel_Bottom = new System.Windows.Forms.Panel();
      this.panel_Top = new System.Windows.Forms.Panel();
      ((System.ComponentModel.ISupportInitialize)(this.pictureBox_Legend)).BeginInit();
      ((System.ComponentModel.ISupportInitialize)(this.pictureBox_Field)).BeginInit();
      this.panel_Side.SuspendLayout();
      this.panel_Bottom.SuspendLayout();
      this.SuspendLayout();
      // 
      // pictureBox_Legend
      // 
      this.pictureBox_Legend.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
      this.pictureBox_Legend.Location = new System.Drawing.Point(0, 8);
      this.pictureBox_Legend.Name = "pictureBox_Legend";
      this.pictureBox_Legend.Size = new System.Drawing.Size(190, 23);
      this.pictureBox_Legend.TabIndex = 0;
      this.pictureBox_Legend.TabStop = false;
      // 
      // label_Caption
      // 
      this.label_Caption.AutoSize = true;
      this.label_Caption.Location = new System.Drawing.Point(193, 13);
      this.label_Caption.Name = "label_Caption";
      this.label_Caption.Size = new System.Drawing.Size(43, 13);
      this.label_Caption.TabIndex = 1;
      this.label_Caption.Text = "Caption";
      // 
      // pictureBox_Field
      // 
      this.pictureBox_Field.BackColor = System.Drawing.Color.Black;
      this.pictureBox_Field.Dock = System.Windows.Forms.DockStyle.Fill;
      this.pictureBox_Field.Location = new System.Drawing.Point(17, 7);
      this.pictureBox_Field.Name = "pictureBox_Field";
      this.pictureBox_Field.Size = new System.Drawing.Size(423, 198);
      this.pictureBox_Field.TabIndex = 2;
      this.pictureBox_Field.TabStop = false;
      this.pictureBox_Field.MouseMove += new System.Windows.Forms.MouseEventHandler(this.pictureBox_Field_MouseMove);
      this.pictureBox_Field.MouseDown += new System.Windows.Forms.MouseEventHandler(this.pictureBox_Field_MouseDown);
      this.pictureBox_Field.MouseUp += new System.Windows.Forms.MouseEventHandler(this.pictureBox_Field_MouseUp);
      // 
      // panel_Side
      // 
      this.panel_Side.Controls.Add(this.slider1);
      this.panel_Side.Dock = System.Windows.Forms.DockStyle.Left;
      this.panel_Side.Location = new System.Drawing.Point(0, 0);
      this.panel_Side.Name = "panel_Side";
      this.panel_Side.Size = new System.Drawing.Size(17, 238);
      this.panel_Side.TabIndex = 3;
      // 
      // slider1
      // 
      this.slider1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
                  | System.Windows.Forms.AnchorStyles.Left)
                  | System.Windows.Forms.AnchorStyles.Right)));
      this.slider1.Location = new System.Drawing.Point(0, 0);
      this.slider1.Name = "slider1";
      this.slider1.Size = new System.Drawing.Size(17, 213);
      this.slider1.TabIndex = 0;
      this.slider1.SliderMove += new DunefieldModel.Slider.MovementHandler(this.slider1_SliderMove);
      // 
      // panel_Bottom
      // 
      this.panel_Bottom.Controls.Add(this.pictureBox_Legend);
      this.panel_Bottom.Controls.Add(this.label_Caption);
      this.panel_Bottom.Dock = System.Windows.Forms.DockStyle.Bottom;
      this.panel_Bottom.Location = new System.Drawing.Point(17, 205);
      this.panel_Bottom.Name = "panel_Bottom";
      this.panel_Bottom.Size = new System.Drawing.Size(423, 33);
      this.panel_Bottom.TabIndex = 4;
      // 
      // panel_Top
      // 
      this.panel_Top.Dock = System.Windows.Forms.DockStyle.Top;
      this.panel_Top.Location = new System.Drawing.Point(17, 0);
      this.panel_Top.Name = "panel_Top";
      this.panel_Top.Size = new System.Drawing.Size(423, 7);
      this.panel_Top.TabIndex = 5;
      // 
      // Field
      // 
      this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
      this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
      this.Controls.Add(this.pictureBox_Field);
      this.Controls.Add(this.panel_Top);
      this.Controls.Add(this.panel_Bottom);
      this.Controls.Add(this.panel_Side);
      this.Name = "Field";
      this.Size = new System.Drawing.Size(440, 238);
      this.Load += new System.EventHandler(this.Field_Load);
      ((System.ComponentModel.ISupportInitialize)(this.pictureBox_Legend)).EndInit();
      ((System.ComponentModel.ISupportInitialize)(this.pictureBox_Field)).EndInit();
      this.panel_Side.ResumeLayout(false);
      this.panel_Bottom.ResumeLayout(false);
      this.panel_Bottom.PerformLayout();
      this.ResumeLayout(false);

    }

    #endregion

    private System.Windows.Forms.PictureBox pictureBox_Legend;
    private System.Windows.Forms.Label label_Caption;
    private System.Windows.Forms.PictureBox pictureBox_Field;
    private System.Windows.Forms.Panel panel_Side;
    private Slider slider1;
    private System.Windows.Forms.Panel panel_Bottom;
    private System.Windows.Forms.Panel panel_Top;
  }
}
