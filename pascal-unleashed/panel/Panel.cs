using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using System.Windows.Forms;

internal static class Backend
{
    internal const string Dll = "PascalUnleashed.dll";
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    internal struct Device
    {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string Name;
        public int Major, Minor, Driver, Compatible;
        public ulong Memory;
    }
    [StructLayout(LayoutKind.Sequential)]
    internal struct Run
    {
        public int Mode, Reason;
        public double Milliseconds, Tflops;
        public ulong Accelerated, Native, Fallback, Ops, Workspace;
    }
    [StructLayout(LayoutKind.Sequential)]
    internal struct Bench
    {
        public Run Run;
        public double Error, MaxAbsolute, Multiplier, NativeMs;
        public ulong Bytes;
    }
    [DllImport(Dll, CallingConvention=CallingConvention.Cdecl)] internal static extern int pu_devices();
    [DllImport(Dll, CallingConvention=CallingConvention.Cdecl)] internal static extern int pu_device(int ordinal, out Device value);
    [DllImport(Dll, CallingConvention=CallingConvention.Cdecl, CharSet=CharSet.Unicode)]
    internal static extern int pu_open(int ordinal, string directory, out IntPtr context);
    [DllImport(Dll, CallingConvention=CallingConvention.Cdecl)] internal static extern int pu_benchmark(IntPtr ctx, int m, int n,
        int k, int mode, int samples, uint seed, int dataset, uint flags, out Bench result);
    [DllImport(Dll, CallingConvention=CallingConvention.Cdecl)] internal static extern void pu_close(IntPtr context);
    [DllImport(Dll, CallingConvention=CallingConvention.Cdecl)] private static extern IntPtr pu_last_error();
    internal static void Check(int status)
    {
        if (status != 0) throw new InvalidOperationException(Marshal.PtrToStringAnsi(pu_last_error()));
    }
    internal static Bench Measure(int ordinal, string directory, int m, int n, int k, int mode, uint flags)
    {
        IntPtr handle;
        Check(pu_open(ordinal, directory, out handle));
        try {
            Bench result;
            Check(pu_benchmark(handle,m,n,k,mode,mode==5?1:3,2109,0,flags,out result));
            return result;
        } finally { pu_close(handle); }
    }
}

public sealed class Profile
{
    public string Path { get; set; }
    public string Arguments { get; set; }
    public string Directory { get; set; }
    public int Mode { get; set; }
    public bool Python { get; set; }
    public bool CastFloat { get; set; }
    public override string ToString() { return System.IO.Path.GetFileName(Path) + "   ·   " + Panel.Modes[Mode]; }
}

public sealed class Settings
{
    public bool Enabled { get; set; }
    public bool Fallback { get; set; }
    public bool Guard { get; set; }
    public bool Cache { get; set; }
    public int Mode { get; set; }
    public string Cuda { get; set; }
    public List<Profile> Applications { get; set; }
    public Settings() { Enabled=true; Fallback=true; Guard=true; Cuda=""; Applications=new List<Profile>(); }
}

internal sealed class Panel : Form
{
    internal static readonly string[] Modes={"自动","极速","均衡","高精度","原生 FP32","原生 FP16"};
    private static readonly string[] ModeIds={"auto","fast","balanced","precise","native","fp16"};
    private static readonly Color Background=Color.FromArgb(17,22,30), Card=Color.FromArgb(27,34,47),
        Ink=Color.FromArgb(234,240,247), Secondary=Color.FromArgb(172,184,201), Accent=Color.FromArgb(110,228,188);
    private readonly string home=AppDomain.CurrentDomain.BaseDirectory;
    private readonly string state=Environment.GetEnvironmentVariable("PU_STATE_DIR") ??
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),"PascalUnleashed","State");
    private readonly JavaScriptSerializer json=new JavaScriptSerializer();
    private readonly ComboBox device=new ComboBox(), profileMode=new ComboBox(), defaultMode=new ComboBox();
    private readonly ComboBox benchMode=new ComboBox(), shape=new ComboBox();
    private readonly ListBox applications=new ListBox();
    private readonly Label deviceDetails=new Label(), gpu=new Label(), counters=new Label(), status=new Label();
    private readonly NumericUpDown dimM=Dim(),dimN=Dim(),dimK=Dim();
    private readonly TextBox cuda=new TextBox(), args=new TextBox(), working=new TextBox();
    private readonly CheckBox enabled=Toggle("加速总开关"), fallback=Toggle("允许回退"), guard=Toggle("保护高精度档的同列异常值"),
        cache=Toggle("缓存已编码权重"), python=Toggle("启用 Python / PyTorch 接入"), cast=Toggle("允许 FP32 推理线性层转换为 FP16");
    private readonly ListView results=new ListView();
    private readonly List<Backend.Device> devices=new List<Backend.Device>();
    private readonly Timer timer=new Timer();
    private Settings settings;
    private bool loading=true,refreshing;
    private Process child;
    private string telemetry;

    public Panel()
    {
        Text="Pascal Unleashed";
        Icon=Icon.ExtractAssociatedIcon(Application.ExecutablePath);
        BackColor=Background; ForeColor=Ink; Font=new Font("Microsoft YaHei UI",10F);
        MinimumSize=new Size(1010,710); Size=new Size(1240,810); StartPosition=FormStartPosition.CenterScreen;
        Directory.CreateDirectory(state);
        string profileFile=Path.Combine(state,"settings.json");
        try { settings=File.Exists(profileFile)?json.Deserialize<Settings>(File.ReadAllText(profileFile)):new Settings(); }
        catch (ArgumentException) { settings=new Settings(); }
        catch (InvalidOperationException) { settings=new Settings(); }
        if (settings==null) settings=new Settings();
        var body=new TableLayoutPanel {Dock=DockStyle.Fill, BackColor=Background, Padding=new Padding(23,16,23,16),
                                       ColumnCount=2, RowCount=3};
        body.ColumnStyles.Add(new ColumnStyle(SizeType.Percent,42)); body.ColumnStyles.Add(new ColumnStyle(SizeType.Percent,58));
        body.RowStyles.Add(new RowStyle(SizeType.Absolute,69)); body.RowStyles.Add(new RowStyle(SizeType.Percent,47));
        body.RowStyles.Add(new RowStyle(SizeType.Percent,53));
        Controls.Add(body);
        var top=Column();
        top.AutoScroll=false;
        top.Controls.Add(TextLabel("Pascal Unleashed",23F,Ink));
        top.Controls.Add(TextLabel("P40 计算加速  /  v1.0",9F,Accent));
        body.Controls.Add(top,0,0); body.SetColumnSpan(top,2);

        var devicesCard=Section(body,0,1,"设备与计算设置");
        device.Width=314; Style(device); devicesCard.Controls.Add(device);
        Describe(deviceDetails,"正在检测 CUDA 设备",365,60); devicesCard.Controls.Add(deviceDetails);
        var row=Row(); row.Controls.Add(enabled); row.Controls.Add(fallback); devicesCard.Controls.Add(row);
        row=Row(); row.Controls.Add(guard); row.Controls.Add(cache); devicesCard.Controls.Add(row);
        row=Row(); row.Controls.Add(TextLabel("默认计算档",9F,Secondary)); BindModes(defaultMode,5); row.Controls.Add(defaultMode); devicesCard.Controls.Add(row);
        row=Row(); cuda.Width=245; Style(cuda); row.Controls.Add(cuda);
        row.Controls.Add(Action("CUDA 12 路径", (s,e)=>{using(var d=new FolderBrowserDialog()) if(d.ShowDialog()==DialogResult.OK) cuda.Text=d.SelectedPath;}));
        devicesCard.Controls.Add(row);
        devicesCard.Controls.Add(Action("检查后端",async (s,e)=>await CheckBackend((Button)s)));

        var appCard=Section(body,1,1,"应用程序");
        applications.Width=485; applications.Height=70; Style(applications); appCard.Controls.Add(applications);
        row=Row(); row.Controls.Add(Action("添加",(s,e)=>Add())); row.Controls.Add(Action("移除",(s,e)=>Remove()));
        row.Controls.Add(Action("启动加速",(s,e)=>Launch(true),true)); row.Controls.Add(Action("按原环境启动",(s,e)=>Launch(false))); appCard.Controls.Add(row);
        row=Row(); BindModes(profileMode,5); row.Controls.Add(profileMode); row.Controls.Add(python); row.Controls.Add(cast); appCard.Controls.Add(row);
        row=Row(); row.Controls.Add(TextLabel("启动参数",9F,Secondary)); args.Width=395; Style(args); row.Controls.Add(args); appCard.Controls.Add(row);
        row=Row(); row.Controls.Add(TextLabel("工作目录",9F,Secondary)); working.Width=395; Style(working); row.Controls.Add(working); appCard.Controls.Add(row);
        Describe(status,"选择一个应用来启动",480,38); appCard.Controls.Add(status);

        var monitor=Section(body,0,2,"实时监控");
        Describe(gpu,"利用率、显存、温度、功耗 · 不可用",365,78); monitor.Controls.Add(gpu);
        Describe(counters,"计算模式、加速、回退、吞吐和覆盖率 · 不可用",365,90); monitor.Controls.Add(counters);

        var bench=Section(body,1,2,"矩阵 Benchmark");
        row=Row(); BindModes(benchMode,6); row.Controls.Add(benchMode);
        ShapeChoices(); row.Controls.Add(shape); row.Controls.Add(Action("开始测试",async (s,e)=>await Benchmark((Button)s),true)); bench.Controls.Add(row);
        row=Row(); row.Controls.Add(TextLabel("M",9F,Secondary)); row.Controls.Add(dimM);
        row.Controls.Add(TextLabel("N",9F,Secondary)); row.Controls.Add(dimN); row.Controls.Add(TextLabel("K",9F,Secondary)); row.Controls.Add(dimK); bench.Controls.Add(row);
        results.View=View.Details; results.HeaderStyle=ColumnHeaderStyle.Nonclickable; results.FullRowSelect=true;
        results.Width=508; results.Height=154; Style(results);
        foreach(var pair in new[]{"档位|90","T/s|62","耗时|74","相对误差|86","对 FP32|77","显存|94"}) {
            string[] col=pair.Split('|'); results.Columns.Add(col[0],Int32.Parse(col[1]));
        }
        bench.Controls.Add(results);
        bench.Controls.Add(TextLabel("误差使用全矩阵 Frobenius 范数，参考为 FP32。显存为测试分配量。",8F,Secondary));

        enabled.Checked=settings.Enabled; fallback.Checked=settings.Fallback; guard.Checked=settings.Guard; cache.Checked=settings.Cache;
        defaultMode.SelectedIndex=Math.Max(0,Math.Min(4,settings.Mode)); cuda.Text=settings.Cuda;
        foreach(var p in settings.Applications.Where(p=>p!=null && p.Mode>=0 && p.Mode<5)) applications.Items.Add(p);
        applications.SelectedIndexChanged+=(s,e)=>ShowProfile();
        foreach (var checkbox in new[]{enabled,fallback,guard,cache}) checkbox.CheckedChanged+=(s,e)=>Save();
        foreach (var checkbox in new[]{python,cast}) checkbox.CheckedChanged+=(s,e)=>SaveProfile();
        defaultMode.SelectedIndexChanged+=(s,e)=>Save(); cuda.TextChanged+=(s,e)=>{ Save(); DeviceDetails(); };
        profileMode.SelectedIndexChanged+=(s,e)=>SaveProfile(); args.TextChanged+=(s,e)=>SaveProfile(); working.TextChanged+=(s,e)=>SaveProfile();
        device.SelectedIndexChanged+=(s,e)=>DeviceDetails();
        Shown+=(s,e)=>Discover();
        timer.Interval=2200; timer.Tick+=async (s,e)=>await RefreshMonitor(); timer.Start();
        FormClosed+=(s,e)=>{timer.Stop(); if(child!=null) child.Dispose();};
        loading=false;
    }

    private static NumericUpDown Dim() { return new NumericUpDown {Minimum=1,Maximum=65536,Value=2048,Width=75,BackColor=Background,ForeColor=Ink}; }
    private static void Style(Control control) { control.BackColor=Background; control.ForeColor=Ink; }
    private static CheckBox Toggle(string text) { return new CheckBox {Text=text,AutoSize=true,ForeColor=Ink,Margin=new Padding(4,5,7,3)}; }
    private static FlowLayoutPanel Column() { return new FlowLayoutPanel {Dock=DockStyle.Fill,FlowDirection=FlowDirection.TopDown,WrapContents=false,AutoScroll=true}; }
    private static FlowLayoutPanel Row() { return new FlowLayoutPanel {AutoSize=true,WrapContents=false,Margin=new Padding(0),Padding=new Padding(0)}; }
    private static Label TextLabel(string text,float size,Color ink) { return new Label {Text=text,ForeColor=ink,AutoSize=true,Font=new Font("Microsoft YaHei UI",size),Margin=new Padding(2,3,9,3)}; }
    private static void Describe(Label label,string text,int width,int height) { label.Text=text; label.Size=new Size(width,height); label.ForeColor=Secondary; }
    private static Button Action(string text,EventHandler click,bool primary=false) {
        var button=new Button {Text=text,AutoSize=true,FlatStyle=FlatStyle.Flat,BackColor=primary?Accent:Background,ForeColor=primary?Background:Ink,
                               Padding=new Padding(5,0,5,0),Height=30};
        button.FlatAppearance.BorderColor=Color.FromArgb(57,72,89); button.Click+=click;
        return button;
    }
    private static FlowLayoutPanel Section(TableLayoutPanel root,int col,int row,string name) {
        var panel=Column(); panel.BackColor=Card; panel.Padding=new Padding(14,10,8,8); panel.Margin=new Padding(5,6,7,6);
        panel.Controls.Add(TextLabel(name,13F,Ink)); root.Controls.Add(panel,col,row); return panel;
    }
    private static void BindModes(ComboBox box,int count) {
        box.DropDownStyle=ComboBoxStyle.DropDownList; box.Width=122; Style(box);
        box.Items.AddRange(Modes.Take(count).Cast<object>().ToArray()); box.SelectedIndex=0;
    }
    private void ShapeChoices() {
        shape.Items.AddRange(new object[]{"2048³","4096³","8192³","16384³","16384×16384×8192","自定义"});
        shape.DropDownStyle=ComboBoxStyle.DropDownList; shape.Width=151; Style(shape); shape.SelectedIndex=0;
        shape.SelectedIndexChanged+=(s,e)=>{
            if (shape.SelectedIndex>=5) return;
            int index=Math.Max(0,shape.SelectedIndex);
            dimM.Value=dimN.Value=2048<<Math.Min(3,index);
            dimK.Value=index==4?8192:dimM.Value;
        };
    }
    private void Save() {
        if (loading) return;
        settings.Enabled=enabled.Checked; settings.Fallback=fallback.Checked; settings.Guard=guard.Checked; settings.Cache=cache.Checked;
        settings.Mode=defaultMode.SelectedIndex; settings.Cuda=cuda.Text;
        string path=Path.Combine(state,"settings.json"),temp=path+".tmp";
        try {
            File.WriteAllText(temp,json.Serialize(settings));
            if (File.Exists(path)) File.Replace(temp,path,null); else File.Move(temp,path);
        } catch (IOException e) { status.Text=e.Message; }
    }
    private void Discover() {
        try {
            int count=Backend.pu_devices();
            for(int i=0;i<count;i++) { Backend.Device value; if(Backend.pu_device(i,out value)!=0) continue;
                devices.Add(value); device.Items.Add(""+i+"   "+value.Name); }
            if (devices.Count>0) device.SelectedIndex=0;
            else deviceDetails.Text="未检测到 CUDA 设备";
        } catch(Exception e) { deviceDetails.Text=e.Message; }
    }
    private void DeviceDetails() {
        int index=device.SelectedIndex;
        if (index<0 || index>=devices.Count) return;
        Backend.Device d=devices[index];
        string architecture=d.Major==6?"Pascal":"CC "+d.Major+"."+d.Minor;
        deviceDetails.Text=String.Format("{0} · CC {1}.{2} · {3:F1} GiB\r\nCUDA 驱动 API {4}.{5} · {6}",architecture,d.Major,d.Minor,
            d.Memory/1073741824.0,d.Driver/1000,(d.Driver%1000)/10,d.Name.Contains("P40")?"P40 已验证":"此设备待验证");
        if(!String.IsNullOrWhiteSpace(cuda.Text)) deviceDetails.Text+=File.Exists(Path.Combine(cuda.Text,"cublas64_12.dll"))?
            "\r\n找到 CUDA 12 cuBLAS":"\r\n指定目录缺少 CUDA 12 cuBLAS";
    }
    private async Task CheckBackend(Button button) {
        int ordinal=device.SelectedIndex;
        if(ordinal<0) return;
        string directory=String.IsNullOrWhiteSpace(cuda.Text)?null:cuda.Text;
        button.Enabled=false;
        try {
            await Task.Run(()=>{
                IntPtr context; Backend.Check(Backend.pu_open(ordinal,directory,out context)); Backend.pu_close(context);
            });
            DeviceDetails(); deviceDetails.Text+="\r\nCUDA 后端已加载";
        } catch(Exception e) { MessageBox.Show(e.Message,"CUDA 后端不可用"); }
        finally { button.Enabled=true; }
    }
    private Profile Selected { get { return applications.SelectedItem as Profile; } }
    private void Add() {
        using(var dialog=new OpenFileDialog {Filter="应用程序|*.exe|所有文件|*.*"}) {
            if(dialog.ShowDialog()!=DialogResult.OK) return;
            var p=new Profile {Path=dialog.FileName,Directory=Path.GetDirectoryName(dialog.FileName),Arguments="",Mode=defaultMode.SelectedIndex,
                Python=Path.GetFileName(dialog.FileName).StartsWith("python",StringComparison.OrdinalIgnoreCase)};
            settings.Applications.Add(p); applications.Items.Add(p); applications.SelectedItem=p; Save();
        }
    }
    private void Remove() { var p=Selected; if(p==null) return; settings.Applications.Remove(p); applications.Items.Remove(p); Save(); }
    private void ShowProfile() {
        var p=Selected; if(p==null) return;
        loading=true;
        args.Text=p.Arguments; working.Text=p.Directory; profileMode.SelectedIndex=p.Mode; python.Checked=p.Python; cast.Checked=p.CastFloat;
        loading=false;
    }
    private void SaveProfile() {
        var p=Selected; if(loading || p==null) return;
        p.Arguments=args.Text; p.Directory=working.Text; p.Mode=profileMode.SelectedIndex; p.Python=python.Checked; p.CastFloat=cast.Checked; Save();
    }
    private void Launch(bool acceleration) {
        var p=Selected; if(p==null) { status.Text="请先添加并选中应用"; return; }
        try {
            var start=new ProcessStartInfo(p.Path,p.Arguments) {WorkingDirectory=p.Directory,UseShellExecute=false};
            string[] scoped={"PU_ENABLED","PU_MODE","PU_FALLBACK","PU_GUARD","PU_CACHE","PU_CAST_FP32","PU_TELEMETRY","PASCAL_UNLEASHED_DLL"};
            foreach(string key in scoped) start.EnvironmentVariables.Remove(key);
            telemetry=null;
            if(acceleration && enabled.Checked && p.Python && p.Mode!=4) {
                telemetry=Path.Combine(state,"app-"+Guid.NewGuid().ToString("N")+".json");
                start.EnvironmentVariables["PU_ENABLED"]="1";
                start.EnvironmentVariables["PU_MODE"]=ModeIds[p.Mode];
                start.EnvironmentVariables["PU_FALLBACK"]=fallback.Checked?"1":"0";
                start.EnvironmentVariables["PU_GUARD"]=guard.Checked?"1":"0";
                start.EnvironmentVariables["PU_CACHE"]=cache.Checked?"1":"0";
                start.EnvironmentVariables["PU_CAST_FP32"]=p.CastFloat?"1":"0";
                start.EnvironmentVariables["PU_TELEMETRY"]=telemetry;
                start.EnvironmentVariables["PASCAL_UNLEASHED_DLL"]=Path.Combine(home,Backend.Dll);
                start.EnvironmentVariables["PYTHONDONTWRITEBYTECODE"]="1";
                if(!String.IsNullOrWhiteSpace(cuda.Text)) start.EnvironmentVariables["PASCAL_CUDA_DLLS"]=cuda.Text;
                string existing=start.EnvironmentVariables["PYTHONPATH"];
                start.EnvironmentVariables["PYTHONPATH"]=Path.Combine(home,"python","bootstrap")+";"+Path.Combine(home,"python")+
                    (String.IsNullOrEmpty(existing)?"":";"+existing);
            }
            if(child!=null) child.Dispose(); child=Process.Start(start);
            status.Text=telemetry==null?"已按原环境启动":"已启动 · 等待 Python 接入报告";
        } catch(Exception e) { status.Text=e.Message; }
    }

    private async Task RefreshMonitor() {
        if(refreshing) return; refreshing=true;
        try {
            string expected=device.SelectedIndex>=0?devices[device.SelectedIndex].Name:"";
            string sample=await Task.Run(()=>ReadGpu(expected));
            if(IsDisposed) return;
            gpu.Text=sample;
            if(telemetry==null || !File.Exists(telemetry)) return;
            var report=json.Deserialize<Dictionary<string,object>>(File.ReadAllText(telemetry));
            if(!Convert.ToBoolean(report["loaded"])) { status.Text="后端未加载 · "+report["error"]; return; }
            long accel=Convert.ToInt64(report["accelerated"]),native=Convert.ToInt64(report["native"]),fallen=Convert.ToInt64(report["fallback"]);
            bool finished=child!=null && child.HasExited;
            status.Text=accel>0?(finished?"应用已退出 · 本次确实执行了 DP4A":"已验证 DP4A 执行"):
                (finished?"应用已退出 · 未观察到加速运算":"Python 接入已加载 · 等待算子");
            object throughput=report["tflops"];
            string speed=throughput==null?"不可用":Convert.ToDouble(throughput).ToString("F2")+" T/s";
            string coverage=accel+native==0?"不可用":(accel/(double)(accel+native)).ToString("P1");
            string active=Convert.ToString(report["actual"]);
            int index=Array.IndexOf(ModeIds,active);
            counters.Text=String.Format("当前模式  {0}     吞吐  {1}\r\n加速算子  {2}     回退算子  {3}\r\n算子加速覆盖率  {4}",
                index<0?"不可用":Modes[index],speed,accel,fallen,coverage);
        } catch(Exception e) { if(!IsDisposed) counters.Text="指标不可用 · "+e.Message; }
        finally { refreshing=false; }
    }
    private static string ReadGpu(string name) {
        try {
            var start=new ProcessStartInfo("nvidia-smi.exe","--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,driver_version --format=csv,noheader")
                {CreateNoWindow=true,UseShellExecute=false,RedirectStandardOutput=true};
            using(Process process=Process.Start(start)) {
                if(!process.WaitForExit(1500)) { process.Kill(); return "GPU 指标暂不可用"; }
                foreach (string line in process.StandardOutput.ReadToEnd().Split('\n')) {
                    string[] col=line.Split(',').Select(x=>x.Trim()).ToArray();
                    if(col.Length!=7 || col[0]!=name) continue;
                    for(int i=1;i<7;i++) if(col[i].StartsWith("[")) col[i]="不可用";
                    return String.Format("利用率  {0}     温度  {1}\r\n显存  {2} / {3}\r\n功耗  {4}     驱动  {5}",col[1],col[4],col[2],col[3],col[5],col[6]);
                }
            }
        } catch(System.ComponentModel.Win32Exception) { }
        return "GPU 利用率、显存、温度及功耗不可用";
    }
    private async Task Benchmark(Button button) {
        int ordinal=device.SelectedIndex;
        if(ordinal<0) { MessageBox.Show("未找到 GPU"); return; }
        int m=(int)dimM.Value,n=(int)dimN.Value,k=(int)dimK.Value,mode=benchMode.SelectedIndex;
        string directory=String.IsNullOrWhiteSpace(cuda.Text)?null:cuda.Text;
        uint flags=(fallback.Checked?1u:0u)|(guard.Checked?2u:0u);
        button.Enabled=false; button.Text="测试中…";
        try {
            var b=await Task.Run(()=>Backend.Measure(ordinal,directory,m,n,k,mode,flags));
            var entry=new ListViewItem(Modes[b.Run.Mode]+(b.Run.Reason==0?"":" 回退"));
            entry.SubItems.Add(b.Run.Tflops.ToString("F2")); entry.SubItems.Add(b.Run.Milliseconds.ToString("F1")+" ms");
            entry.SubItems.Add(b.Error.ToString("P4")); entry.SubItems.Add(b.Multiplier.ToString("F2")+"×");
            entry.SubItems.Add((b.Bytes/1073741824.0).ToString("F2")+" GiB"); results.Items.Insert(0,entry);
        } catch(Exception e) { MessageBox.Show(e.Message,"Benchmark 未完成"); }
        finally { button.Text="开始测试"; button.Enabled=true; }
    }

    [STAThread] public static int Main(string[] argv) {
        if(argv.Length>0 && argv[0]=="--probe") {
            var data=new List<Backend.Device>();
            for(int i=0;i<Backend.pu_devices();i++) { Backend.Device d; if(Backend.pu_device(i,out d)==0) data.Add(d); }
            File.WriteAllText(argv[1],new JavaScriptSerializer().Serialize(data)); return data.Count==0?1:0;
        }
        if(argv.Length>0 && argv[0]=="--benchmark") {
            try {
                var b=Backend.Measure(Int32.Parse(argv[2]),null,Int32.Parse(argv[3]),Int32.Parse(argv[4]),Int32.Parse(argv[5]),Int32.Parse(argv[6]),3);
                File.WriteAllText(argv[1],new JavaScriptSerializer().Serialize(b)); return 0;
            } catch(Exception e) { File.WriteAllText(argv[1],e.ToString()); return 1; }
        }
        Application.EnableVisualStyles(); Application.SetCompatibleTextRenderingDefault(false);
        if(argv.Length>1 && argv[0]=="--render") {
            using(var panel=new Panel()) {
                panel.ShowInTaskbar=false;
                panel.StartPosition=FormStartPosition.Manual;
                panel.Location=new Point(-3000,-3000);
                panel.Show(); Application.DoEvents();
                using(var image=new Bitmap(panel.ClientSize.Width,panel.ClientSize.Height)) {
                    panel.DrawToBitmap(image,panel.ClientRectangle); image.Save(argv[1],System.Drawing.Imaging.ImageFormat.Png);
                }
            }
            return 0;
        }
        if(argv.Length==4 && argv[0]=="--verify-app") {
            using(var panel=new Panel()) {
                var profile=new Profile {Path=argv[2],Arguments=argv[3],Directory=Path.GetDirectoryName(argv[2]),Mode=2,Python=true};
                panel.applications.Items.Add(profile);
                panel.applications.SelectedItem=profile;
                panel.enabled.Checked=true;
                panel.Launch(true);
                if(panel.child==null || !panel.child.WaitForExit(90000) || panel.child.ExitCode!=0) return 1;
                string text=File.ReadAllText(panel.telemetry);
                var telemetryData=panel.json.Deserialize<Dictionary<string,object>>(text);
                panel.Launch(false);
                bool restored=panel.telemetry==null && panel.child!=null && panel.child.WaitForExit(90000) && panel.child.ExitCode==0;
                bool accelerated=Convert.ToInt64(telemetryData["accelerated"])>0;
                File.WriteAllText(argv[1],panel.json.Serialize(new {accelerated=accelerated,restored=restored,operators=telemetryData["accelerated"]}));
                return accelerated && restored?0:1;
            }
        }
        Application.Run(new Panel()); return 0;
    }
}
