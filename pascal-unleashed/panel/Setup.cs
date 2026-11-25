using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Security.Cryptography;
using System.Windows.Forms;
using Microsoft.Win32;

internal static class Setup
{
    private const string Product="Pascal Unleashed";
    private const string Marker=".pascal-unleashed-installed";
    private const string Key=@"Software\Microsoft\Windows\CurrentVersion\Uninstall\PascalUnleashed";
    private const string Manifest="manifest.sha256";
    private static readonly string Local=Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
    private static string Arg(string[] argv,string name) {
        int index=Array.IndexOf(argv,name);
        return index>=0 && index+1<argv.Length?argv[index+1]:null;
    }
    private static string Quote(string path) { return "\""+path+"\""; }
    private static string Canonical(string directory) {
        string result=Path.GetFullPath(directory).TrimEnd(Path.DirectorySeparatorChar,Path.AltDirectorySeparatorChar);
        if(result.Length<=3 || result.IndexOf('"')>=0) throw new IOException("请选择一个专用安装目录");
        return result;
    }
    private static string Inside(string root,string item) {
        string path=Path.GetFullPath(Path.Combine(root,item));
        if(!path.StartsWith(root+Path.DirectorySeparatorChar,StringComparison.OrdinalIgnoreCase)) throw new IOException("安装包路径无效");
        return path;
    }
    private static string Digest(string path) {
        using(var sha=SHA256.Create()) using(var file=File.OpenRead(path))
            return BitConverter.ToString(sha.ComputeHash(file)).Replace("-","").ToLowerInvariant();
    }
    private static void Extract(string staging) {
        using(Stream payload=Assembly.GetExecutingAssembly().GetManifestResourceStream("release.zip"))
        using(var zip=new ZipArchive(payload,ZipArchiveMode.Read)) {
            foreach(var item in zip.Entries) {
                string path=Inside(staging,item.FullName);
                if (String.IsNullOrEmpty(item.Name)) { Directory.CreateDirectory(path); continue; }
                Directory.CreateDirectory(Path.GetDirectoryName(path));
                item.ExtractToFile(path);
            }
        }
        foreach(string line in File.ReadLines(Path.Combine(staging,Manifest))) {
            if(line.Length<67 || !String.Equals(line.Substring(0,64),Digest(Inside(staging,line.Substring(66))),StringComparison.Ordinal))
                throw new IOException("安装文件的 SHA-256 校验失败");
        }
    }
    private static void Register(string root) {
        using(var key=Registry.CurrentUser.CreateSubKey(Key)) {
            key.SetValue("DisplayName",Product); key.SetValue("DisplayVersion","1.0.0");
            key.SetValue("InstallLocation",root); key.SetValue("DisplayIcon",Path.Combine(root,"PascalUnleashedPanel.exe"));
            key.SetValue("UninstallString",Quote(Path.Combine(root,"Uninstall.exe"))+" --uninstall --dir "+Quote(root));
            key.SetValue("NoModify",1,RegistryValueKind.DWord); key.SetValue("NoRepair",1,RegistryValueKind.DWord);
        }
        string programs=Environment.GetFolderPath(Environment.SpecialFolder.Programs);
        string shortcut=Path.Combine(programs,Product+".lnk");
        Type type=Type.GetTypeFromProgID("WScript.Shell");
        if(type!=null) {
            dynamic shell=Activator.CreateInstance(type),link=shell.CreateShortcut(shortcut);
            link.TargetPath=Path.Combine(root,"PascalUnleashedPanel.exe"); link.WorkingDirectory=root; link.Save();
        }
    }
    private static void Install(string root,bool registration) {
        string staging=root+".stage-"+Guid.NewGuid().ToString("N");
        if(Directory.Exists(root)) throw new IOException("安装目录已存在。请选择新目录，或先卸载旧版本。");
        if(registration) using(var key=Registry.CurrentUser.OpenSubKey(Key)) {
            if(key!=null) throw new IOException("已注册 Pascal Unleashed。请先卸载旧版本，或使用便携包。");
        }
        Directory.CreateDirectory(staging);
        try {
            Extract(staging);
            File.Copy(Application.ExecutablePath,Path.Combine(staging,"Uninstall.exe"));
            File.WriteAllText(Path.Combine(staging,Marker),registration?"registered":"standalone");
            Directory.Move(staging,root);
            if(registration) Register(root);
        } finally {
            if(Directory.Exists(staging)) Directory.Delete(staging,true);
        }
    }
    private static void Remove(string root) {
        string marker=Path.Combine(root,Marker);
        if(!File.Exists(marker)) throw new IOException("找不到专用安装标记");
        if((File.GetAttributes(root)&FileAttributes.ReparsePoint)!=0) throw new IOException("安装目录已被重定向");
        bool registration=File.ReadAllText(marker)=="registered";
        var directories=new System.Collections.Generic.HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach(string line in File.ReadLines(Path.Combine(root,Manifest))) {
            string path=Inside(root,line.Substring(66));
            bool redirected=false;
            for(string d=Path.GetDirectoryName(path); !d.Equals(root,StringComparison.OrdinalIgnoreCase); d=Path.GetDirectoryName(d))
                if(Directory.Exists(d) && (File.GetAttributes(d)&FileAttributes.ReparsePoint)!=0) { redirected=true; break; }
            if(redirected) continue;
            File.Delete(path);
            for(string d=Path.GetDirectoryName(path); !d.Equals(root,StringComparison.OrdinalIgnoreCase); d=Path.GetDirectoryName(d)) directories.Add(d);
        }
        var sorted=new System.Collections.Generic.List<string>(directories);
        sorted.Sort((a,b)=>b.Length.CompareTo(a.Length));
        foreach(var dir in sorted) if(Directory.Exists(dir) && Directory.GetFileSystemEntries(dir).Length==0) Directory.Delete(dir);
        File.Delete(Path.Combine(root,Manifest)); File.Delete(Path.Combine(root,"Uninstall.exe")); File.Delete(marker);
        if(registration) {
            using(var key=Registry.CurrentUser.OpenSubKey(Key)) {
                if(key!=null && String.Equals(root,key.GetValue("InstallLocation") as string,StringComparison.OrdinalIgnoreCase)) {
                    key.Close(); Registry.CurrentUser.DeleteSubKeyTree(Key,false);
                    File.Delete(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Programs),Product+".lnk"));
                }
            }
        }
        if(Directory.GetFileSystemEntries(root).Length==0) Directory.Delete(root);
    }
    [STAThread] public static int Main(string[] argv) {
        bool quiet=Array.IndexOf(argv,"--quiet")>=0;
        try {
            string root=Canonical(Arg(argv,"--dir") ?? Path.Combine(Local,"Programs","PascalUnleashed"));
            if(Array.IndexOf(argv,"--remove")>=0) {
                string parent=Arg(argv,"--parent");
                if(parent!=null) try { using(var p=Process.GetProcessById(Int32.Parse(parent))) p.WaitForExit(30000); } catch(ArgumentException) { }
                Remove(root); if(!quiet) MessageBox.Show("已卸载。用户的应用配置保存在本地。",Product); return 0;
            }
            if(Array.IndexOf(argv,"--uninstall")>=0) {
                if(!quiet && MessageBox.Show("卸载 Pascal Unleashed？",Product,MessageBoxButtons.OKCancel)!=DialogResult.OK) return 0;
                string helper=Path.Combine(Path.GetTempPath(),"PascalUnleashed-uninstall-"+Guid.NewGuid().ToString("N")+".exe");
                File.Copy(Application.ExecutablePath,helper);
                bool outside=!String.Equals(Path.GetDirectoryName(Application.ExecutablePath),root,StringComparison.OrdinalIgnoreCase);
                using(var p=Process.Start(new ProcessStartInfo(helper,"--remove --dir "+Quote(root)+(outside?"":" --parent "+Process.GetCurrentProcess().Id)+
                    (quiet?" --quiet":"")) {UseShellExecute=false,CreateNoWindow=true})) {
                    if(outside) { p.WaitForExit(); int status=p.ExitCode; File.Delete(helper); return status; }
                }
                return 0;
            }
            if(!quiet && MessageBox.Show("安装 Pascal Unleashed 到下面的位置？\r\n\r\n"+root,Product,MessageBoxButtons.OKCancel)!=DialogResult.OK) return 0;
            Install(root,Array.IndexOf(argv,"--no-register")<0);
            if(!quiet && MessageBox.Show("安装完成。打开控制面板？",Product,MessageBoxButtons.YesNo)==DialogResult.Yes)
                Process.Start(Path.Combine(root,"PascalUnleashedPanel.exe"));
            return 0;
        } catch(Exception error) {
            if(!quiet) MessageBox.Show(error.Message,Product,MessageBoxButtons.OK,MessageBoxIcon.Error);
            string log=Arg(argv,"--log"); if(log!=null) File.WriteAllText(log,error.ToString());
            return 1;
        }
    }
}
