// ─────────────────────────────────────────────────────────────────────────────
// ProcessScanCommand.cs — "Process Scan" ribbon button (Sprint P2)
//
// Flow:
//   1. OpenFileDialog  → scan file path
//   2. ZoneInputDialog → zone ID (default: zone-001)
//   3. Locate stb-processor.exe next to the addin DLL
//   4. ScanProgressWindow (WPF) — runs sidecar, streams PROGRESS tokens
//   5. Deserialise results.json → SidecarResults
//   6. Store in ScanResultStore.Latest for Sprint P3 geometry builder
//   7. TaskDialog summary (segment count, warnings, timing)
//
// The heavy processing runs OUTSIDE Revit's main thread via Process.Start().
// No Revit API calls are made during processing — safe by design.
// ─────────────────────────────────────────────────────────────────────────────

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shell;
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using Microsoft.Win32;
using ScanToBIM.Models;
using TextBox = System.Windows.Controls.TextBox;
using Color = System.Windows.Media.Color;

namespace ScanToBIM.Commands;

// ─────────────────────────────────────────────────────────────────────────────
// IExternalCommand
// ─────────────────────────────────────────────────────────────────────────────

[Transaction(TransactionMode.Manual)]
[Regeneration(RegenerationOption.Manual)]
public class ProcessScanCommand : IExternalCommand
{
    // Supported scan formats (must match processor_cli.py)
    private const string FileFilter =
        "Point Cloud Files|*.e57;*.las;*.laz;*.ply;*.pcd;*.xyz" +
        "|E57 (*.e57)|*.e57" +
        "|LAS/LAZ (*.las;*.laz)|*.las;*.laz" +
        "|PLY (*.ply)|*.ply" +
        "|All files (*.*)|*.*";

    public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
    {
        try
        {
            // ── 1. Pick scan file ─────────────────────────────────────────────
            var dlg = new OpenFileDialog
            {
                Title            = "Select Point Cloud Scan File",
                Filter           = FileFilter,
                CheckFileExists  = true,
                Multiselect      = false,
            };

            if (dlg.ShowDialog() != true)
                return Result.Cancelled;

            var scanPath = dlg.FileName;

            // ── 2. Zone input ─────────────────────────────────────────────────
            var zone = PromptZone();
            if (zone is null)
                return Result.Cancelled;

            // ── 3. Find sidecar executable ────────────────────────────────────
            var exePath = FindSidecarExe();
            if (exePath is null)
            {
                TaskDialog.Show(
                    "ScanToBIM — Sidecar Not Found",
                    "stb-processor.exe was not found.\n\n" +
                    "Expected location: same directory as the addin DLL.\n\n" +
                    "Build it with:\n  bash build/build-sidecar.sh\n\n" +
                    "Or set the environment variable STB_PROCESSOR_PATH to the full path.");
                return Result.Failed;
            }

            // ── 4. Show progress window, run sidecar ──────────────────────────

            // Ensure results directory exists
            var resultsDir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "ScanToBIM",
                "Results");
            if (!Directory.Exists(resultsDir))
                Directory.CreateDirectory(resultsDir);
            var resultsPath = Path.Combine(resultsDir, $"stb-results-{Guid.NewGuid():N}.json");

            // ── 2.5. Voxel size input ─────────────────────────────────────────────
            double voxelSize = PromptVoxelSize();
            if (voxelSize <= 0)
                return Result.Cancelled;

            var win = new ScanProgressWindow(scanPath, zone, exePath, resultsPath, voxelSize);
            var ok  = win.ShowDialog();          // blocks until done or cancelled

            if (ok != true)
                return Result.Cancelled;

            // ── 5. Deserialise results ────────────────────────────────────────
            if (!File.Exists(resultsPath))
            {
                TaskDialog.Show("ScanToBIM", "Sidecar completed but results file was not created.");
                return Result.Failed;
            }

            SidecarResults results;
            try
            {
                var json = File.ReadAllText(resultsPath);
                results  = JsonSerializer.Deserialize<SidecarResults>(json)
                           ?? new SidecarResults();
            }
            catch (Exception ex)
            {
                TaskDialog.Show("ScanToBIM — Parse Error", $"Could not read results.json:\n{ex.Message}");
                return Result.Failed;
            }

            // ── 6. Store for Sprint P3 ────────────────────────────────────────
            ScanResultStore.Latest      = results;
            ScanResultStore.LatestResultsPath = resultsPath;

            // ── 7. Summary dialog ─────────────────────────────────────────────
            ShowSummary(results, scanPath, exePath);

            return Result.Succeeded;
        }
        catch (Exception ex)
        {
            message = ex.Message;
            return Result.Failed;
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static string? PromptZone()
    {
        var inputWin = new ZoneInputWindow();
        return inputWin.ShowDialog() == true ? inputWin.ZoneId : null;
    }

    private static string? FindSidecarExe()
    {
        // Priority 1: environment variable override
        var envPath = Environment.GetEnvironmentVariable("STB_PROCESSOR_PATH");
        if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            return envPath;

        // Resolve assembly and repository roots once.
        var asmDir  = Path.GetDirectoryName(typeof(ProcessScanCommand).Assembly.Location) ?? "";
        var repoRoot  = FindRepoRoot(asmDir);

        // Priority 2: development outputs in repo (prefer freshest local build)
        if (repoRoot is not null)
        {
            foreach (var candidate in new[]
            {
                Path.Combine(repoRoot, "build", "dist", "stb-processor.exe"),
                Path.Combine(repoRoot, "dist", "stb-processor.exe"),
                Path.Combine(repoRoot, "build", "dist", "stb-processor"),
                Path.Combine(repoRoot, "dist", "stb-processor"),
            })
            {
                if (File.Exists(candidate))
                    return candidate;
            }
        }

        // Priority 3: same directory as the addin DLL
        var sideBySide = Path.Combine(asmDir, "stb-processor.exe");
        if (File.Exists(sideBySide))
            return sideBySide;

        // Priority 4: local AppData folder (fallback).
        var appDataFolder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ScanToBIM");
        var appDataExe = Path.Combine(appDataFolder, "stb-processor.exe");
        if (File.Exists(appDataExe))
            return appDataExe;

        // Priority 5: Linux — no .exe extension (dev-machine build)
        var noExt = Path.Combine(asmDir, "stb-processor");
        if (File.Exists(noExt))
            return noExt;

        return null;
    }

    private static string? FindRepoRoot(string startDir)
    {
        var dir = new DirectoryInfo(startDir);
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "docker-compose.yml")))
                return dir.FullName;
            dir = dir.Parent;
        }
        return null;
    }

    private static void ShowSummary(SidecarResults results, string scanPath, string exePath)
    {
        var meta  = results.Metadata;
        var lines = new List<string>
        {
            $"File          : {Path.GetFileName(scanPath)}",
            $"Sidecar       : {exePath}",
            $"Segments found: {results.Segments.Count:N0}",
        };

        if (meta.PointCountRaw.HasValue)
            lines.Add($"Points (raw)  : {meta.PointCountRaw.Value:N0}");

        lines.Add($"Processing    : {meta.ProcessingTimeS:F1} s");

        if (results.Warnings.Count > 0)
        {
            lines.Add("");
            lines.Add($"Warnings ({results.Warnings.Count}):");
            foreach (var w in results.Warnings)
                lines.Add($"  • {w}");
        }

        lines.Add("");
        lines.Add("Results stored — use 'Build Geometry' (Sprint P3) to push to Revit.");

        var td = new TaskDialog("ScanToBIM — Scan Complete")
        {
            MainContent      = string.Join(Environment.NewLine, lines),
            MainIcon         = TaskDialogIcon.TaskDialogIconInformation,
        };
        td.Show();
    }
// End of ProcessScanCommand class

// Voxel size dialog helpers (must be in namespace, not inside class)
internal static double PromptVoxelSize()
{
    var inputWin = new VoxelSizeInputWindow();
    return inputWin.ShowDialog() == true ? inputWin.VoxelSize : -1;
}

internal sealed class VoxelSizeInputWindow : Window
{
    public double VoxelSize { get; private set; } = 0.05;
    private readonly TextBox _tb;

    public VoxelSizeInputWindow()
    {
        Title = "ScanToBIM — Voxel Size";
        Width = 340;
        Height = 160;
        ResizeMode = ResizeMode.NoResize;
        WindowStartupLocation = WindowStartupLocation.CenterScreen;
        Background = Brushes.White;

        var sp = new StackPanel { Margin = new Thickness(16) };

        sp.Children.Add(new TextBlock
        {
            Text = "Voxel size for segmentation (meters):",
            FontSize = 13,
            Margin = new Thickness(0, 0, 0, 8),
        });

        _tb = new TextBox
        {
            Text = "0.05",
            FontSize = 13,
            Padding = new Thickness(4),
            Margin = new Thickness(0, 0, 0, 14),
        };
        sp.Children.Add(_tb);

        var btns = new StackPanel
        {
            Orientation = Orientation.Horizontal,
            HorizontalAlignment = HorizontalAlignment.Right,
        };

        var ok = new Button
        {
            Content = "OK",
            IsDefault = true,
            Width = 72,
            Height = 28,
            Margin = new Thickness(0, 0, 8, 0),
            Background = new SolidColorBrush(System.Windows.Media.Color.FromRgb(37, 99, 235)),
            Foreground = Brushes.White,
            BorderThickness = new Thickness(0),
        };
        ok.Click += (_, _) =>
        {
            if (double.TryParse(_tb.Text.Trim(), out var val) && val > 0)
            {
                VoxelSize = val;
                DialogResult = true;
            }
            else
            {
                MessageBox.Show("Please enter a valid positive number for voxel size.", "Invalid Input", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        };

        var cancel = new Button
        {
            Content = "Cancel",
            IsCancel = true,
            Width = 72,
            Height = 28,
        };

        btns.Children.Add(ok);
        btns.Children.Add(cancel);
        sp.Children.Add(btns);

        Content = sp;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Zone input dialog  (pure C# WPF — no XAML required)
// ─────────────────────────────────────────────────────────────────────────────

internal sealed class ZoneInputWindow : Window
{
    public string ZoneId { get; private set; } = "zone-001";

    private readonly TextBox _tb;

    public ZoneInputWindow()
    {
        Title           = "ScanToBIM — Zone";
        Width           = 340;
        Height          = 160;
        ResizeMode      = ResizeMode.NoResize;
        WindowStartupLocation = WindowStartupLocation.CenterScreen;
        Background      = Brushes.White;

        var sp = new StackPanel { Margin = new Thickness(16) };

        sp.Children.Add(new TextBlock
        {
            Text       = "Zone identifier for this scan:",
            FontSize   = 13,
            Margin     = new Thickness(0, 0, 0, 8),
        });

        _tb = new System.Windows.Controls.TextBox
        {
            Text      = "zone-001",
            FontSize  = 13,
            Padding   = new Thickness(4),
            Margin    = new Thickness(0, 0, 0, 14),
        };
        sp.Children.Add(_tb);

        var btns = new StackPanel
        {
            Orientation = Orientation.Horizontal,
            HorizontalAlignment = HorizontalAlignment.Right,
        };

        var ok = new Button
        {
            Content         = "OK",
            IsDefault       = true,
            Width           = 72,
            Height          = 28,
            Margin          = new Thickness(0, 0, 8, 0),
            Background      = new SolidColorBrush(System.Windows.Media.Color.FromRgb(37, 99, 235)),
            Foreground      = Brushes.White,
            BorderThickness = new Thickness(0),
        };
        ok.Click += (_, _) => { ZoneId = _tb.Text.Trim(); DialogResult = true; };

        var cancel = new Button
        {
            Content  = "Cancel",
            IsCancel = true,
            Width    = 72,
            Height   = 28,
        };

        btns.Children.Add(ok);
        btns.Children.Add(cancel);
        sp.Children.Add(btns);

        Content = sp;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Progress window — launches sidecar, streams PROGRESS tokens, reads results
// ─────────────────────────────────────────────────────────────────────────────


internal sealed class ScanProgressWindow : Window
{
    // ── State ─────────────────────────────────────────────────────────────────
    private readonly string _scanPath;
    private readonly string _zone;
    private readonly string _exePath;
    private readonly string _resultsPath;
    private readonly double _voxelSize;

    private readonly BackgroundWorker _worker = new() { WorkerReportsProgress = true, WorkerSupportsCancellation = true };

    // ── UI elements ───────────────────────────────────────────────────────────
    private readonly ProgressBar _bar;
    private readonly TextBlock   _statusLabel;
    private readonly TextBox     _log;
    private readonly Button      _cancelBtn;

    public ScanProgressWindow(string scanPath, string zone, string exePath, string resultsPath, double voxelSize)
    {
        _scanPath    = scanPath;
        _zone        = zone;
        _exePath     = exePath;
        _resultsPath = resultsPath;
        _voxelSize = voxelSize;

        Title                 = "ScanToBIM — Processing Scan";
        Width                 = 560;
        Height                = 380;
        ResizeMode            = ResizeMode.NoResize;
        WindowStartupLocation = WindowStartupLocation.CenterScreen;
        Background            = Brushes.White;

        // Taskbar progress (Windows 7+)
        TaskbarItemInfo = new TaskbarItemInfo
        {
            ProgressState = TaskbarItemProgressState.Normal,
            ProgressValue = 0,
        };

        // ── Layout ────────────────────────────────────────────────────────────
        var root = new DockPanel { Margin = new Thickness(16) };

        // File label
        var fileLabel = new TextBlock
        {
            Text       = $"File: {System.IO.Path.GetFileName(scanPath)}",
            FontWeight = FontWeights.Bold,
            FontSize   = 13,
            Margin     = new Thickness(0, 0, 0, 10),
        };
        DockPanel.SetDock(fileLabel, Dock.Top);
        root.Children.Add(fileLabel);

        // Voxel size label
        var voxelLabel = new TextBlock
        {
            Text = $"Voxel size (m): {_voxelSize}",
            FontSize = 12,
            Margin = new Thickness(0, 0, 0, 6),
        };
        DockPanel.SetDock(voxelLabel, Dock.Top);
        root.Children.Add(voxelLabel);

        // Status label
        _statusLabel = new TextBlock
        {
            Text     = "Initialising…",
            FontSize = 12,
            Margin   = new Thickness(0, 0, 0, 6),
        };
        DockPanel.SetDock(_statusLabel, Dock.Top);
        root.Children.Add(_statusLabel);

        // Progress bar
        _bar = new ProgressBar
        {
            Minimum = 0,
            Maximum = 100,
            Height  = 18,
            Margin  = new Thickness(0, 0, 0, 10),
        };
        DockPanel.SetDock(_bar, Dock.Top);
        root.Children.Add(_bar);

        // Cancel button
        _cancelBtn = new Button
        {
            Content             = "Cancel",
            Width               = 80,
            Height              = 28,
            HorizontalAlignment = HorizontalAlignment.Right,
        };
        _cancelBtn.Click += OnCancel;
        DockPanel.SetDock(_cancelBtn, Dock.Bottom);
        root.Children.Add(_cancelBtn);

        // Log box
        _log = new TextBox
        {
            IsReadOnly          = true,
            VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
            FontFamily          = new FontFamily("Consolas"),
            FontSize            = 11,
            Background          = new SolidColorBrush(Color.FromRgb(245, 245, 245)),
            Margin              = new Thickness(0, 0, 0, 8),
        };
        root.Children.Add(_log);

        Content = root;

        // ── Wire background worker ────────────────────────────────────────────
        _worker.DoWork             += DoWork;
        _worker.ProgressChanged    += OnProgressChanged;
        _worker.RunWorkerCompleted += OnCompleted;

        Loaded += (_, _) => _worker.RunWorkerAsync();
    }

    // ── BackgroundWorker ──────────────────────────────────────────────────────

    private void DoWork(object? sender, DoWorkEventArgs e)
    {
        // var psi = new ProcessStartInfo
        // {
        //     FileName               = _exePath,
        //     Arguments              = $"--input \"{_scanPath}\" --zone \"{_zone}\" --output \"{_resultsPath}\" --voxel-size {_voxelSize}",
        //     RedirectStandardOutput = true,
        //     RedirectStandardError  = true,
        //     UseShellExecute        = false,
        //     CreateNoWindow         = true,
        // };
        var exeDir = Path.GetDirectoryName(_exePath)!;
        var repoRoot = FindRepoRoot(exeDir) ?? exeDir;

        var psi = new ProcessStartInfo
        {
            FileName               = _exePath,
            WorkingDirectory       = repoRoot,
            Arguments              = $"--input \"{_scanPath}\" --zone \"{_zone}\" --output \"{_resultsPath}\" --voxel-size {_voxelSize}",
            RedirectStandardOutput = true,
            RedirectStandardError  = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding  = Encoding.UTF8,
            UseShellExecute        = false,
            CreateNoWindow         = true,
        };
        psi.EnvironmentVariables["PYTHONUTF8"] = "1";
        psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
        psi.EnvironmentVariables["STB_ENABLE_CYLINDER_CHAIN_MERGE"] = "1";

        using var proc = new Process { StartInfo = psi, EnableRaisingEvents = true };

        var errors = new System.Text.StringBuilder();
        proc.ErrorDataReceived += (_, args) =>
        {
            if (args.Data is not null)
                errors.AppendLine(args.Data);
        };

        proc.Start();
        proc.BeginErrorReadLine();

        // Read stdout line by line — parse PROGRESS and INFO tokens
        while (!proc.StandardOutput.EndOfStream)
        {
            if (_worker.CancellationPending)
            {
                proc.Kill();
                e.Cancel = true;
                return;
            }

            var line = proc.StandardOutput.ReadLine();
            if (line is null) continue;

            if (line.StartsWith("PROGRESS:", StringComparison.Ordinal))
            {
                // Format: PROGRESS:N:Description
                var parts = line.Split(':', 3);
                if (parts.Length == 3 && int.TryParse(parts[1], out var pct))
                    _worker.ReportProgress(pct, parts[2]);
            }
            else if (line.StartsWith("{", StringComparison.Ordinal))
            {
                // JSON output (stdout mode — no --output flag) — ignore here
            }
            else
            {
                // INFO or unrecognised — show in log
                _worker.ReportProgress(-1, line);
            }
        }

        proc.WaitForExit();

        if (proc.ExitCode != 0)
        {
            e.Result = new WorkerError($"stb-processor exited with code {proc.ExitCode}.\n{errors}");
        }
    }

    private void OnProgressChanged(object? sender, ProgressChangedEventArgs e)
    {
        if (e.ProgressPercentage >= 0)
        {
            _bar.Value = e.ProgressPercentage;
            TaskbarItemInfo.ProgressValue = e.ProgressPercentage / 100.0;
            if (e.UserState is string label)
                _statusLabel.Text = label;
        }

        if (e.UserState is string logLine)
        {
            _log.AppendText(logLine + "\n");
            _log.ScrollToEnd();
        }
    }

    private void OnCompleted(object? sender, RunWorkerCompletedEventArgs e)
    {
        _cancelBtn.Content   = "Close";
        _cancelBtn.Click    -= OnCancel;
        _cancelBtn.Click    += (_, _) => Close();

        if (e.Cancelled)
        {
            _statusLabel.Text = "Cancelled.";
            TaskbarItemInfo.ProgressState = TaskbarItemProgressState.Error;
            //DialogResult = false;
            return;
        }

        // if (e.Result is WorkerError err)
        // {
        //     _statusLabel.Text = $"Error: {err.Message}";
        //     TaskbarItemInfo.ProgressState = TaskbarItemProgressState.Error;
        //     DialogResult = false;
        //     return;
        // }
        if (e.Result is WorkerError err)
        {
            _statusLabel.Text = $"Error occurred.";
            TaskbarItemInfo.ProgressState = TaskbarItemProgressState.Error;

            _log.AppendText("\n=== ERROR ===\n");
            _log.AppendText(err.Message + "\n");
            _log.ScrollToEnd();

            _cancelBtn.Content = "Close";

            return; // IMPORTANT: do NOT set DialogResult here
        }

        _statusLabel.Text             = "Complete!";
        _bar.Value                    = 100;
        TaskbarItemInfo.ProgressState = TaskbarItemProgressState.None;
        DialogResult                  = true;
    }

    private void OnCancel(object? sender, RoutedEventArgs e)
    {
        _cancelBtn.IsEnabled = false;
        _cancelBtn.Content   = "Cancelling…";
        _worker.CancelAsync();
    }

    // ── Error carrier ─────────────────────────────────────────────────────────

    internal sealed record WorkerError(string Message);
}
}
