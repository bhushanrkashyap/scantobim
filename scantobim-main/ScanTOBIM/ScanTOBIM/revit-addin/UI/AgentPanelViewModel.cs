// ─────────────────────────────────────────────────────────────────────────────
// AgentPanelViewModel.cs
//
// MVVM ViewModel for the ScanToBIM Agent dockable panel.
// Subscribes to AgentBridgeService events and exposes data-bindable properties
// for the WPF view in AgentPanelPage.xaml.
// ─────────────────────────────────────────────────────────────────────────────

using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Input;
using ScanToBIM.Models;
using ScanToBIM.Services;

namespace ScanToBIM.UI;

public sealed class AgentPanelViewModel : INotifyPropertyChanged
{
    private readonly AgentBridgeService _bridge;

    // ── Status ────────────────────────────────────────────────────────────────

    private bool _isRunning;
    public bool IsRunning
    {
        get => _isRunning;
        set
        {
            _isRunning = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(StatusText));
            OnPropertyChanged(nameof(StatusColor));
            OnPropertyChanged(nameof(ToggleLabel));
        }
    }

    public string StatusText  => IsRunning ? "● CONNECTED  port:8766" : "○  OFFLINE";
    public string StatusColor => IsRunning ? "#00c896" : "#4a5e78";
    public string ToggleLabel => IsRunning ? "Stop Bridge" : "Start Bridge";

    // ── Stats ─────────────────────────────────────────────────────────────────

    private int _created;
    public int ElementsCreated
    {
        get => _created;
        set { _created = value; OnPropertyChanged(); }
    }

    private int _blocked;
    public int ElementsBlocked
    {
        get => _blocked;
        set { _blocked = value; OnPropertyChanged(); }
    }

    private int _queued;
    public int QueueDepth
    {
        get => _queued;
        set { _queued = value; OnPropertyChanged(); }
    }

    // ── Collections ───────────────────────────────────────────────────────────

    public ObservableCollection<string>    Logs    { get; } = [];
    public ObservableCollection<ResultRow> Results { get; } = [];

    // ── Commands ──────────────────────────────────────────────────────────────

    public ICommand ToggleBridgeCommand { get; }
    public ICommand ClearCommand        { get; }

    // ── Constructor ───────────────────────────────────────────────────────────

    public AgentPanelViewModel(AgentBridgeService bridge)
    {
        _bridge = bridge;

        // Wire bridge events
        _bridge.OnLog += msg =>
        {
            Application.Current?.Dispatcher.Invoke(() =>
            {
                Logs.Insert(0, msg);
                if (Logs.Count > 300) Logs.RemoveAt(Logs.Count - 1);
                RefreshStats();
            });
        };

        _bridge.OnResult += result =>
        {
            Application.Current?.Dispatcher.Invoke(() =>
            {
                Results.Insert(0, ResultRow.FromActionResult(result));
                if (Results.Count > 150) Results.RemoveAt(Results.Count - 1);
                RefreshStats();
            });
        };

        ToggleBridgeCommand = new RelayCommand(() =>
        {
            if (IsRunning) { _bridge.Stop();  IsRunning = false; }
            else           { _bridge.Start(); IsRunning = true;  }
        });

        ClearCommand = new RelayCommand(() =>
        {
            Logs.Clear();
            Results.Clear();
        });
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void RefreshStats()
    {
        ElementsCreated = _bridge.ElementsCreated;
        ElementsBlocked = _bridge.ElementsBlocked;
        QueueDepth      = _bridge.QueueDepth;
    }

    // ── INotifyPropertyChanged ────────────────────────────────────────────────

    public event PropertyChangedEventHandler? PropertyChanged;

    private void OnPropertyChanged([CallerMemberName] string? name = null) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}

// ── Result row model for the DataGrid ─────────────────────────────────────────

public sealed class ResultRow
{
    public string Time          { get; init; } = "";
    public string InstructionId { get; init; } = "";
    public string ElementId     { get; init; } = "";
    public string Status        { get; init; } = "";
    public string StatusColor   { get; init; } = "#b0c4e0";
    public long   DurationMs    { get; init; }

    public static ResultRow FromActionResult(ActionResult r)
    {
        string status, color;
        if (r.Success)
        {
            status = "✓  Created";
            color  = "#40c060";
        }
        else if (r.Error?.StartsWith("SC1_") == true)
        {
            status = "🔴 SC1 BLOCKED";
            color  = "#e84040";
        }
        else if (r.Error?.StartsWith("SC2_") == true)
        {
            status = "🟡 SC2 Pending";
            color  = "#f5a623";
        }
        else
        {
            status = $"✗  {r.Error?[..Math.Min(35, r.Error?.Length ?? 0)]}";
            color  = "#c06060";
        }

        return new ResultRow
        {
            Time          = DateTime.Now.ToString("HH:mm:ss"),
            InstructionId = r.InstructionId.Length >= 8 ? r.InstructionId[..8] : r.InstructionId,
            ElementId     = r.ElementId ?? "—",
            Status        = status,
            StatusColor   = color,
            DurationMs    = r.DurationMs,
        };
    }
}

// ── Minimal relay command ─────────────────────────────────────────────────────

public sealed class RelayCommand : ICommand
{
    private readonly Action _execute;
    public RelayCommand(Action execute) => _execute = execute;
    public bool CanExecute(object? parameter) => true;
    public void Execute(object? parameter) => _execute();
    public event EventHandler? CanExecuteChanged;
}
