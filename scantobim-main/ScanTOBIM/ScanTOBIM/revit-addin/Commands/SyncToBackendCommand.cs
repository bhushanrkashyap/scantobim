// ─────────────────────────────────────────────────────────────────────────────
// SyncToBackendCommand.cs — "Sync to InfraIQ" ribbon button (Sprint P4)
//
// Reads ScanResultStore.Latest (set by ProcessScanCommand) and POSTs the
// structured segment data to the FastAPI backend ingest-elements endpoint.
// No raw file upload — only the ~KB JSON payload crosses the network.
//
// Flow:
//   1. Guard: scan results must be in memory (run Process Scan first)
//   2. BackendSyncDialog (WPF) — prompts URL, username, password
//   3. POST /auth/token → JWT
//   4. POST /sessions   → new session_id
//   5. POST /sessions/{id}/ingest-elements → classification + gates
//   6. TaskDialog summary: segment count, gates pending, clashes
// ─────────────────────────────────────────────────────────────────────────────

using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using ScanToBIM.Models;

namespace ScanToBIM.Commands;

[Transaction(TransactionMode.ReadOnly)]
[Regeneration(RegenerationOption.Manual)]
public class SyncToBackendCommand : IExternalCommand
{
    private static readonly JsonSerializerOptions _json = new()
    {
        PropertyNamingPolicy        = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true,
    };

    public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
    {
        // ── Guard ─────────────────────────────────────────────────────────────
        var results = ScanResultStore.Latest;
        if (results is null || results.Segments.Count == 0)
        {
            TaskDialog.Show(
                "ScanToBIM — No Scan Data",
                "No scan results found in memory.\n\nRun 'Process Scan' first.");
            return Result.Cancelled;
        }

        // ── Prompt for connection details ─────────────────────────────────────
        var dialog = new BackendSyncDialog();
        if (dialog.ShowDialog() != true)
            return Result.Cancelled;

        // ── Run sync (async via Task.Run — keep UI responsive) ─────────────────
        SyncResult? syncResult = null;
        string?     error      = null;

        var progress = new SyncProgressWindow();
        progress.Show();

        Task.Run(async () =>
        {
            try
            {
                syncResult = await RunSyncAsync(
                    dialog.BackendUrl, dialog.Username, dialog.Password,
                    results, progress.SetStatus);
            }
            catch (Exception ex)
            {
                error = ex.Message;
            }
            finally
            {
                progress.Dispatcher.Invoke(() => progress.Close());
            }
        }).Wait();

        if (error is not null)
        {
            TaskDialog.Show("ScanToBIM — Sync Error", error);
            return Result.Failed;
        }

        if (syncResult is not null)
            ShowSummary(syncResult, results);

        return Result.Succeeded;
    }

    // ── HTTP sync logic ───────────────────────────────────────────────────────

    private async Task<SyncResult> RunSyncAsync(
        string url, string username, string password,
        SidecarResults scanResults,
        Action<string> status)
    {
        using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
        var baseUrl    = url.TrimEnd('/');

        // 1. Login
        status("Authenticating…");
        var tokenResp = await http.PostAsync(
            $"{baseUrl}/auth/token",
            new FormUrlEncodedContent(new Dictionary<string, string>
            {
                ["username"] = username,
                ["password"] = password,
            }));

        if (!tokenResp.IsSuccessStatusCode)
            throw new InvalidOperationException(
                $"Login failed ({tokenResp.StatusCode}). Check credentials.");

        var tokenJson  = await tokenResp.Content.ReadAsStringAsync();
        var tokenDoc   = JsonDocument.Parse(tokenJson);
        var token      = tokenDoc.RootElement.GetProperty("access_token").GetString()
                         ?? throw new InvalidOperationException("No access_token in response.");

        http.DefaultRequestHeaders.Authorization =
            new AuthenticationHeaderValue("Bearer", token);

        // 2. Create session
        status("Creating session…");
        var sessionPayload = JsonSerializer.Serialize(new
        {
            site_name    = scanResults.Metadata.InputFile ?? "Plugin Scan",
            zone_id      = scanResults.Metadata.ZoneId ?? "zone-001",
            use_synthetic = false,
        });

        var sessionResp = await http.PostAsync(
            $"{baseUrl}/sessions",
            new StringContent(sessionPayload, Encoding.UTF8, "application/json"));

        sessionResp.EnsureSuccessStatusCode();
        var sessionJson = await sessionResp.Content.ReadAsStringAsync();
        var sessionDoc  = JsonDocument.Parse(sessionJson);
        var sessionId   = sessionDoc.RootElement.GetProperty("session_id").GetString()
                          ?? throw new InvalidOperationException("No session_id in response.");

        // 3. Post ingest-elements
        status($"Sending {scanResults.Segments.Count:N0} segments…");
        var ingestPayload = JsonSerializer.Serialize(new
        {
            segments      = scanResults.Segments,
            scan_metadata = scanResults.Metadata,
            source        = "revit_plugin",
        }, new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower });

        var ingestResp = await http.PostAsync(
            $"{baseUrl}/sessions/{sessionId}/ingest-elements",
            new StringContent(ingestPayload, Encoding.UTF8, "application/json"));

        ingestResp.EnsureSuccessStatusCode();
        var ingestJson = await ingestResp.Content.ReadAsStringAsync();
        var ingestDoc  = JsonDocument.Parse(ingestJson);

        status("Done.");
        return new SyncResult
        {
            SessionId          = sessionId,
            SegmentsIngested   = ingestDoc.RootElement.GetProperty("segments_ingested").GetInt32(),
            ElementsClassified = ingestDoc.RootElement.GetProperty("elements_classified").GetInt32(),
            GatesPending       = ingestDoc.RootElement.GetProperty("gates_pending").GetInt32(),
            ClashCount         = ingestDoc.RootElement.GetProperty("clash_count").GetInt32(),
            Sc1Blocked         = ingestDoc.RootElement.GetProperty("sc1_blocked").GetInt32(),
            BackendUrl         = baseUrl,
        };
    }

    private static void ShowSummary(SyncResult r, SidecarResults scan)
    {
        var msg = $"""
            Session   : {r.SessionId[..8]}…
            Segments  : {r.SegmentsIngested:N0}
            Classified: {r.ElementsClassified:N0}
            SC1 blocked: {r.Sc1Blocked}
            Gates pending (SC2): {r.GatesPending}
            Clashes   : {r.ClashCount}

            Open the web dashboard to review gates and push elements to Revit.
            {r.BackendUrl}/ui/gates.html?session={r.SessionId}
            """;

        TaskDialog.Show("ScanToBIM — Sync Complete", msg);
    }

    private sealed class SyncResult
    {
        public string SessionId          { get; init; } = "";
        public int    SegmentsIngested   { get; init; }
        public int    ElementsClassified { get; init; }
        public int    GatesPending       { get; init; }
        public int    ClashCount         { get; init; }
        public int    Sc1Blocked         { get; init; }
        public string BackendUrl         { get; init; } = "";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend connection dialog  (pure C# WPF)
// ─────────────────────────────────────────────────────────────────────────────

internal sealed class BackendSyncDialog : Window
{
    public string BackendUrl { get; private set; } = "http://localhost:8765";
    public string Username   { get; private set; } = "admin";
    public string Password   { get; private set; } = "";

    private readonly System.Windows.Controls.TextBox     _urlBox;
    private readonly System.Windows.Controls.TextBox     _userBox;
    private readonly PasswordBox _passBox;

    public BackendSyncDialog()
    {
        Title                 = "ScanToBIM — Sync to Backend";
        Width                 = 380;
        Height                = 240;
        ResizeMode            = ResizeMode.NoResize;
        WindowStartupLocation = WindowStartupLocation.CenterScreen;
        Background            = Brushes.White;

        var sp = new StackPanel { Margin = new Thickness(16) };

        sp.Children.Add(Label("Backend URL"));
        _urlBox = Field("http://localhost:8765");
        sp.Children.Add(_urlBox);

        sp.Children.Add(Label("Username"));
        _userBox = Field("admin");
        sp.Children.Add(_userBox);

        sp.Children.Add(Label("Password"));
        _passBox = new PasswordBox { Padding = new Thickness(4), Margin = new Thickness(0,0,0,12) };
        sp.Children.Add(_passBox);

        var btns = new StackPanel
        {
            Orientation         = Orientation.Horizontal,
            HorizontalAlignment = HorizontalAlignment.Right,
        };

        var ok = new Button
        {
            Content         = "Sync",
            IsDefault       = true,
            Width           = 72, Height = 28,
            Margin          = new Thickness(0,0,8,0),
            Background      = new SolidColorBrush(System.Windows.Media.Color.FromRgb(37, 99, 235)),
            Foreground      = Brushes.White,
            BorderThickness = new Thickness(0),
        };
        ok.Click += (_, _) =>
        {
            BackendUrl = _urlBox.Text.Trim();
            Username   = _userBox.Text.Trim();
            Password   = _passBox.Password;
            DialogResult = true;
        };

        btns.Children.Add(ok);
        btns.Children.Add(new Button { Content = "Cancel", IsCancel = true, Width = 72, Height = 28 });
        sp.Children.Add(btns);
        Content = sp;
    }

    private static TextBlock Label(string text) =>
        new TextBlock { Text = text, FontSize = 12, Margin = new Thickness(0,4,0,2) };

    private static System.Windows.Controls.TextBox Field(string @default) =>
        new System.Windows.Controls.TextBox { Text = @default, Padding = new Thickness(4), Margin = new Thickness(0,0,0,8) };
}

// ─────────────────────────────────────────────────────────────────────────────
// Lightweight progress window (shown while HTTP calls are in flight)
// ─────────────────────────────────────────────────────────────────────────────

internal sealed class SyncProgressWindow : Window
{
    private readonly TextBlock _label;

    public SyncProgressWindow()
    {
        Title                 = "ScanToBIM — Syncing…";
        Width                 = 340; Height = 100;
        ResizeMode            = ResizeMode.NoResize;
        WindowStartupLocation = WindowStartupLocation.CenterScreen;
        Background            = Brushes.White;

        _label = new TextBlock
        {
            Text                = "Connecting…",
            FontSize            = 13,
            HorizontalAlignment = HorizontalAlignment.Center,
            VerticalAlignment   = VerticalAlignment.Center,
            Margin              = new Thickness(16),
        };
        Content = _label;
    }

    public void SetStatus(string msg) =>
        Dispatcher.Invoke(() => _label.Text = msg);
}
