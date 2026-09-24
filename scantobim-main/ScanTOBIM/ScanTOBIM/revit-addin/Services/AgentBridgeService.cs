// ─────────────────────────────────────────────────────────────────────────────
// AgentBridgeService.cs
//
// Hosts a lightweight HTTP server on port 8766 that the Python agent pushes
// ElementInstructions to. Instructions are queued and dispatched to Revit's
// main thread via Revit's ExternalEvent mechanism.
//
// Endpoints exposed:
//   GET  /health                      — bridge status + queue depth
//   POST /revit-actions               — enqueue one ElementInstruction
//   GET  /revit-results/{id}          — poll for result of a processed instruction
//
// Architecture:
//   Python agent (port 8765) ──POST──▶ HttpListener (port 8766, background thread)
//                                              │
//                                   ConcurrentQueue<ElementInstruction>
//                                              │
//                                   ExternalEvent.Raise()
//                                              │
//                              Revit idle thread ──▶ RevitElementFactory.Execute()
//                                              │
//                                   ConcurrentDictionary results
//                                              │
//   Python agent ◀──GET───────────────── /revit-results/{id}
// ─────────────────────────────────────────────────────────────────────────────

using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Text;
using System.Text.Json;
using Autodesk.Revit.UI;
using ScanToBIM.Models;

namespace ScanToBIM.Services;

public sealed class AgentBridgeService : IDisposable
{
    // ── Configuration ─────────────────────────────────────────────────────────
    public const int  Port    = 8766;
    private const int MaxBatch = 50; // max elements to process per Revit idle event

    // ── State ─────────────────────────────────────────────────────────────────
    private readonly HttpListener                                _listener   = new();
    private readonly ConcurrentQueue<ElementInstruction>         _queue      = new();
    private readonly ConcurrentDictionary<string, ActionResult>  _results    = new();
    // P4: ActionRateLimiter — max 10 concurrent Revit transactions
    private readonly SemaphoreSlim _rateLimiter = new(10, 10);
    // P4: SessionRegistry — tracks active zones + disciplines per session
    private readonly ConcurrentDictionary<string, SessionInfo>   _sessions   = new();
    private readonly ExternalEvent                             _revitEvent;
    private readonly BridgeEventHandler                        _handler;

    private Thread? _listenerThread;
    private volatile bool _running;
    private string? _lastClearedSessionId;

    // ── Sprint 2: Agent callback URL (push results back to Python agent) ──────
    // When set, Revit posts ActionResult to this URL after each element creation.
    // Format: http://localhost:8765/sessions/{session_id}/revit-callback
    private string? _agentCallbackBase;

    // ── Stats (read by WPF ViewModel) ─────────────────────────────────────────
    public int  ElementsCreated { get; private set; }
    public int  ElementsBlocked { get; private set; }
    public int  QueueDepth      => _queue.Count;
    public bool IsRunning       => _running;

    // ── Sprint 2: Configure agent callback base URL ────────────────────────────
    public void SetAgentCallbackBase(string agentBaseUrl)
    {
        // agentBaseUrl = "http://localhost:8765"
        _agentCallbackBase = agentBaseUrl.TrimEnd('/');
        Log($"[Bridge] Agent callback base: {_agentCallbackBase}");
    }

    // ── Events for WPF data binding ───────────────────────────────────────────
    public event Action<string>?      OnLog;
    public event Action<ActionResult>? OnResult;

    // ── JSON options (shared, thread-safe) ───────────────────────────────────
    private static readonly JsonSerializerOptions _jsonOpts = new()
    {
        PropertyNameCaseInsensitive = true,
        Converters = { new System.Text.Json.Serialization.JsonStringEnumConverter() },
        WriteIndented = false,
    };

    // ── Constructor ───────────────────────────────────────────────────────────

    public AgentBridgeService(UIApplication uiApp)
    {
        _handler    = new BridgeEventHandler(this, uiApp);
        _revitEvent = ExternalEvent.Create(_handler);
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    public void Start()
    {
        if (_running) return;
        _running = true;

        _listener.Prefixes.Add($"http://localhost:{Port}/");
        _listener.Start();

        _listenerThread = new Thread(ListenLoop)
        {
            IsBackground = true,
            Name         = "ScanToBIM-Bridge-Listener",
        };
        _listenerThread.Start();

        Log($"[Bridge] Listening on http://localhost:{Port}/");
        Log($"[Bridge] Python agent should point REVIT_BRIDGE_URL=http://localhost:{Port}");
    }

    public void Stop()
    {
        if (!_running) return;
        _running = false;

        try { _listener.Stop(); } catch { /* already stopped */ }

        Log("[Bridge] Stopped.");
    }

    public void Dispose()
    {
        Stop();
        _listener.Close();
    }

    // ── HTTP listener loop (background thread) ────────────────────────────────

    private void ListenLoop()
    {
        while (_running)
        {
            try
            {
                var ctx = _listener.GetContext();
                ThreadPool.QueueUserWorkItem(_ => HandleRequest(ctx));
            }
            catch (HttpListenerException) when (!_running)
            {
                break; // normal shutdown
            }
            catch (Exception ex)
            {
                Log($"[Bridge] Listener error: {ex.Message}");
            }
        }
    }

    private void HandleRequest(HttpListenerContext ctx)
    {
        var req  = ctx.Request;
        var resp = ctx.Response;

        try
        {
            var path   = req.Url?.AbsolutePath ?? "/";
            var method = req.HttpMethod;

            // ── GET /health ─────────────────────────────────────────────────
            if (method == "GET" && path == "/health")
            {
                WriteJson(resp, new BridgeHealthResponse
                {
                    Connected       = true,
                    QueueDepth      = QueueDepth,
                    ElementsCreated = ElementsCreated,
                    ElementsBlocked = ElementsBlocked,
                }, 200);
                return;
            }

            // ── POST /revit-actions ─────────────────────────────────────────
            if (method == "POST" && path == "/revit-actions")
            {
                using var reader      = new StreamReader(req.InputStream, Encoding.UTF8);
                var body              = reader.ReadToEnd();
                var instruction       = JsonSerializer.Deserialize<ElementInstruction>(body, _jsonOpts);

                if (instruction == null)
                {
                    WriteText(resp, "Invalid or missing instruction payload.", 400);
                    return;
                }

                _queue.Enqueue(instruction);
                Log($"[Bridge] Queued  {instruction.ElementType,-8} " +
                    $"[{ShortId(instruction.InstructionId)}]  " +
                    $"{instruction.SafetyCategory}  zone={instruction.ZoneId}");

                // Raise Revit external event to process the queue on next idle
                _revitEvent.Raise();

                WriteJson(resp, new { queued = true, instruction_id = instruction.InstructionId }, 202);
                return;
            }

            // ── GET /revit-results/{instruction_id} ─────────────────────────
            if (method == "GET" && path.StartsWith("/revit-results/"))
            {
                var instrId = path["/revit-results/".Length..];
                if (_results.TryGetValue(instrId, out var result))
                    WriteJson(resp, result, 200);
                else
                    WriteJson(resp, new { status = "pending", instruction_id = instrId }, 202);
                return;
            }

            WriteText(resp, $"Not found: {method} {path}", 404);
        }
        catch (Exception ex)
        {
            Log($"[Bridge] Request handler error: {ex.Message}");
            WriteText(resp, ex.Message, 500);
        }
        finally
        {
            resp.Close();
        }
    }

    // ── Called by ExternalEventHandler on Revit main thread ───────────────────

    internal void ProcessQueue(Autodesk.Revit.DB.Document doc)
    {
        if (doc == null) return;

        var factory   = new RevitElementFactory(doc);
        int processed = 0;

        while (_queue.TryDequeue(out var instruction) && processed < MaxBatch)
        {
            // P4: ActionRateLimiter — wait for a slot (non-blocking check on main thread)
            if (!_rateLimiter.Wait(0))
            {
                // Re-queue and exit — will be retried on next Revit idle event
                _queue.Enqueue(instruction);
                Log($"[Bridge] Rate limit reached ({_rateLimiter.CurrentCount}/10 slots). Re-queued.");
                break;
            }

            string? currentSessionId = null;
            if (instruction.Parameters != null &&
                instruction.Parameters.TryGetValue("session_id", out var sessIdObj) &&
                sessIdObj is System.Text.Json.JsonElement sessIdEl &&
                sessIdEl.ValueKind == System.Text.Json.JsonValueKind.String)
            {
                currentSessionId = sessIdEl.GetString();
            }

            if (instruction.ElementType == ElementType.pipe && currentSessionId != _lastClearedSessionId)
            {
                _lastClearedSessionId = currentSessionId;
                using (var txDelete = new Autodesk.Revit.DB.Transaction(doc, "ScanToBIM: Delete All Pipes"))
                {
                    txDelete.Start();
                    var existingPipes = new Autodesk.Revit.DB.FilteredElementCollector(doc)
                        .OfClass(typeof(Autodesk.Revit.DB.Plumbing.Pipe))
                        .Select(p => p.Id)
                        .ToList();
                    if (existingPipes.Count > 0)
                    {
                        doc.Delete(existingPipes);
                        Log($"[Bridge] Deleted {existingPipes.Count} existing pipes for new session {currentSessionId ?? "default"}.");
                    }
                    txDelete.Commit();
                }
            }

            ActionResult result;
            try
            {
                result = factory.Execute(instruction);
            }
            finally
            {
                _rateLimiter.Release();
            }

            _results[instruction.InstructionId] = result;

            // P4: Update SessionRegistry
            var sessionId = instruction.InstructionId; // use as session proxy in PoV
            _sessions.AddOrUpdate(
                sessionId,
                _ => new SessionInfo(instruction.ZoneId, instruction.Discipline.ToString()),
                (_, info) => info with { LastActivity = DateTime.UtcNow });

            if (result.Success)
            {
                ElementsCreated++;
                Log($"[Factory] ✓  {instruction.ElementType,-8}  →  {result.ElementId}  " +
                    $"({result.DurationMs} ms)  [{instruction.SafetyCategory}]");
            }
            else
            {
                ElementsBlocked++;
                var errPrefix = result.Error?.StartsWith("SC1_") == true ? "🔴 BLOCKED" :
                                result.Error?.StartsWith("SC2_") == true ? "🟡 PENDING" : "✗ ERROR";
                Log($"[Factory] {errPrefix}  {instruction.ElementType,-8}  " +
                    $"[{ShortId(instruction.InstructionId)}]  {result.Error?[..Math.Min(60, result.Error.Length)]}");
            }

            OnResult?.Invoke(result);

            // ── Sprint 2: Push result back to Python agent callback ────────────
            // Fire-and-forget on thread pool so Revit main thread isn't blocked
            if (_agentCallbackBase != null)
            {
                var callbackSessionId = instruction.InstructionId; // Renamed to avoid CS0136
                // Extract real session_id from parameters if present
                if (instruction.Parameters != null &&
                    instruction.Parameters.TryGetValue("session_id", out var sidObj) &&
                    sidObj is JsonElement sidEl &&
                    sidEl.ValueKind == JsonValueKind.String)
                {
                    callbackSessionId = sidEl.GetString() ?? callbackSessionId;
                }
                var callbackUrl = $"{_agentCallbackBase}/sessions/{callbackSessionId}/revit-callback";
                var resultCopy  = result;
                ThreadPool.QueueUserWorkItem(_ => PostCallbackResult(callbackUrl, resultCopy));
            }

            processed++;
        }
    }

    // ── Sprint 2: HTTP callback to Python agent ────────────────────────────────

    private void PostCallbackResult(string callbackUrl, ActionResult result)
    {
        try
        {
            using var http    = new System.Net.Http.HttpClient { Timeout = TimeSpan.FromSeconds(5) };
            var       json    = JsonSerializer.Serialize(result, _jsonOpts);
            var       content = new System.Net.Http.StringContent(json, Encoding.UTF8, "application/json");
            var       resp    = http.PostAsync(callbackUrl, content).GetAwaiter().GetResult();
            if (!resp.IsSuccessStatusCode)
                Log($"[Callback] Agent rejected result for {ShortId(result.InstructionId)}: {resp.StatusCode}");
            else
                Log($"[Callback] ✓ Pushed {ShortId(result.InstructionId)} → {callbackUrl}");
        }
        catch (Exception ex)
        {
            // Non-fatal — agent may not be running or callback URL may be wrong
            Log($"[Callback] Failed to push {ShortId(result.InstructionId)}: {ex.Message}");
        }
    }

    // ── Logging ───────────────────────────────────────────────────────────────

    public void Log(string message)
    {
        System.Diagnostics.Debug.WriteLine(message);
        OnLog?.Invoke($"[{DateTime.Now:HH:mm:ss}] {message}");
    }

    // ── HTTP helpers ──────────────────────────────────────────────────────────

    private void WriteJson(HttpListenerResponse resp, object data, int statusCode)
    {
        var json  = JsonSerializer.Serialize(data, _jsonOpts);
        var bytes = Encoding.UTF8.GetBytes(json);
        resp.StatusCode      = statusCode;
        resp.ContentType     = "application/json; charset=utf-8";
        resp.ContentLength64 = bytes.Length;
        resp.OutputStream.Write(bytes);
    }

    private static void WriteText(HttpListenerResponse resp, string text, int statusCode)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        resp.StatusCode      = statusCode;
        resp.ContentType     = "text/plain; charset=utf-8";
        resp.ContentLength64 = bytes.Length;
        resp.OutputStream.Write(bytes);
    }

    private static string ShortId(string id) =>
        id.Length >= 8 ? id[..8] : id;
}

// ── P4: SessionInfo record ────────────────────────────────────────────────────

/// <summary>Tracks zone + discipline activity for a session in the registry.</summary>
internal sealed record SessionInfo(string ZoneId, string Discipline)
{
    public DateTime LastActivity { get; init; } = DateTime.UtcNow;
    public int ElementsCreated   { get; init; } = 0;
}

// ── ExternalEventHandler — dispatches work to Revit main thread ───────────────

internal sealed class BridgeEventHandler : IExternalEventHandler
{
    private readonly AgentBridgeService _bridge;
    private readonly UIApplication      _uiApp;

    public BridgeEventHandler(AgentBridgeService bridge, UIApplication uiApp)
    {
        _bridge = bridge;
        _uiApp  = uiApp;
    }

    public void Execute(UIApplication app)
    {
        var doc = app.ActiveUIDocument?.Document;
        if (doc == null)
        {
            _bridge.Log("[Bridge] No active document — cannot process queue.");
            return;
        }
        _bridge.ProcessQueue(doc);
    }

    public string GetName() => "ScanToBIM Agent Bridge";
}
