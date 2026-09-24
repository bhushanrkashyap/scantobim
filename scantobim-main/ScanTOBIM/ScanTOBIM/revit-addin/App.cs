// ─────────────────────────────────────────────────────────────────────────────
// App.cs — IExternalApplication entry point
//
// Registers:
//   • Dockable panel (ScanToBIM Agent) on the right side of Revit
//   • Ribbon tab "ScanToBIM" with five push buttons:
//       "Start Agent"   → opens/shows the agent panel and HTTP bridge
//       "Load Params"   → binds the 9 shared parameters to element categories
//       "Process Scan"  → local point cloud processing via stb-processor sidecar
//       "Build Geometry" → create DirectShape elements from last processed scan
//       "Sync to Backend" → POST segments to FastAPI ingest-elements endpoint
// ─────────────────────────────────────────────────────────────────────────────

using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using ScanToBIM.UI;

namespace ScanToBIM;

[Regeneration(RegenerationOption.Manual)]
public class App : IExternalApplication
{
    // Fixed GUID for the dockable pane — must match ScanToBIMAgent.addin
    public static readonly Guid PanelGuid = new("A1B2C3D4-E5F6-7890-ABCD-EF1234567890");

    public Result OnStartup(UIControlledApplication application)
    {
        try
        {
            // ── Register dockable pane ────────────────────────────────────────
            application.RegisterDockablePane(
                new DockablePaneId(PanelGuid),
                "ScanToBIM Agent",
                new AgentPanelCreator());

            // ── Add ribbon tab ────────────────────────────────────────────────
            try
            {
                application.CreateRibbonTab("ScanToBIM");
            }
            catch
            {
                // Tab may already exist if add-in reloaded — safe to ignore
            }

            var ribbonPanel  = application.CreateRibbonPanel("ScanToBIM", "Agent");
            var assemblyPath = typeof(App).Assembly.Location;

            // "Start Agent" button
            ribbonPanel.AddItem(new PushButtonData(
                "StartAgent",
                "Start\nAgent",
                assemblyPath,
                "ScanToBIM.Commands.StartAgentCommand")
            {
                ToolTip      = "Open the ScanToBIM Agent panel and start the HTTP bridge on port 8766.",
                LongDescription =
                    "Starts the Revit bridge that receives ElementInstructions from the Python AI agent. " +
                    "The Python agent must be running on port 8765 and have REVIT_BRIDGE_URL=http://localhost:8766.",
            });

            // "Load Params" button
            ribbonPanel.AddItem(new PushButtonData(
                "LoadParams",
                "Load\nParams",
                assemblyPath,
                "ScanToBIM.Commands.LoadSharedParamsCommand")
            {
                ToolTip      = "Bind the 9 ScanToBIM shared parameters to Walls, Floors, Columns, and Pipes.",
                LongDescription =
                    "Loads ScanToBIM.SharedParameters.txt and binds: " +
                    "ScanToBIM_SafetyCategory, ScanToBIM_ScanStationId, " +
                    "ScanToBIM_SegmentId, ScanToBIM_AgentSession, and compliance fields. " +
                    "Safe to run multiple times (idempotent).",
            });

            // "Process Scan" button (Sprint P2 — local sidecar processing)
            ribbonPanel.AddItem(new PushButtonData(
                "ProcessScan",
                "Process\nScan",
                assemblyPath,
                "ScanToBIM.Commands.ProcessScanCommand")
            {
                ToolTip         = "Process a local point cloud file without uploading to the backend.",
                LongDescription =
                    "Runs stb-processor.exe (bundled Python sidecar) locally to segment " +
                    "the scan file — no file size limits, no network upload. " +
                    "Results are stored in memory for the 'Build Geometry' command (Sprint P3). " +
                    "Supported formats: .e57 .las .laz .ply .pcd .xyz",
            });


            // "Build Geometry" button (Sprint P3 — DirectShape from scan results)
            ribbonPanel.AddItem(new PushButtonData(
                "BuildGeometry",
                "Build\nGeometry",
                assemblyPath,
                "ScanToBIM.Commands.BuildGeometryCommand")
            {
                ToolTip         = "Create DirectShape elements from the last processed scan.",
                LongDescription =
                    "Reads the scan results stored by 'Process Scan' and creates colour-coded " +
                    "DirectShape elements in the active document. " +
                    "Green = high confidence (≥85%), amber = medium (60–85%), red = low (<60%). " +
                    "ScanToBIM_SegmentId shared parameter written per element for traceability.",
            });


            // "Sync to Backend" button (Sprint P4 — ingest-elements endpoint)
            ribbonPanel.AddItem(new PushButtonData(
                "SyncToBackend",
                "Sync to\nBackend",
                assemblyPath,
                "ScanToBIM.Commands.SyncToBackendCommand")
            {
                ToolTip         = "Upload classified segments to the InfraIQ backend (no raw file transfer).",
                LongDescription =
                    "Authenticates, creates a session, then POSTs the structured segment " +
                    "data to /sessions/{id}/ingest-elements. The backend runs classification, " +
                    "safety gate evaluation, and clash detection. " +
                    "View results on the web dashboard.",
            });

            return Result.Succeeded;
        }
        catch (Exception ex)
        {
            TaskDialog.Show("ScanToBIM — Startup Error", ex.Message);
            return Result.Failed;
        }
    }

    public Result OnShutdown(UIControlledApplication application)
    {
        // Cleanly stop the HTTP bridge to free port 8766
        AgentPanelPage.Bridge?.Dispose();
        return Result.Succeeded;
    }
}

// ── Dockable pane factory ─────────────────────────────────────────────────────

internal sealed class AgentPanelCreator : IDockablePaneProvider
{
    public void SetupDockablePane(DockablePaneProviderData data)
    {
        // Note: uiApp is null here — the panel initialises its bridge
        // when ShowOrCreate() is called from StartAgentCommand
        data.FrameworkElement = new AgentPanelPage(null!);
        data.InitialState = new DockablePaneState
        {
            DockPosition = DockPosition.Right,
            TabBehind    = DockablePanes.BuiltInDockablePanes.ProjectBrowser,
        };
    }
}
