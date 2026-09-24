// ─────────────────────────────────────────────────────────────────────────────
// BuildGeometryCommand.cs — "Build Geometry" ribbon button (Sprint P3)
//
// Reads ScanResultStore.Latest (populated by ProcessScanCommand) and
// materialises each SidecarSegment as a colour-coded DirectShape element
// in the active Revit document.
//
// Prerequisite: "Process Scan" must have been run first in this session.
// ─────────────────────────────────────────────────────────────────────────────

using System;
using System.Linq;
using System.Text;
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using ScanToBIM.Models;
using ScanToBIM.Services;

// Use fully qualified name to resolve ambiguity
namespace ScanToBIM.Commands;

[Transaction(TransactionMode.Manual)]
[Regeneration(RegenerationOption.Manual)]
public class BuildGeometryCommand : IExternalCommand
{

    public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
    {
        SidecarResults? results = null;
        var resultsPath = Models.ScanResultStore.LatestResultsPath;

        // ── Always prompt user to pick a results JSON file ───────────────────
        var resultsDir = System.IO.Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "ScanToBIM", "Results");

        var openFileDialog = new Microsoft.Win32.OpenFileDialog
        {
            Filter           = "JSON Files (*.json)|*.json|All Files (*.*)|*.*",
            Title            = "Select Scan Results JSON File",
            InitialDirectory = System.IO.Directory.Exists(resultsDir) ? resultsDir : "",
        };

        // Pre-select the most recently generated file if available
        if (!string.IsNullOrEmpty(resultsPath)
            && System.IO.File.Exists(resultsPath))
        {
            openFileDialog.FileName = resultsPath;
        }
        else if (System.IO.Directory.Exists(resultsDir))
        {
            try
            {
                var newestFile = new System.IO.DirectoryInfo(resultsDir)
                    .GetFiles("*.json")
                    .OrderByDescending(f => f.LastWriteTime)
                    .FirstOrDefault();

                if (newestFile != null)
                {
                    openFileDialog.FileName = newestFile.FullName;
                }
            }
            catch { /* ignore directory listing errors */ }
        }

        if (openFileDialog.ShowDialog() != true)
            return Result.Cancelled;

        try
        {
            string jsonContent = System.IO.File.ReadAllText(openFileDialog.FileName);
            results = System.Text.Json.JsonSerializer.Deserialize<SidecarResults>(jsonContent, new System.Text.Json.JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true,
                PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.SnakeCaseLower
            });

            if (results != null)
            {
                Models.ScanResultStore.Latest = results;
                Models.ScanResultStore.LatestResultsPath = openFileDialog.FileName;
            }
        }
        catch (Exception ex)
        {
            TaskDialog.Show("ScanToBIM — Error", $"Failed to load JSON file: {ex.Message}");
            return Result.Failed;
        }

        if (results is null || results.Segments.Count == 0)
        {
            TaskDialog.Show(
                "ScanToBIM — No Scan Data",
                "No scan segments found in the selected file.");
            return Result.Cancelled;
        }

        var doc  = commandData.Application.ActiveUIDocument?.Document;
        if (doc is null)
        {
            message = "No active document.";
            return Result.Failed;
        }

        // ── Confirm ───────────────────────────────────────────────────────────
        var meta    = results.Metadata;
        var confirm = new TaskDialog("ScanToBIM — Build Geometry")
        {
            MainContent =
                $"File    : {meta.InputFile ?? "unknown"}\n" +
                $"Zone    : {meta.ZoneId ?? "zone-001"}\n" +
                $"Segments: {results.Segments.Count:N0}\n\n" +
                "This will create DirectShape elements in the active document.\n" +
                "Colour coding: green = high confidence, amber = medium, red = low.\n\n" +
                "Proceed?",
            CommonButtons = TaskDialogCommonButtons.Yes | TaskDialogCommonButtons.No,
        };

        if (confirm.Show() != TaskDialogResult.Yes)
            return Result.Cancelled;

        // ── Build inside a transaction ────────────────────────────────────────
        var builder = new ScanGeometryBuilder();
        ScanGeometryBuilder.BuildResult buildResult;

        using (var tx = new Transaction(doc, "ScanToBIM — Build Geometry"))
        {
            var failureOptions = tx.GetFailureHandlingOptions();
            failureOptions.SetFailuresPreprocessor(new WarningDismissor());
            tx.SetFailureHandlingOptions(failureOptions);

            tx.Start();
            try
            {                   // ── Delete previous DirectShape elements from ScanToBIM to avoid overlaps ──
                 var existingOldElements = new FilteredElementCollector(doc)
                     .OfClass(typeof(DirectShape))
                     .Cast<DirectShape>()
                     .Where(ds => ds.ApplicationId == "ScanToBIM.Processor")
                     .Select(ds => ds.Id)
                     .ToList();
                 
                 if (existingOldElements.Count > 0)
                 {
                     doc.Delete(existingOldElements);
                 }

                 // Delete all native pipes in the document to satisfy "remove all pipes" instruction
                 var existingPipes = new FilteredElementCollector(doc)
                     .OfClass(typeof(Autodesk.Revit.DB.Plumbing.Pipe))
                     .Select(p => p.Id)
                     .ToList();
                 if (existingPipes.Count > 0)
                 {
                     doc.Delete(existingPipes);
                 }

                buildResult = ScanGeometryBuilder.Build(doc, results.Segments);
                tx.Commit();
            }
            catch (Exception ex)
            {
                tx.RollBack();
                message = $"Geometry build failed: {ex.Message}";
                return Result.Failed;
            }
        }

        // ── Select created elements so user can see them ──────────────────────
        if (buildResult.ElementIds.Count > 0)
        {
            try
            {
                commandData.Application.ActiveUIDocument
                    .Selection.SetElementIds(buildResult.ElementIds);
            }
            catch { /* selection is a UX nicety — ignore if it fails */ }
        }

        // ── Summary ───────────────────────────────────────────────────────────
        ShowSummary(buildResult, results);
        return Result.Succeeded;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static void ShowSummary(
        ScanGeometryBuilder.BuildResult result, SidecarResults scanResults)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Created : {result.Created:N0} DirectShape elements");
        sb.AppendLine($"Skipped : {result.Skipped:N0} (degenerate or error)");

        // Breakdown by shape type
        var byShape = scanResults.Segments
            .GroupBy(s => s.Shape)
            .OrderByDescending(g => g.Count());

        sb.AppendLine();
        sb.AppendLine("By shape:");
        foreach (var g in byShape)
            sb.AppendLine($"  {g.Key,-22} {g.Count(),4}");

        if (result.Errors.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine($"Errors ({result.Errors.Count}):");
            foreach (var err in result.Errors.Take(5))
                sb.AppendLine($"  • {err}");
            if (result.Errors.Count > 5)
                sb.AppendLine($"  … and {result.Errors.Count - 5} more");
        }

        sb.AppendLine();
        sb.AppendLine("Next: use 'Start Agent' to run classification and push to backend.");

        TaskDialog.Show("ScanToBIM — Geometry Built", sb.ToString());
    }
}

internal class WarningDismissor : IFailuresPreprocessor
{
    public FailureProcessingResult PreprocessFailures(FailuresAccessor failuresAccessor)
    {
        var failures = failuresAccessor.GetFailureMessages();
        foreach (var f in failures)
        {
            if (f.GetSeverity() == FailureSeverity.Warning)
            {
                failuresAccessor.DeleteWarning(f);
            }
        }
        return FailureProcessingResult.Continue;
    }
}
