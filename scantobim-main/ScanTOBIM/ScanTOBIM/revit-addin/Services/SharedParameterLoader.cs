// ─────────────────────────────────────────────────────────────────────────────
// SharedParameterLoader.cs
//
// IExternalCommand that binds all 9 ScanToBIM shared parameters to the
// relevant Revit categories (Walls, Floors, StructuralColumns, etc.).
//
// Safe to run multiple times — idempotent.
// Mapped to the "Load Params" ribbon button in App.cs.
// ─────────────────────────────────────────────────────────────────────────────

using System;
using System.Collections.Generic;
using System.IO;
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.DB.Structure;
using Autodesk.Revit.UI;
using ScanToBIM.SafetyGate;
using BuiltInParameterGroup = Autodesk.Revit.DB.GroupTypeId;

namespace ScanToBIM.Services;

[Transaction(TransactionMode.Manual)]
public class SharedParameterLoader : IExternalCommand
{
    public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
    {
        var doc = commandData.Application.ActiveUIDocument.Document;
        var app = commandData.Application.Application;

        // ── Locate the shared parameter file ─────────────────────────────────
        var assemblyDir      = Path.GetDirectoryName(typeof(SharedParameterLoader).Assembly.Location)!;
        var sharedParamPath  = Path.Combine(assemblyDir, "Resources", "ScanToBIM.SharedParameters.txt");

        if (!File.Exists(sharedParamPath))
        {
            message = $"Shared parameter file not found.\nExpected at: {sharedParamPath}";
            return Result.Failed;
        }

        // ── Open shared parameter file ────────────────────────────────────────
        // Save existing path so we can restore it after loading ours
        var previousPath = app.SharedParametersFilename;
        app.SharedParametersFilename = sharedParamPath;

        var sharedParamsFile = app.OpenSharedParameterFile();
        if (sharedParamsFile == null)
        {
            message = $"Failed to open shared parameter file at:\n{sharedParamPath}";
            app.SharedParametersFilename = previousPath;
            return Result.Failed;
        }

        // ── Get or create the ScanToBIM group ─────────────────────────────────
        var group = sharedParamsFile.Groups.get_Item("ScanToBIM")
                    ?? sharedParamsFile.Groups.Create("ScanToBIM");

        // ── Build category set ────────────────────────────────────────────────
        var categories = new CategorySet();
        void TryAddCategory(BuiltInCategory bic)
        {
            try { categories.Insert(doc.Settings.Categories.get_Item(bic)); }
            catch { /* category not available in this template — skip */ }
        }
        TryAddCategory(BuiltInCategory.OST_Walls);
        TryAddCategory(BuiltInCategory.OST_Floors);
        TryAddCategory(BuiltInCategory.OST_StructuralColumns);
        TryAddCategory(BuiltInCategory.OST_StructuralFraming);
        TryAddCategory(BuiltInCategory.OST_PipeCurves);
        TryAddCategory(BuiltInCategory.OST_GenericModel);

        var binding = doc.Application.Create.NewInstanceBinding(categories);

        // ── Bind parameters ───────────────────────────────────────────────────
        // All 9 params — SharedParamNames.All keeps this in sync automatically
        var paramNames = SharedParamNames.All;

        var results = new List<string>();

        using var tx = new Transaction(doc, "ScanToBIM: Load Shared Parameters");
        tx.Start();

        foreach (var paramName in paramNames)
        {
            try
            {
                // Get or create the definition in the file
                var def = group.Definitions.get_Item(paramName)
                          ?? group.Definitions.Create(
                              new ExternalDefinitionCreationOptions(paramName, SpecTypeId.String.Text)
                              {
                                  UserModifiable  = false,
                                  Visible         = true,
                                  Description     = $"ScanToBIM compliance parameter: {paramName}"
                              });

                // Idempotent: only bind if not already bound
                var existingBinding = doc.ParameterBindings.get_Item(def);
                if (existingBinding == null)
                {
                    doc.ParameterBindings.Insert(def, binding, BuiltInParameterGroup.IdentityData);
                    results.Add($"  ✓  {paramName}  —  bound to {categories.Size} categories");
                }
                else
                {
                    results.Add($"  ·  {paramName}  —  already bound (skipped)");
                }
            }
            catch (Exception ex)
            {
                results.Add($"  ✗  {paramName}  —  FAILED: {ex.Message}");
            }
        }

        tx.Commit();

        // Restore previous shared param file
        app.SharedParametersFilename = previousPath;

        // ── Show result dialog ────────────────────────────────────────────────
        TaskDialog.Show(
            "ScanToBIM — Shared Parameters",
            "Results:\n\n" + string.Join("\n", results) +
            "\n\nSave and re-open the project to verify parameter visibility in properties.");

        return Result.Succeeded;
    }
}
