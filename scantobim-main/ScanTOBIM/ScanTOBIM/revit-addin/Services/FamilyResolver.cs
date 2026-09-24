// ─────────────────────────────────────────────────────────────────────────────
// FamilyResolver.cs
//
// Resolves the Revit FamilySymbol to use for a given element type.
//
// Resolution priority:
//   1. revit_family_hint on the instruction  (operator hard-override)
//   2. family_map.json config entry          (site-specific preferred family)
//   3. Category first-match fallback         (any loaded family in that category)
//
// The resolved path ("hint" | "config" | "fallback") is returned in the result
// so the caller can write it to the audit trail — NQA-1 traceability.
// ─────────────────────────────────────────────────────────────────────────────

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using Autodesk.Revit.DB;
using ScanToBIM.Models;

namespace ScanToBIM.Services;

/// <summary>
/// Outcome of a family resolution attempt.
/// </summary>
public record FamilyResolution(
    FamilySymbol?  Symbol,
    string         Path,        // "hint" | "config" | "fallback" | "none"
    string?        FamilyName,
    string?        TypeName
);

/// <summary>
/// Resolves the best-matched Revit FamilySymbol for a scan-derived element.
/// Reads <c>family_map.json</c> from the addin directory at construction time;
/// missing or invalid maps degrade gracefully to category fallback.
/// </summary>
public sealed class FamilyResolver
{
    private readonly Document _doc;
    private readonly IReadOnlyDictionary<string, FamilyMapEntry> _map;

    // BuiltInCategory names → enum values for JSON config resolution
    private static readonly Dictionary<string, BuiltInCategory> _categoryLookup =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ["OST_MechanicalEquipment"] = BuiltInCategory.OST_MechanicalEquipment,
            ["OST_PipeAccessory"]       = BuiltInCategory.OST_PipeAccessory,
            ["OST_ElectricalEquipment"] = BuiltInCategory.OST_ElectricalEquipment,
            ["OST_LightingFixtures"]    = BuiltInCategory.OST_LightingFixtures,
            ["OST_Sprinklers"]          = BuiltInCategory.OST_Sprinklers,
            ["OST_FireAlarmDevices"]    = BuiltInCategory.OST_FireAlarmDevices,
            ["OST_FireProtection"]      = BuiltInCategory.OST_FireProtection,
            ["OST_PlumbingFixtures"]    = BuiltInCategory.OST_PlumbingFixtures,
            ["OST_StructuralFraming"]   = BuiltInCategory.OST_StructuralFraming,
            ["OST_StructuralColumns"]   = BuiltInCategory.OST_StructuralColumns,
            ["OST_Doors"]               = BuiltInCategory.OST_Doors,
            ["OST_Windows"]             = BuiltInCategory.OST_Windows,
            ["OST_PipeAccessories"]     = BuiltInCategory.OST_PipeAccessory,  // alias
        };

    public FamilyResolver(Document doc, string? mapFilePath = null)
    {
        _doc = doc;
        _map = LoadMap(mapFilePath ?? DefaultMapPath());
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// <summary>
    /// Resolve a FamilySymbol for the given instruction.
    /// </summary>
    /// <param name="instruction">The element instruction from the agent.</param>
    /// <param name="fallbackCategory">Category to use if config lookup fails.</param>
    public FamilyResolution Resolve(
        ElementInstruction instruction,
        BuiltInCategory    fallbackCategory)
    {
        var gp          = GeometryParamsFromInstruction(instruction);
        var elementKey  = instruction.ElementType.ToString();

        // Priority 1: explicit operator hint on the instruction
        if (!string.IsNullOrWhiteSpace(instruction.RevitFamilyHint))
        {
            var sym = FindSymbolByName(
                instruction.RevitFamilyHint,
                instruction.RevitTypeHint,
                fallbackCategory);
            if (sym != null)
                return new FamilyResolution(sym, "hint",
                    instruction.RevitFamilyHint, instruction.RevitTypeHint);
        }

        // Priority 2: family_map.json config entry
        if (_map.TryGetValue(elementKey, out var entry))
        {
            var typeName  = SelectType(entry, gp);
            var configCat = ResolveCategory(entry.Category, fallbackCategory);
            var sym       = FindSymbolByName(entry.Family, typeName, configCat);
            if (sym != null)
                return new FamilyResolution(sym, "config", entry.Family, typeName);
        }

        // Priority 3: category first-match fallback
        var fallback = FirstSymbolInCategory(fallbackCategory);
        return new FamilyResolution(
            fallback, fallback != null ? "fallback" : "none", null, null);
    }

    // ── Type rule evaluation ──────────────────────────────────────────────────

    private static string SelectType(
        FamilyMapEntry              entry,
        Dictionary<string, double>  gp)
    {
        foreach (var rule in entry.TypeRules)
        {
            if (RuleMatches(rule, gp))
                return rule.Type;
        }
        return entry.DefaultType ?? string.Empty;
    }

    private static bool RuleMatches(
        TypeRule                    rule,
        Dictionary<string, double>  gp)
    {
        if (rule.MaxDiameterMm.HasValue
            && gp.TryGetValue("diameter_mm", out var diam)
            && diam >= rule.MaxDiameterMm.Value)
            return false;

        if (rule.MaxHeightMm.HasValue
            && gp.TryGetValue("height_mm", out var ht)
            && ht >= rule.MaxHeightMm.Value)
            return false;

        if (rule.MaxMajorMm.HasValue
            && gp.TryGetValue("major_mm", out var maj)
            && maj >= rule.MaxMajorMm.Value)
            return false;

        return true;
    }

    // ── Revit lookups ─────────────────────────────────────────────────────────

    private FamilySymbol? FindSymbolByName(
        string?         familyName,
        string?         typeName,
        BuiltInCategory category)
    {
        if (string.IsNullOrWhiteSpace(familyName))
            return null;

        return new FilteredElementCollector(_doc)
            .OfClass(typeof(FamilySymbol))
            .OfCategory(category)
            .Cast<FamilySymbol>()
            .FirstOrDefault(s =>
                s.Family.Name.Equals(familyName, StringComparison.OrdinalIgnoreCase)
                && (string.IsNullOrWhiteSpace(typeName)
                    || s.Name.Equals(typeName, StringComparison.OrdinalIgnoreCase)));
    }

    private FamilySymbol? FirstSymbolInCategory(BuiltInCategory category) =>
        new FilteredElementCollector(_doc)
            .OfClass(typeof(FamilySymbol))
            .OfCategory(category)
            .Cast<FamilySymbol>()
            .FirstOrDefault();

    // ── Geometry params from instruction ──────────────────────────────────────

    private static Dictionary<string, double> GeometryParamsFromInstruction(
        ElementInstruction instruction)
    {
        var bb = instruction.BoundingBox;
        var w  = bb.WidthMm;
        var d  = bb.DepthMm;
        var h  = bb.HeightMm;
        var ordered = new[] { w, d, h }.OrderBy(x => x).ToArray();

        var gp = new Dictionary<string, double>
        {
            ["width_mm"]    = w,
            ["depth_mm"]    = d,
            ["height_mm"]   = h,
            ["minor_mm"]    = ordered[0],
            ["major_mm"]    = ordered[2],
            ["diameter_mm"] = (ordered[0] + ordered[1]) / 2.0,
        };

        // Merge pre-computed params sent by the Python pipeline
        // (diameter_mm from RANSAC, wall_thickness_mm, etc.)
        if (instruction.Parameters != null)
        {
            foreach (var kv in instruction.Parameters)
            {
                if (kv.Value is JsonElement je
                    && je.ValueKind == JsonValueKind.Number
                    && je.TryGetDouble(out var val))
                {
                    gp[kv.Key] = val;
                }
            }
        }

        return gp;
    }

    // ── Config loading ────────────────────────────────────────────────────────

    private static BuiltInCategory ResolveCategory(
        string?         configName,
        BuiltInCategory fallback)
    {
        if (!string.IsNullOrWhiteSpace(configName)
            && _categoryLookup.TryGetValue(configName, out var cat))
            return cat;
        return fallback;
    }

    private static IReadOnlyDictionary<string, FamilyMapEntry> LoadMap(string path)
    {
        try
        {
            if (!File.Exists(path))
                return new Dictionary<string, FamilyMapEntry>();

            var json = File.ReadAllText(path);
            var opts = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true,
                ReadCommentHandling         = JsonCommentHandling.Skip,
                AllowTrailingCommas         = true,
            };

            var raw = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(json, opts)
                      ?? new();

            var result = new Dictionary<string, FamilyMapEntry>(StringComparer.OrdinalIgnoreCase);
            foreach (var kv in raw)
            {
                // Skip the _comment key
                if (kv.Key.StartsWith('_'))
                    continue;

                var entry = JsonSerializer.Deserialize<FamilyMapEntry>(kv.Value.GetRawText(), opts);
                if (entry != null)
                    result[kv.Key] = entry;
            }
            return result;
        }
        catch
        {
            // Config load failure is non-fatal — all lookups fall back to category
            return new Dictionary<string, FamilyMapEntry>();
        }
    }

    private static string DefaultMapPath()
    {
        var addinDir = Path.GetDirectoryName(typeof(FamilyResolver).Assembly.Location)
                       ?? AppDomain.CurrentDomain.BaseDirectory;
        return Path.Combine(addinDir, "family_map.json");
    }
}

// ── Config DTOs ───────────────────────────────────────────────────────────────

public class FamilyMapEntry
{
    [JsonPropertyName("category")]     public string?         Category    { get; set; }
    [JsonPropertyName("family")]       public string?         Family      { get; set; }
    [JsonPropertyName("default_type")] public string?         DefaultType { get; set; }
    [JsonPropertyName("type_rules")]   public List<TypeRule>  TypeRules   { get; set; } = new();
}

public class TypeRule
{
    [JsonPropertyName("type")]            public string  Type            { get; set; } = "";
    [JsonPropertyName("max_diameter_mm")] public double? MaxDiameterMm   { get; set; }
    [JsonPropertyName("max_height_mm")]   public double? MaxHeightMm     { get; set; }
    [JsonPropertyName("max_major_mm")]    public double? MaxMajorMm      { get; set; }
}
