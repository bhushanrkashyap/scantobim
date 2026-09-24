// ─────────────────────────────────────────────────────────────────────────────
// ElementTypeConfigService.cs
//
// Reads family_map.json at runtime and answers two questions for any element
// type string:
//
//   1. GetRevitCategory(elementType)  — which BuiltInCategory to use in Revit
//   2. IsCylindricalGeometry(elementType) — should we render as a cylinder shell?
//
// This replaces ALL hardcoded HashSets and switch/case lists in
// ScanGeometryBuilder.cs that previously required code edits to add new types.
//
// Adding a new element type to family_map.json (with the right "category" and
// optional "geometry_style": "cylindrical") is the ONLY change required.
// ─────────────────────────────────────────────────────────────────────────────

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Autodesk.Revit.DB;

namespace ScanToBIM.Services;

/// <summary>
/// Provides dynamic element-type metadata loaded from family_map.json.
/// All lookups are case-insensitive and degrade gracefully when the config
/// file is absent or the key is not found.
/// </summary>
public static class ElementTypeConfigService
{
    // ── Known Revit category string → BuiltInCategory ─────────────────────────
    // This mapping is the only fixed table in this file; it is a stable
    // Revit API constant list and does not change with project configuration.
    private static readonly IReadOnlyDictionary<string, BuiltInCategory> _categoryLookup =
        new Dictionary<string, BuiltInCategory>(StringComparer.OrdinalIgnoreCase)
        {
            ["OST_MechanicalEquipment"] = BuiltInCategory.OST_MechanicalEquipment,
            ["OST_PipeAccessory"]       = BuiltInCategory.OST_PipeAccessory,
            ["OST_PipeAccessories"]     = BuiltInCategory.OST_PipeAccessory,
            ["OST_PipeCurves"]          = BuiltInCategory.OST_PipeCurves,
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
            ["OST_Walls"]               = BuiltInCategory.OST_Walls,
            ["OST_Floors"]              = BuiltInCategory.OST_Floors,
            ["OST_Roofs"]               = BuiltInCategory.OST_Roofs,
            ["OST_Stairs"]              = BuiltInCategory.OST_Stairs,
            ["OST_DuctCurves"]          = BuiltInCategory.OST_DuctCurves,
            ["OST_Conduit"]             = BuiltInCategory.OST_Conduit,
            ["OST_CableTray"]           = BuiltInCategory.OST_CableTray,
            ["OST_GenericModel"]        = BuiltInCategory.OST_GenericModel,
        };

    // ── Lazy-loaded config data ────────────────────────────────────────────────

    private sealed class ConfigEntry
    {
        public string?  Category      { get; init; }
        public string?  GeometryStyle { get; init; }  // "cylindrical" | "box" | null
    }

    private static readonly Lazy<IReadOnlyDictionary<string, ConfigEntry>> _config =
        new(LoadConfig, System.Threading.LazyThreadSafetyMode.ExecutionAndPublication);

    // ── Public API ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Returns the Revit BuiltInCategory for an element type string (e.g. "mechanical_equipment_receiver").
    /// Returns <see langword="null"/> when the type is not found in family_map.json.
    /// </summary>
    public static BuiltInCategory? GetRevitCategory(string elementType)
    {
        if (string.IsNullOrWhiteSpace(elementType))
            return null;

        if (_config.Value.TryGetValue(elementType, out var entry)
            && !string.IsNullOrWhiteSpace(entry.Category)
            && _categoryLookup.TryGetValue(entry.Category, out var cat))
            return cat;

        return null;
    }

    /// <summary>
    /// Returns <see langword="true"/> if the element type should be rendered as a
    /// closed cylinder shell in Revit (geometry_style == "cylindrical" in family_map.json).
    /// </summary>
    public static bool IsCylindricalGeometry(string elementType)
    {
        if (string.IsNullOrWhiteSpace(elementType))
            return false;

        return _config.Value.TryGetValue(elementType, out var entry)
               && string.Equals(entry.GeometryStyle, "cylindrical", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Returns <see langword="true"/> for primitive shape tokens that should be
    /// re-classified via geometry inference rather than being trusted as-is.
    ///
    /// When the Python agent emits an element_type that IS in family_map.json,
    /// the C# layer should trust it directly (no InferLinearServiceType override).
    /// When the agent emits a raw shape name ("cylinder", "box", "generic", …),
    /// the C# layer should infer the correct service type from geometry/color.
    /// </summary>
    public static bool ShouldInferFromGeometry(string elementTypeUpper)
    {
        // Raw primitive shape tokens emitted when the agent couldn't classify
        if (elementTypeUpper is "CYLINDER" or "BOX" or "PLANE_VERTICAL"
            or "PLANE_HORIZONTAL" or "PLANE_SLOPED" or "VOID"
            or "VALVE_CANDIDATE" or "UNKNOWN" or "GENERIC" or "")
            return true;

        // If the type IS recognised in family_map.json it is a proper semantic
        // label — trust the agent's classification, do not override with PIPE.
        var lower = elementTypeUpper.ToLowerInvariant();
        return !_config.Value.ContainsKey(lower);
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    private static IReadOnlyDictionary<string, ConfigEntry> LoadConfig()
    {
        try
        {
            var path = FamilyMapPath();
            if (!File.Exists(path))
                return new Dictionary<string, ConfigEntry>(StringComparer.OrdinalIgnoreCase);

            var json = File.ReadAllText(path);
            var opts = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true,
                ReadCommentHandling         = JsonCommentHandling.Skip,
                AllowTrailingCommas         = true,
            };

            var raw = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(json, opts)
                      ?? new Dictionary<string, JsonElement>();

            var result = new Dictionary<string, ConfigEntry>(StringComparer.OrdinalIgnoreCase);
            foreach (var kv in raw)
            {
                if (kv.Key.StartsWith('_'))
                    continue;   // skip comment keys

                string? category = null;
                string? style    = null;

                if (kv.Value.TryGetProperty("category", out var catEl))
                    category = catEl.GetString();

                if (kv.Value.TryGetProperty("geometry_style", out var styleEl))
                    style = styleEl.GetString();

                result[kv.Key] = new ConfigEntry { Category = category, GeometryStyle = style };
            }

            return result;
        }
        catch
        {
            // Config load failure is non-fatal — callers handle null/false gracefully
            return new Dictionary<string, ConfigEntry>(StringComparer.OrdinalIgnoreCase);
        }
    }

    private static string FamilyMapPath()
    {
        var addinDir = Path.GetDirectoryName(typeof(ElementTypeConfigService).Assembly.Location)
                       ?? AppDomain.CurrentDomain.BaseDirectory;
        return Path.Combine(addinDir, "family_map.json");
    }
}
