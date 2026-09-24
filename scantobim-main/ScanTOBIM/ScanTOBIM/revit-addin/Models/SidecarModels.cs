// ─────────────────────────────────────────────────────────────────────────────
// SidecarModels.cs — DTOs that mirror the results.json written by
//                    stb-processor.exe (processor_cli.py).
//
// Schema version: 1.0
// These classes are intentionally flat / nullable-tolerant so that older
// sidecar builds remain deserializable as the schema evolves.
// ─────────────────────────────────────────────────────────────────────────────

using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace ScanToBIM.Models;

// ── Top-level results.json ────────────────────────────────────────────────────

public sealed class SidecarResults
{
    [JsonPropertyName("schema_version")]
    public string SchemaVersion { get; init; } = "1.0";

    [JsonPropertyName("metadata")]
    public SidecarMetadata Metadata { get; init; } = new();

    [JsonPropertyName("segments")]
    public List<SidecarSegment> Segments { get; init; } = new();

    [JsonPropertyName("warnings")]
    public List<string> Warnings { get; init; } = new();
}

// ── Metadata block ────────────────────────────────────────────────────────────

public sealed class SidecarMetadata
{
    [JsonPropertyName("processor_version")]
    public string? ProcessorVersion { get; init; }

    [JsonPropertyName("processed_at")]
    public string? ProcessedAt { get; init; }

    [JsonPropertyName("input_file")]
    public string? InputFile { get; init; }

    [JsonPropertyName("input_path")]
    public string? InputPath { get; init; }

    [JsonPropertyName("file_size_bytes")]
    public long FileSizeBytes { get; init; }

    [JsonPropertyName("file_format")]
    public string? FileFormat { get; init; }

    [JsonPropertyName("point_count_raw")]
    public long? PointCountRaw { get; init; }

    [JsonPropertyName("zone_id")]
    public string? ZoneId { get; init; }

    [JsonPropertyName("voxel_size_m")]
    public double VoxelSizeM { get; init; }

    [JsonPropertyName("segment_count")]
    public int SegmentCount { get; init; }

    [JsonPropertyName("processing_time_s")]
    public double ProcessingTimeS { get; init; }

    [JsonPropertyName("coord_origin_x_mm")]
    public double? CoordOriginXMm { get; init; }

    [JsonPropertyName("coord_origin_y_mm")]
    public double? CoordOriginYMm { get; init; }

    [JsonPropertyName("coord_origin_z_mm")]
    public double? CoordOriginZMm { get; init; }

    [JsonPropertyName("scan_z_min_mm")]
    public double? ScanZMinMm { get; init; }

    [JsonPropertyName("scan_z_max_mm")]
    public double? ScanZMaxMm { get; init; }
}

// ── Per-segment ───────────────────────────────────────────────────────────────

public sealed class SidecarSegment
{
    [JsonPropertyName("segment_id")]
    public string SegmentId { get; init; } = "";

    [JsonPropertyName("zone_id")]
    public string ZoneId { get; init; } = "zone-001";

    /// <summary>
    /// SegmentShape value from Python: "plane_vertical", "plane_horizontal",
    /// "plane_sloped", "cylinder", "box", "sphere", "valve_candidate", etc.
    /// </summary>
    [JsonPropertyName("shape")]
    public string Shape { get; init; } = "";

    [JsonPropertyName("normal")]
    public SidecarPoint3D? Normal { get; init; }

    [JsonPropertyName("centroid")]
    public SidecarPoint3D Centroid { get; set; } = new();

    [JsonPropertyName("bounding_box")]
    public SidecarBoundingBox BoundingBox { get; set; } = new();

    [JsonPropertyName("point_count")]
    public int PointCount { get; init; }

    /// <summary>Detection confidence 0–1.</summary>
    [JsonPropertyName("confidence")]
    public double Confidence { get; init; }

    [JsonPropertyName("scan_station_id")]
    public string? ScanStationId { get; init; }

    [JsonPropertyName("source_file")]
    public string? SourceFile { get; init; }

    /// <summary>Extra metadata tags e.g. {"stair_steps": 4}.</summary>
    [JsonPropertyName("tags")]
    public Dictionary<string, object> Tags { get; set; } = new();

    [JsonPropertyName("element_type")]
    public string? ElementType { get; set; }

    [JsonPropertyName("safety_category")]
    public string? SafetyCategory { get; set; }


    // ── Derived helpers ───────────────────────────────────────────────────────

    /// <summary>Width of the bounding box in millimetres (X axis).</summary>
    public double WidthMm  => BoundingBox.MaxX - BoundingBox.MinX;

    /// <summary>Depth of the bounding box in millimetres (Y axis).</summary>
    public double DepthMm  => BoundingBox.MaxY - BoundingBox.MinY;

    /// <summary>Height of the bounding box in millimetres (Z axis).</summary>
    public double HeightMm => BoundingBox.MaxZ - BoundingBox.MinZ;

    public bool IsVerticalPlane   => Shape == "plane_vertical";
    public bool IsHorizontalPlane => Shape == "plane_horizontal";
    public bool IsCylinder        => Shape == "cylinder";
    public bool IsValveCandidate  => Shape == "valve_candidate";

    // ── USIBD LOA v3.1 helpers ────────────────────────────────────────────────

    /// <summary>
    /// USIBD LOA σ (mm) — standard deviation of point-to-fit residual.
    /// Pulled from Tags["loa_sigma_mm"] written by the Python segmentation
    /// pipeline. Returns 0 if missing (caller should treat as "unknown").
    /// </summary>
    public double LoaSigmaMm => GetTagDouble("loa_sigma_mm") ?? 0.0;

    /// <summary>
    /// USIBD LOA tier string (LOA10–LOA50) from Tags["loa_tier"].
    /// Returns "LOA10" if missing — worst tier assumed until proven otherwise.
    /// </summary>
    public string LoaTier
    {
        get
        {
            if (Tags != null && Tags.TryGetValue("loa_tier", out var v))
            {
                return v switch
                {
                    string s                      => s,
                    System.Text.Json.JsonElement e
                        when e.ValueKind == System.Text.Json.JsonValueKind.String
                        => e.GetString() ?? "LOA10",
                    _ => "LOA10",
                };
            }
            return "LOA10";
        }
    }

    // ── Semantic Wall & Storey Helpers (Sprint P0.4) ───────────────────────────

    /// <summary>
    /// PCA-derived or measured wall thickness in millimetres.
    /// Falls back to minimum bounding-box horizontal dimension, clamped to [100, 500] mm.
    /// </summary>
    public double WallThicknessMm
    {
        get
        {
            var tagged = GetTagDouble("wall_thickness_mm");
            if (tagged.HasValue && tagged.Value > 0.0)
                return Math.Min(Math.Max(tagged.Value, 100.0), 500.0);

            var fallback = Math.Min(WidthMm, DepthMm);
            return Math.Min(Math.Max(fallback, 100.0), 500.0);
        }
    }

    /// <summary>
    /// Storey base elevation in millimetres.
    /// Prefers reconstructed storey base tag, then source min Z tag, then bounding box min Z.
    /// </summary>
    public double StoreyBaseMm =>
        GetTagDouble("reconstructed_storey_base_mm")
        ?? GetTagDouble("source_min_z_mm")
        ?? BoundingBox.MinZ;

    /// <summary>
    /// Storey top elevation in millimetres.
    /// Prefers reconstructed storey top tag, then source max Z tag, then bounding box max Z.
    /// </summary>
    public double StoreyTopMm =>
        GetTagDouble("reconstructed_storey_top_mm")
        ?? GetTagDouble("source_max_z_mm")
        ?? BoundingBox.MaxZ;

    /// <summary>
    /// Storey height in millimetres, clamped to reasonable architectural single-storey range [500, 6000] mm.
    /// </summary>
    public double StoreyHeightMm
    {
        get
        {
            var h = StoreyTopMm - StoreyBaseMm;
            if (h <= 500.0)
                return 3000.0; // standard default storey height (3.0 m)
            if (h > 6000.0)
                return 3500.0; // clamp accidental multi-storey spans
            return h;
        }
    }

    /// <summary>
    /// Attempts to read explicit wall 2D centerline endpoints from tags.
    /// Handles negative coordinates correctly.
    /// </summary>
    public bool TryGetWallEndpoints(out double startX, out double startY, out double endX, out double endY)
    {
        startX = startY = endX = endY = 0.0;
        var sx = GetTagDouble("wall_start_x_mm");
        var sy = GetTagDouble("wall_start_y_mm");
        var ex = GetTagDouble("wall_end_x_mm");
        var ey = GetTagDouble("wall_end_y_mm");

        if (sx.HasValue && sy.HasValue && ex.HasValue && ey.HasValue)
        {
            startX = sx.Value;
            startY = sy.Value;
            endX = ex.Value;
            endY = ey.Value;
            return true;
        }

        return false;
    }

    public double? GetTagDouble(string key)
    {
        if (Tags != null && Tags.TryGetValue(key, out var v) && v != null)
        {
            return v switch
            {
                double d => d,
                float f => (double)f,
                int i => (double)i,
                long l => (double)l,
                decimal m => (double)m,
                System.Text.Json.JsonElement e when e.ValueKind == System.Text.Json.JsonValueKind.Number && e.TryGetDouble(out var d) => d,
                System.Text.Json.JsonElement e when e.ValueKind == System.Text.Json.JsonValueKind.String && double.TryParse(e.GetString(), System.Globalization.NumberStyles.Float | System.Globalization.NumberStyles.AllowThousands, System.Globalization.CultureInfo.InvariantCulture, out var parsed) => parsed,
                _ => double.TryParse(v.ToString(), System.Globalization.NumberStyles.Float | System.Globalization.NumberStyles.AllowThousands, System.Globalization.CultureInfo.InvariantCulture, out var parsedStr) ? parsedStr : null,
            };
        }
        return null;
    }
}

// ── Supporting geometry types ─────────────────────────────────────────────────

[JsonConverter(typeof(SidecarPoint3DConverter))]
public sealed class SidecarPoint3D
{
    [JsonPropertyName("x")] public double X { get; set; }
    [JsonPropertyName("y")] public double Y { get; set; }
    [JsonPropertyName("z")] public double Z { get; set; }
}

public sealed class SidecarPoint3DConverter : JsonConverter<SidecarPoint3D>
{
    public override SidecarPoint3D Read(
        ref System.Text.Json.Utf8JsonReader reader,
        Type typeToConvert,
        System.Text.Json.JsonSerializerOptions options)
    {
        using var document = System.Text.Json.JsonDocument.ParseValue(ref reader);
        var point = document.RootElement;
        if (point.ValueKind == System.Text.Json.JsonValueKind.Array && point.GetArrayLength() == 3)
        {
            return new SidecarPoint3D
            {
                X = ReadCoordinate(point[0]),
                Y = ReadCoordinate(point[1]),
                Z = ReadCoordinate(point[2]),
            };
        }

        if (point.ValueKind == System.Text.Json.JsonValueKind.Object &&
            point.TryGetProperty("x", out var coordinateX) &&
            point.TryGetProperty("y", out var coordinateY) &&
            point.TryGetProperty("z", out var coordinateZ))
        {
            return new SidecarPoint3D
            {
                X = ReadCoordinate(coordinateX),
                Y = ReadCoordinate(coordinateY),
                Z = ReadCoordinate(coordinateZ),
            };
        }

        throw new System.Text.Json.JsonException("Expected a point as [x, y, z] or an object with x, y, z coordinates.");
    }

    private static double ReadCoordinate(System.Text.Json.JsonElement coordinate)
    {
        if (coordinate.ValueKind != System.Text.Json.JsonValueKind.Number ||
            !coordinate.TryGetDouble(out var value) || !double.IsFinite(value))
            throw new System.Text.Json.JsonException("Point coordinates must be finite numbers.");
        return value;
    }

    public override void Write(
        System.Text.Json.Utf8JsonWriter writer,
        SidecarPoint3D value,
        System.Text.Json.JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        writer.WriteNumber("x", value.X);
        writer.WriteNumber("y", value.Y);
        writer.WriteNumber("z", value.Z);
        writer.WriteEndObject();
    }
}

public sealed class SidecarBoundingBox
{
    [JsonPropertyName("min_x")] public double MinX { get; set; }
    [JsonPropertyName("max_x")] public double MaxX { get; set; }
    [JsonPropertyName("min_y")] public double MinY { get; set; }
    [JsonPropertyName("max_y")] public double MaxY { get; set; }
    [JsonPropertyName("min_z")] public double MinZ { get; set; }
    [JsonPropertyName("max_z")] public double MaxZ { get; set; }
}

// ── Session-scoped result store (consumed by Sprint P3 geometry builder) ──────

public static class ScanResultStore
{
    /// <summary>
    /// Holds the most recently processed scan results so that
    /// ScanGeometryBuilder (Sprint P3) can consume them without file I/O.
    /// Set by ProcessScanCommand after sidecar completes successfully.
    /// </summary>
    public static SidecarResults? Latest { get; set; }

    /// <summary>Path of the results.json written by the last sidecar run.</summary>
    public static string? LatestResultsPath { get; set; }
}
