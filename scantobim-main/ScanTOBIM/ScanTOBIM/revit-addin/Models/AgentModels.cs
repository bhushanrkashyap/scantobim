// ─────────────────────────────────────────────────────────────────────────────
// AgentModels.cs
// C# DTOs that mirror the Python Pydantic models in agent/models.py.
// JSON property names use snake_case to match the Python API response format.
// ─────────────────────────────────────────────────────────────────────────────

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;
using Autodesk.Revit.DB;

namespace ScanToBIM.Models;

// ── Enumerations ─────────────────────────────────────────────────────────────

/// <summary>
/// Safety category per NQA-1 nuclear quality assurance standard.
/// SC1 = Safety-Critical (hard block — MUST NEVER be auto-created).
/// SC2 = Safety-Related  (requires engineer approval signature).
/// SC3 = Safety-Relevant (flagged but auto-proceed).
/// NS  = Non-Safety      (standard, no restrictions).
/// </summary>
public enum SafetyCategory { SC1, SC2, SC3, NS }

[JsonConverter(typeof(JsonStringEnumConverter))]
public enum ElementType
{
    // Structural
    wall, floor, ceiling, column, beam, stair,
    // Structural — Nuclear / Civil
    ladder, grating, overhead_crane, seismic_isolator, containment_penetration,
    // Architectural / Access
    door, window, railing, ramp, hatch,
    // Civil
    trench, bund_wall, kerb, slab_opening,
    // MEP — Piping / Inline
    pipe, conduit, duct, cable_tray, valve,
    safety_relief_valve, strainer, expansion_joint, fire_damper, penetration_seal, pipe_support,
    // MEP — Vessels / Equipment
    tank, pump, hvac_equipment, sprinkler, drainage,
    fire_extinguisher,
    pressure_vessel, heat_exchanger, compressor,
    mechanical_equipment_receiver,
    // Nuclear primary-circuit vessels (SC1)
    pressurizer, steam_generator, emergency_diesel_generator,
    // Nuclear monitoring
    radiation_monitor,
    // Electrical
    electrical_panel, transformer, switchgear, ups_system, junction_box, lighting_fitting,
    // Fire / Life Safety
    fire_hydrant, deluge_valve, fire_alarm_panel, smoke_detector,
    // Electrical containment / wire
    wire,
}

[JsonConverter(typeof(JsonStringEnumConverter))]
public enum Discipline { structural, architectural, mep, fire_protection, civil, infrastructure }

// ── Geometry ─────────────────────────────────────────────────────────────────

/// <summary>Axis-aligned bounding box. All values in millimetres.</summary>
public class BoundingBox
{
    [JsonPropertyName("min_x")] public double MinX { get; set; }
    [JsonPropertyName("min_y")] public double MinY { get; set; }
    [JsonPropertyName("min_z")] public double MinZ { get; set; }
    [JsonPropertyName("max_x")] public double MaxX { get; set; }
    [JsonPropertyName("max_y")] public double MaxY { get; set; }
    [JsonPropertyName("max_z")] public double MaxZ { get; set; }

    public double WidthMm  => Math.Abs(MaxX - MinX);
    public double DepthMm  => Math.Abs(MaxY - MinY);
    public double HeightMm => Math.Abs(MaxZ - MinZ);
}

/// <summary>3D point. All values in millimetres.</summary>
public class Point3D
{
    [JsonPropertyName("x")] public double X { get; set; }
    [JsonPropertyName("y")] public double Y { get; set; }
    [JsonPropertyName("z")] public double Z { get; set; }
}

// ── Core Instruction ─────────────────────────────────────────────────────────

/// <summary>
/// A single instruction from the AI agent to create one BIM element.
/// Mirrors Python ElementInstruction Pydantic model exactly.
/// </summary>
public class ElementInstruction
{
    [JsonPropertyName("instruction_id")]   public string InstructionId   { get; set; } = "";
    [JsonPropertyName("segment_id")]       public string SegmentId       { get; set; } = "";
    [JsonPropertyName("zone_id")]          public string ZoneId          { get; set; } = "";
    [JsonPropertyName("element_type")]     public ElementType ElementType { get; set; }
    [JsonPropertyName("discipline")]       public Discipline  Discipline  { get; set; }
    [JsonPropertyName("safety_category")]  public SafetyCategory SafetyCategory { get; set; }
    [JsonPropertyName("dsear_zone")]       public string? DsearZone      { get; set; }
    [JsonPropertyName("bounding_box")]     public BoundingBox BoundingBox { get; set; } = new();
    [JsonPropertyName("centroid")]         public Point3D     Centroid    { get; set; } = new();
    [JsonPropertyName("parameters")]       public Dictionary<string, object>? Parameters { get; set; }

    // Revit family resolution — operator overrides.
    // When set, FamilyResolver uses these before consulting family_map.json.
    [JsonPropertyName("revit_family_hint")] public string? RevitFamilyHint { get; set; }
    [JsonPropertyName("revit_type_hint")]   public string? RevitTypeHint   { get; set; }

    // SC2 approval fields — must be present for SC2 elements
    [JsonPropertyName("approval_signature")] public string?   ApprovalSignature { get; set; }
    [JsonPropertyName("approver_upn")]       public string?   ApproverUpn       { get; set; }
    [JsonPropertyName("approved_at_utc")]    public DateTime? ApprovedAtUtc     { get; set; }

    // P4: additional compliance fields (dsear_zone already declared above)
    [JsonPropertyName("ncr_ref")]            public string?   NcrRef            { get; set; }
    [JsonPropertyName("compliance_status")]  public string?   ComplianceStatus  { get; set; }
}

// ── Results ───────────────────────────────────────────────────────────────────

/// <summary>Result returned to the Python agent after processing an instruction.</summary>
public class ActionResult
{
    [JsonPropertyName("success")]        public bool   Success       { get; set; }
    [JsonPropertyName("element_id")]     public string? ElementId    { get; set; }
    [JsonPropertyName("instruction_id")] public string InstructionId { get; set; } = "";
    [JsonPropertyName("error")]          public string? Error        { get; set; }
    [JsonPropertyName("duration_ms")]    public long    DurationMs   { get; set; }
}

// ── Bridge Health ─────────────────────────────────────────────────────────────

public class BridgeHealthResponse
{
    [JsonPropertyName("connected")]        public bool Connected       { get; set; } = true;
    [JsonPropertyName("queue_depth")]      public int  QueueDepth      { get; set; }
    [JsonPropertyName("elements_created")] public int  ElementsCreated { get; set; }
    [JsonPropertyName("elements_blocked")] public int  ElementsBlocked { get; set; }
    [JsonPropertyName("version")]          public string Version       { get; set; } = "pov-1.0";
}

// ── Wall geometry resolution ──────────────────────────────────────────────────
// Extracts the correct wall centreline (start XYZ, end XYZ, thickness) from
// the ElementInstruction.Parameters dict populated by the Python agent.
//
// Resolution order:
//   Strategy 1 — explicit start/end from RANSAC axis (wall_start_x_mm etc.)
//   Strategy 2 — axis vector + length + centroid (origin-independent)
//   null        — insufficient data; caller falls back to AABB longest-edge
//
// All input values in the dict are in millimetres; outputs are in Revit feet.

public static class WallGeometryHelper
{
    /// <summary>
    /// Attempts to resolve wall centreline geometry from agent parameters.
    /// Returns (start, end, thicknessMm) in Revit feet, or null on failure.
    /// </summary>
    public static (XYZ Start, XYZ End, double ThicknessMm)?
        TryExtract(ElementInstruction instruction, double baseElevationFt)
    {
        var p = instruction.Parameters;
        if (p is null) return null;

        double thicknessMm = TryGetDouble(p, "wall_thickness_mm") ?? 200.0;

        // ── Strategy 1: explicit RANSAC start/end coordinates ─────────────
        double? sx = TryGetDouble(p, "wall_start_x_mm");
        double? sy = TryGetDouble(p, "wall_start_y_mm");
        double? ex = TryGetDouble(p, "wall_end_x_mm");
        double? ey = TryGetDouble(p, "wall_end_y_mm");

        if (sx.HasValue && sy.HasValue && ex.HasValue && ey.HasValue)
        {
            var start = new XYZ(MmToFt(sx.Value), MmToFt(sy.Value), baseElevationFt);
            var end   = new XYZ(MmToFt(ex.Value), MmToFt(ey.Value), baseElevationFt);
            if (start.DistanceTo(end) >= MmToFt(50))  // require ≥ 50 mm wall length
                return (start, end, thicknessMm);
        }

        // ── Strategy 2: axis vector + centroid + length ───────────────────
        double? axisX  = TryGetDouble(p, "wall_axis_x");
        double? axisY  = TryGetDouble(p, "wall_axis_y");
        double? lenMm  = TryGetDouble(p, "wall_length_mm");

        if (axisX.HasValue && axisY.HasValue && lenMm.HasValue)
        {
            double axLen = Math.Sqrt(axisX.Value * axisX.Value + axisY.Value * axisY.Value);
            if (axLen < 1e-6) return null;

            double unitX = axisX.Value / axLen;
            double unitY = axisY.Value / axLen;
            double halfFt = MmToFt(lenMm.Value) / 2.0;

            double cx = MmToFt(instruction.Centroid.X);
            double cy = MmToFt(instruction.Centroid.Y);

            var start = new XYZ(cx - unitX * halfFt, cy - unitY * halfFt, baseElevationFt);
            var end   = new XYZ(cx + unitX * halfFt, cy + unitY * halfFt, baseElevationFt);

            if (start.DistanceTo(end) >= MmToFt(50))
                return (start, end, thicknessMm);
        }

        // ── Strategy 3: recompute axis from normal vector if present ──────
        double? nx = TryGetDouble(p, "normal_x");
        double? ny = TryGetDouble(p, "normal_y");
        double? nz = TryGetDouble(p, "normal_z");

        if (nx.HasValue && ny.HasValue && nz.HasValue)
        {
            // wall_axis = cross(normal, Z_up) then normalised in XY
            double wax = -ny.Value;  // cross((nx,ny,nz),(0,0,1)) = (ny*1-nz*0, nz*0-nx*1, nx*0-ny*0) = (ny,-nx,0)
            double way =  nx.Value;
            double waLen = Math.Sqrt(wax * wax + way * way);
            if (waLen < 1e-6) return null;

            wax /= waLen; way /= waLen;

            // Estimate length from AABB: use longer of width/depth
            var bb = instruction.BoundingBox;
            double estLenMm = Math.Max(bb.WidthMm, bb.DepthMm);
            double halfFt   = MmToFt(estLenMm) / 2.0;

            double cx = MmToFt(instruction.Centroid.X);
            double cy = MmToFt(instruction.Centroid.Y);

            var start = new XYZ(cx - wax * halfFt, cy - way * halfFt, baseElevationFt);
            var end   = new XYZ(cx + wax * halfFt, cy + way * halfFt, baseElevationFt);

            if (start.DistanceTo(end) >= MmToFt(50))
                return (start, end, thicknessMm);
        }

        return null;  // no axis data — caller uses AABB fallback
    }

    private static double? TryGetDouble(Dictionary<string, object> p, string key)
    {
        if (!p.TryGetValue(key, out var raw)) return null;
        if (raw is System.Text.Json.JsonElement je)
        {
            if (je.ValueKind == System.Text.Json.JsonValueKind.Number)
                return je.GetDouble();
            return null;
        }
        try { return Convert.ToDouble(raw); } catch { return null; }
    }

    private static double MmToFt(double mm) => mm / 304.8;
}

// ── Floor geometry resolution ─────────────────────────────────────────────────
public static class FloorGeometryHelper
{
    /// <summary>
    /// Attempts to resolve floor boundary from agent OBB parameters.
    /// Returns 4 corner points in Revit feet, or null on failure.
    /// </summary>
    public static (XYZ P1, XYZ P2, XYZ P3, XYZ P4)? TryExtract(ElementInstruction instruction, double baseElevationFt)
    {
        var p = instruction.Parameters;
        if (p is null) return null;

        double? axisX = TryGetDouble(p, "floor_axis_x");
        double? axisY = TryGetDouble(p, "floor_axis_y");
        double? lenMm = TryGetDouble(p, "floor_length_mm");
        double? widMm = TryGetDouble(p, "floor_width_mm");
        double? cxMm  = TryGetDouble(p, "floor_center_x_mm");
        double? cyMm  = TryGetDouble(p, "floor_center_y_mm");

        if (axisX.HasValue && axisY.HasValue && lenMm.HasValue && widMm.HasValue)
        {
            double cx = cxMm.HasValue ? MmToFt(cxMm.Value) : MmToFt(instruction.Centroid.X);
            double cy = cyMm.HasValue ? MmToFt(cyMm.Value) : MmToFt(instruction.Centroid.Y);

            double axLen = Math.Sqrt(axisX.Value * axisX.Value + axisY.Value * axisY.Value);
            if (axLen < 1e-6) return null;

            double unitX = axisX.Value / axLen;
            double unitY = axisY.Value / axLen;
            
            double halfLenFt = MmToFt(lenMm.Value) / 2.0;
            double halfWidFt = MmToFt(widMm.Value) / 2.0;
            
            var center = new XYZ(cx, cy, baseElevationFt);
            var xVec = new XYZ(unitX, unitY, 0) * halfLenFt;
            var yVec = new XYZ(-unitY, unitX, 0) * halfWidFt;

            return (
                center - xVec - yVec,
                center + xVec - yVec,
                center + xVec + yVec,
                center - xVec + yVec
            );
        }

        return null;
    }

    private static double? TryGetDouble(Dictionary<string, object> p, string key)
    {
        if (!p.TryGetValue(key, out var raw)) return null;
        if (raw is System.Text.Json.JsonElement je)
        {
            if (je.ValueKind == System.Text.Json.JsonValueKind.Number)
                return je.GetDouble();
            return null;
        }
        try { return Convert.ToDouble(raw); } catch { return null; }
    }

    private static double MmToFt(double mm) => mm / 304.8;
}

// ── Cylinder geometry resolution ──────────────────────────────────────────────
public static class CylinderGeometryHelper
{
    /// <summary>
    /// Attempts to resolve cylinder start/end points and radius from agent parameters.
    /// Returns Start, End, and Radius in Revit feet, or null on failure.
    /// </summary>
    public static (XYZ Start, XYZ End, double RadiusFt)? TryExtract(ElementInstruction instruction)
    {
        var p = instruction.Parameters;
        if (p is null) return null;

        double? sx = TryGetDouble(p, "cyl_start_x_mm");
        double? sy = TryGetDouble(p, "cyl_start_y_mm");
        double? sz = TryGetDouble(p, "cyl_start_z_mm");
        double? ex = TryGetDouble(p, "cyl_end_x_mm");
        double? ey = TryGetDouble(p, "cyl_end_y_mm");
        double? ez = TryGetDouble(p, "cyl_end_z_mm");
        double? radMm = TryGetDouble(p, "fitted_radius_mm");

        if (!radMm.HasValue) return null;

        if (sx.HasValue && sy.HasValue && sz.HasValue && ex.HasValue && ey.HasValue && ez.HasValue)
        {
            var start = new XYZ(MmToFt(sx.Value), MmToFt(sy.Value), MmToFt(sz.Value));
            var end = new XYZ(MmToFt(ex.Value), MmToFt(ey.Value), MmToFt(ez.Value));
            if (start.DistanceTo(end) >= 0.01)
                return (start, end, MmToFt(radMm.Value));
        }

        double? axisX = TryGetDouble(p, "cyl_axis_x");
        double? axisY = TryGetDouble(p, "cyl_axis_y");
        double? axisZ = TryGetDouble(p, "cyl_axis_z");

        if (axisX.HasValue && axisY.HasValue && axisZ.HasValue)
        {
            var c = instruction.Centroid;
            double cx = MmToFt(c.X);
            double cy = MmToFt(c.Y);
            double cz = MmToFt(c.Z);

            var bb = instruction.BoundingBox;
            double dx = bb.MaxX - bb.MinX;
            double dy = bb.MaxY - bb.MinY;
            double dz = bb.MaxZ - bb.MinZ;
            double estLenMm = Math.Max(dx, Math.Max(dy, dz));
            double halfFt = MmToFt(estLenMm) / 2.0;

            var axis = new XYZ(axisX.Value, axisY.Value, axisZ.Value).Normalize();
            var center = new XYZ(cx, cy, cz);
            var start = center - axis * halfFt;
            var end = center + axis * halfFt;

            if (start.DistanceTo(end) < 0.01) return null;

            return (start, end, MmToFt(radMm.Value));
        }

        return null;
    }

    private static double? TryGetDouble(Dictionary<string, object> p, string key)
    {
        if (!p.TryGetValue(key, out var raw)) return null;
        if (raw is System.Text.Json.JsonElement je)
        {
            if (je.ValueKind == System.Text.Json.JsonValueKind.Number)
                return je.GetDouble();
            return null;
        }
        try { return Convert.ToDouble(raw); } catch { return null; }
    }

    private static double MmToFt(double mm) => mm / 304.8;
}
