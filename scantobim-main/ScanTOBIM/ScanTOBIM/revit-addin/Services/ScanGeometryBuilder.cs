// ─────────────────────────────────────────────────────────────────────────────
// ScanGeometryBuilder.cs — Sprint P3
//
// Converts SidecarSegment objects (from stb-processor.exe) into Revit
// DirectShape elements so the detected geometry is visible in the 3D view.
//
// Per segment:
//   • Box tessellation for planes, boxes, openings, slabs
//   • 24-sided prism for cylinders, valves
//   • DirectShape.ApplicationDataId = segment_id (traceability)
//   • ScanToBIM_SegmentId shared parameter written (if loaded)
//   • Color override on active view by confidence level:
//       ≥ 0.85 → green   (high confidence)
//       ≥ 0.60 → amber   (medium)
//       <  0.60 → red    (low confidence / review needed)
//
// All coordinates: mm → Revit feet (÷ 304.8).
// Must be called inside an open Revit Transaction.
// ─────────────────────────────────────────────────────────────────────────────

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text.Json;
using Autodesk.Revit.DB;
using Autodesk.Revit.DB.Electrical;
using Autodesk.Revit.DB.Mechanical;
using Autodesk.Revit.DB.Plumbing;
using ScanToBIM.Models;
using ScanToBIM.SafetyGate;

namespace ScanToBIM.Services;

public sealed class ScanGeometryBuilder
{
    // ── Constants ─────────────────────────────────────────────────────────────
    private const double MmToFeet      = 1.0 / 304.8;
    private const double MinDimFeet    = 0.03;   // ~3 mm — skip degenerate shapes
    private const int    CylinderSides = 16;

    // Inset margin applied to each BB face to reduce visual overlap between
    // adjacent AABB boxes that share a wall/floor boundary. Reduced from 12mm
    // to 6mm to increase visual separation and depth perception.
    private const double InsetMm       = 6.0;
    private const double InsetFeet     = InsetMm * MmToFeet;

    // Preserve the scan-derived extents directly so the generated geometry
    // matches the source point-cloud dimensions instead of applying visual-only
    // scaling that distorts the reconstruction.
    private const double VisualScaleXY = 1.00;
    private const double VisualScaleZ  = 1.00;
    private const double CylinderRadiusVisualBoost = 1.00;
    private const double PipeJoinExtensionMm = 45.0;

    // Category used for all DirectShapes (safe, works with arbitrary solids)
    private static readonly ElementId GenericModelCatId =
        new ElementId(BuiltInCategory.OST_GenericModel);

    private static double _maxCylDiameter = 0.0;
    private static readonly HashSet<string> _selectedTankIds = new HashSet<string>(StringComparer.Ordinal);

    // Element types that should be represented as enclosed mechanical shells
    // in sidecar visualization instead of solid cuboids.
    private static readonly HashSet<string> EnclosedMechanicalTypes =
        new(StringComparer.OrdinalIgnoreCase)
        {
            "tank",
            "pump",
            "hvac_equipment",
            "pressure_vessel",
            "heat_exchanger",
            "compressor",
            "pressurizer",
            "steam_generator",
            "emergency_diesel_generator",
            "transformer",
            "switchgear",
            "ups_system",
        };

    // Confidence → view override colours
    private static readonly Color ColourHigh   = new Color(27,  174, 96);   // green
    private static readonly Color ColourMedium = new Color(245, 158, 11);   // amber
    private static readonly Color ColourLow    = new Color(220, 38,  38);   // red

    // Native Wall.Create can fail for scan-derived geometry and poison the
    // transaction at commit time. Keep this disabled for robust builds.
    private const bool EnableNativeWalls = true;
    private const bool EnableUnifiedBackWall = false;
    private const bool EnableUnifiedCableTrays = false;
    // TASK-12: walls below 300 mm are micro-segments that poison Revit joins
    // ("Can't keep elements joined" / "Highlighted walls overlap").
    private const double MinWallLengthMm = 300.0;
    private const double MinWallHeightMm = 300.0;

    // ── Public API ────────────────────────────────────────────────────────────

    public sealed class BuildResult
    {
        public int          Created  { get; init; }
        public int          Skipped  { get; init; }
        public List<string> Errors   { get; init; } = new();
        public List<ElementId> ElementIds { get; init; } = new();
    }

    private sealed class MergedSegmentGroup
    {
        public string GroupId { get; set; } = string.Empty;
        public SidecarSegment Representative { get; set; } = null!;
        public SidecarBoundingBox BoundingBox { get; set; } = new();
        public List<string> SegmentIds { get; set; } = new();
    }

    /// <summary>
    /// Creates joined wall and roof elements for grouped segments, then falls back
    /// to the per-segment creation flow for the remaining geometry.
    /// Must be called inside an open Transaction.
    /// </summary>
    private static string GetElementType(SidecarSegment seg)
    {
        var directType = NormalizeSemanticLabel(seg.ElementType);
        if (!string.IsNullOrWhiteSpace(directType))
            return directType;

        if (seg.Tags != null)
        {
            if (seg.Tags.TryGetValue("semantic_label", out var labelObj))
            {
                var label = labelObj?.ToString();
                var normalized = NormalizeSemanticLabel(label);
                if (!string.IsNullOrWhiteSpace(normalized))
                    return normalized;
            }

            if (seg.Tags.TryGetValue("element_type", out var elementTypeObj))
            {
                var label = elementTypeObj?.ToString();
                var normalized = NormalizeSemanticLabel(label);
                if (!string.IsNullOrWhiteSpace(normalized))
                    return normalized;
            }
        }

        if (seg.IsCylinder || seg.IsValveCandidate)
            return "PIPE";

        if (LooksLikePipeLikeInternalComponent(seg))
            return "PIPE";

        // Fallback to shape
        return seg.Shape.ToUpperInvariant();
    }

    private static string? NormalizeSemanticLabel(string? label)
    {
        if (string.IsNullOrWhiteSpace(label))
            return null;

        var normalized = label.Trim().ToLowerInvariant();
        normalized = System.Text.RegularExpressions.Regex.Replace(normalized, @"[^a-z0-9]+", "_").Trim('_');

        return normalized switch
        {
            "wall" or "bund_wall" => "WALL",
            "floor" => "FLOOR",
            "ceiling" => "CEILING",
            "roof" => "ROOF",
            "door" => "DOOR",
            "window" => "WINDOW",
            "stair" or "stairs" or "stair_run" => "STAIR",
            "ramp" => "RAMP",
            "ladder" => "LADDER",
            "pipe" => "PIPE",
            "valve_candidate" => "VALVE",
            "cylinder" => "PIPE",
            "duct" => "DUCT",
            "conduit" => "CONDUIT",
            "cable_tray" or "cabletray" => "CABLE_TRAY",
            "pipe_support" => "PIPE_SUPPORT",
            "drainage" => "DRAINAGE",
            "valve" => "VALVE",
            "tank" => "TANK",
            "pump" => "PUMP",
            "hvac_equipment" => "HVAC_EQUIPMENT",
            "pressure_vessel" => "PRESSURE_VESSEL",
            "heat_exchanger" => "HEAT_EXCHANGER",
            "compressor" => "COMPRESSOR",
            "pressurizer" => "PRESSURIZER",
            "steam_generator" => "STEAM_GENERATOR",
            "emergency_diesel_generator" => "EMERGENCY_DIESEL_GENERATOR",
            "electrical_panel" => "ELECTRICAL_PANEL",
            "transformer" => "TRANSFORMER",
            "switchgear" => "SWITCHGEAR",
            "ups_system" => "UPS_SYSTEM",
            "junction_box" => "JUNCTION_BOX",
            "radiation_monitor" => "RADIATION_MONITOR",
            "fire_alarm_panel" => "FIRE_ALARM_PANEL",
            "smoke_detector" => "SMOKE_DETECTOR",
            "sprinkler" => "SPRINKLER",
            "lighting_fitting" => "LIGHTING_FITTING",
            "fire_hydrant" => "FIRE_HYDRANT",
            "column" => "COLUMN",
            "beam" => "BEAM",
            "railing" => "RAILING",
            _ => null
        };
    }

    private static bool LooksLikePipeLikeInternalComponent(SidecarSegment seg)
    {
        if (!string.Equals(seg.Shape, "box", StringComparison.OrdinalIgnoreCase))
            return false;

        var a = Math.Max(seg.WidthMm, 0.0);
        var b = Math.Max(seg.DepthMm, 0.0);
        var c = Math.Max(seg.HeightMm, 0.0);
        var dims = new[] { a, b, c }.OrderBy(x => x).ToArray();
        var minor = dims[0];
        var mid = dims[1];
        var major = dims[2];

        if (major < 500.0)
            return false;

        var aspect = major / Math.Max(mid, 1.0);
        return aspect >= 3.0 && mid <= 800.0 && minor <= 500.0;
    }
    public static BuildResult Build(Document doc, IList<SidecarSegment> segments)
    {
        var view = doc.ActiveView;
        var segmentsList = segments.ToList();

        _maxCylDiameter = 0.0;
        foreach (var s in segmentsList)
        {
            var shape = (s.Shape ?? "").ToLowerInvariant();
            if (shape == "cylinder" || shape == "tanks")
            {
                double diameter = Math.Min(s.WidthMm, s.DepthMm);
                if (diameter > _maxCylDiameter)
                {
                    _maxCylDiameter = diameter;
                }
            }
        }

        WriteErrorLog($"Runtime Check: _maxCylDiameter = {_maxCylDiameter}");

        var segmentLookup = segmentsList.ToDictionary(s => s.SegmentId, StringComparer.Ordinal);
        var consumedSegmentIds = new HashSet<string>(StringComparer.Ordinal);

        _selectedTankIds.Clear();

        // Genuine vertical cylinder candidates (silos/tanks/columns) — preserve authentic point cloud coordinates
        var candidates = segmentsList
            .Where(s => IsVerticalCylinderCandidate(s) && s.Centroid != null && s.BoundingBox != null)
            .ToList();

        WriteErrorLog($"Authentic Cylinder Detection: Found {candidates.Count} candidate cylinders");
        foreach (var c in candidates)
        {
            _selectedTankIds.Add(c.SegmentId);
        }

        string? lowestFloorSegmentId = null;
        double lowestFloorMinZ = double.MaxValue;
        
        // Find bottom floors dynamically
        var floorSegments = segmentsList
            .Where(s => string.Equals(s.ElementType, "FLOOR", StringComparison.OrdinalIgnoreCase) && s.Centroid != null)
            .ToList();

        var bottomFloors = new List<SidecarSegment>();
        if (floorSegments.Count > 0)
        {
            double minFloorZ = floorSegments.Min(f => f.Centroid.Z);
            bottomFloors = floorSegments
                .Where(f => f.Centroid.Z <= minFloorZ + 500.0)
                .ToList();
            lowestFloorMinZ = bottomFloors.Average(f => f.Centroid.Z);
        }
        else
        {
            // When floor segments are absent (e.g., wall-only payloads), anchor
            // geometry to the actual lowest segment so walls are not left floating
            // many meters above the model origin.
            lowestFloorMinZ = segmentsList
                .Where(s => s.BoundingBox != null)
                .Select(s => s.BoundingBox.MinZ)
                .DefaultIfEmpty(0.0)
                .Min();
        }

        if (lowestFloorMinZ != double.MaxValue)
        {
            double shiftZ = lowestFloorMinZ;
            foreach (var s in segmentsList)
            {
                var shape = (s.Shape ?? "").ToLowerInvariant();
                var type = (s.ElementType ?? "").ToUpperInvariant();
                bool isLargeCyl = shape == "cylinder" && s.WidthMm > 400.0 && s.HeightMm > 1000.0;
                bool isFloorOrWall =
                    type == "FLOOR" ||
                    type == "WALL" ||
                    type == "BUND_WALL" ||
                    s.IsVerticalPlane ||
                    string.Equals(shape, "plane_vertical", StringComparison.OrdinalIgnoreCase);

                if (s.BoundingBox != null)
                {
                    if ((isLargeCyl || IsLargeTankSegment(s)) && s.BoundingBox.MinZ < lowestFloorMinZ + 200.0)
                    {
                        s.BoundingBox.MinZ = lowestFloorMinZ;
                    }
                    else if (!isFloorOrWall && s.BoundingBox.MinZ < lowestFloorMinZ + 150.0)
                    {
                        consumedSegmentIds.Add(s.SegmentId);
                    }
                }

                if (s.BoundingBox != null)
                {
                    s.BoundingBox.MinZ -= shiftZ;
                    s.BoundingBox.MaxZ -= shiftZ;
                }
                if (s.Centroid != null)
                {
                    s.Centroid.Z -= shiftZ;
                }
                if (s.Tags != null)
                {
                    if (TryGetTagDouble(s, "cyl_start_z_mm", out var sz))
                        s.Tags["cyl_start_z_mm"] = sz - shiftZ;
                    if (TryGetTagDouble(s, "cyl_end_z_mm", out var ez))
                        s.Tags["cyl_end_z_mm"] = ez - shiftZ;
                    if (TryGetTagDouble(s, "reconstructed_storey_base_mm", out var rbase))
                        s.Tags["reconstructed_storey_base_mm"] = rbase - shiftZ;
                    if (TryGetTagDouble(s, "reconstructed_storey_top_mm", out var rtop))
                        s.Tags["reconstructed_storey_top_mm"] = rtop - shiftZ;
                    if (TryGetTagDouble(s, "wall_start_z_mm", out var wsz))
                        s.Tags["wall_start_z_mm"] = wsz - shiftZ;
                    if (TryGetTagDouble(s, "wall_end_z_mm", out var wez))
                        s.Tags["wall_end_z_mm"] = wez - shiftZ;
                }
            }
            lowestFloorMinZ = 0;
        }

        var created = 0;
        var skipped = 0;
        var errors = new List<string>();
        var elementIds = new List<ElementId>();
        var topHorizontalMaxZ = segmentsList
            .Where(s => s.IsHorizontalPlane)
            .Select(s => s.BoundingBox.MaxZ)
            .DefaultIfEmpty(double.MinValue)
            .Max();

        // Find floor and wall boundaries dynamically in mm
        double floorMinX = double.MaxValue;
        double floorMaxX = double.MinValue;
        double floorMinZ = 0.0;
        double wallMaxZ = 4000.0; // fallback

        if (floorSegments.Count > 0)
        {
            floorMinX = floorSegments.Min(f => f.BoundingBox.MinX);
            floorMaxX = floorSegments.Max(f => f.BoundingBox.MaxX);
            floorMinZ = floorSegments.Min(f => f.BoundingBox.MinZ);
        }
        else
        {
            floorMinX = segmentsList.Min(s => s.BoundingBox.MinX);
            floorMaxX = segmentsList.Max(s => s.BoundingBox.MaxX);
        }

        var allWallSegs = segmentsList
            .Where(s => string.Equals(s.ElementType, "WALL", StringComparison.OrdinalIgnoreCase) || string.Equals(s.ElementType, "BUND_WALL", StringComparison.OrdinalIgnoreCase))
            .ToList();

        if (allWallSegs.Count > 0)
        {
            wallMaxZ = allWallSegs.Max(w => w.BoundingBox.MaxZ);
        }

        // Find back wall segments dynamically
        var backWallSegs = new List<SidecarSegment>();
        if (allWallSegs.Count > 0)
        {
            double minWallY = allWallSegs.Min(w => w.Centroid.Y);
            backWallSegs = allWallSegs
                .Where(w => w.Centroid.Y <= minWallY + 600.0)
                .ToList();
        }

        double backWallY = backWallSegs.Count > 0 ? backWallSegs.Average(w => w.Centroid.Y) : 0.0;
        double floorZ = bottomFloors.Count > 0 ? bottomFloors.Average(f => f.Centroid.Z) : 0.0;

        // Find elevated horizontal cable trays dynamically
        var traySegs = segmentsList
            .Where(s => {
                var type = ResolveElementType(s, topHorizontalMaxZ);
                if (type != "PIPE" && type != "CABLE_TRAY") return false;

                if (TryGetCylinderEndpoints(s, out var start, out var end))
                {
                    var axis = (end - start).Normalize();
                    double dotX = Math.Abs(axis.DotProduct(XYZ.BasisX));
                    double cy = s.Centroid.Y;
                    double cz = s.Centroid.Z;
                    
                    bool nearBackWall = backWallSegs.Count > 0 ? Math.Abs(cy - backWallY) <= 1500.0 : false;
                    bool isElevated = cz > floorZ + 800.0;
                    
                    return dotX > 0.8 && nearBackWall && isElevated;
                }
                return false;
            })
            .ToList();

        double maxTrayZ = 0.0;
        if (traySegs.Count > 0)
        {
            maxTrayZ = traySegs.Max(t => t.BoundingBox.MaxZ);
        }

        if (EnableUnifiedBackWall && backWallSegs.Count > 0)
        {
            double minX = backWallSegs.Min(w => w.BoundingBox.MinX);
            double maxX = backWallSegs.Max(w => w.BoundingBox.MaxX);
            double avgY = backWallSegs.Average(w => w.Centroid.Y);
            double maxWallZ = backWallSegs.Max(w => w.BoundingBox.MaxZ);

            // Extend wall height to at least 300mm above the highest cable tray
            double maxZ = Math.Max(maxWallZ, maxTrayZ + 300.0);

            var unifiedWallSeg = new SidecarSegment
            {
                SegmentId = "unified-back-wall",
                ZoneId = backWallSegs[0].ZoneId,
                Shape = "plane_vertical",
                Normal = new SidecarPoint3D { X = 0, Y = -1, Z = 0 },
                Centroid = new SidecarPoint3D { X = (minX + maxX) * 0.5, Y = avgY, Z = maxZ * 0.5 },
                BoundingBox = new SidecarBoundingBox
                {
                    MinX = minX,
                    MaxX = maxX,
                    MinY = avgY - 100.0,
                    MaxY = avgY + 100.0,
                    MinZ = 0.0, // Extend all the way down to the floor
                    MaxZ = maxZ
                },
                ElementType = "WALL"
            };

            bool unifiedBackWallCreated = false;
            if (EnableNativeWalls && TryCreateWall(doc, unifiedWallSeg, out var wallId))
            {
                elementIds.Add(wallId);
                created++;
                unifiedBackWallCreated = true;
            }
            else if (TryCreateDirectShape(doc, view, unifiedWallSeg, BuiltInCategory.OST_Walls, out var wallDsId))
            {
                elementIds.Add(wallDsId);
                created++;
                unifiedBackWallCreated = true;
            }

            // Consume all wall segments only when the synthesized wall was created.
            // If synthesis fails, preserve original wall segments for normal processing.
            if (unifiedBackWallCreated)
            {
                foreach (var s in segmentsList)
                {
                    var type = ResolveElementType(s, topHorizontalMaxZ);
                    if (type == "WALL" || type == "BUND_WALL")
                    {
                        consumedSegmentIds.Add(s.SegmentId);
                    }
                }
            }
        }

        // Build unified horizontal cable trays only if explicitly enabled
        if (EnableUnifiedCableTrays && traySegs.Count > 0)
        {
            // Group by Y centroid rounded to nearest 600mm and Z centroid rounded to nearest 300mm
            var trayGroups = traySegs
                .GroupBy(s => $"{Math.Round(s.Centroid.Y / 600.0) * 600.0:F0}_{Math.Round(s.Centroid.Z / 300.0) * 300.0:F0}")
                .ToList();

            foreach (var grp in trayGroups)
            {
                double minX = grp.Min(s => s.BoundingBox.MinX);
                double maxX = grp.Max(s => s.BoundingBox.MaxX);
                double avgY = grp.Average(s => s.Centroid.Y);
                double avgZ = grp.Average(s => s.Centroid.Z);
                double avgDiam = grp.Average(s => Math.Min(s.WidthMm, s.DepthMm));
                double halfDiam = Math.Max(avgDiam / 2.0, 100.0);

                // Only build if the run is at least 1 metre long
                if (maxX - minX > 1000.0)
                {
                    var unifiedTraySeg = new SidecarSegment
                    {
                        SegmentId = $"unified-cable-tray-{grp.Key}",
                        ZoneId = grp.First().ZoneId,
                        Shape = "cylinder",
                        Normal = new SidecarPoint3D { X = 1, Y = 0, Z = 0 },
                        Centroid = new SidecarPoint3D { X = (minX + maxX) * 0.5, Y = avgY, Z = avgZ },
                        BoundingBox = new SidecarBoundingBox
                        {
                            MinX = minX,
                            MaxX = maxX,
                            MinY = avgY - halfDiam,
                            MaxY = avgY + halfDiam,
                            MinZ = avgZ - halfDiam,
                            MaxZ = avgZ + halfDiam
                        },
                        Tags = new Dictionary<string, object>
                        {
                            { "cyl_start_x_mm", minX },
                            { "cyl_start_y_mm", avgY },
                            { "cyl_start_z_mm", avgZ },
                            { "cyl_end_x_mm", maxX },
                            { "cyl_end_y_mm", avgY },
                            { "cyl_end_z_mm", avgZ },
                            { "diameter_mm", avgDiam }
                        },
                        ElementType = "CABLE_TRAY"
                    };

                    if (TryCreateCableTray(doc, unifiedTraySeg, out var trayId))
                    {
                        elementIds.Add(trayId);
                        created++;
                    }
                    else if (TryCreateDirectShape(doc, view, unifiedTraySeg, BuiltInCategory.OST_CableTray, out var trayDsId))
                    {
                        elementIds.Add(trayDsId);
                        created++;
                    }
                }
            }

            // Consume all segments used for cable trays so they are not re-processed individually
            foreach (var s in traySegs)
            {
                consumedSegmentIds.Add(s.SegmentId);
            }
        }

        var mergedRoofGroup = (MergedSegmentGroup?)null;

        if (mergedRoofGroup != null)
        {
            foreach (var id in mergedRoofGroup.SegmentIds)
                consumedSegmentIds.Add(id);

            var roofSegment = CreateSyntheticSegment(mergedRoofGroup, "plane_horizontal");
            if (TryCreateDirectShape(doc, view, roofSegment, BuiltInCategory.OST_Roofs, out var roofId))
            {
                elementIds.Add(roofId);
                created++;
            }
            else
            {
                var reason = "merged roof direct-shape creation failed";
                LogSkipGroup(mergedRoofGroup.SegmentIds, segmentLookup, reason);
                skipped += mergedRoofGroup.SegmentIds.Count;
                errors.AddRange(mergedRoofGroup.SegmentIds.Select(id => $"{id}: {reason}"));
            }
        }

        var createdWalls = new List<(XYZ Midpoint, double BaseZ, XYZ Direction)>();

        foreach (var seg in segmentsList)
        {
            if (consumedSegmentIds.Contains(seg.SegmentId))
                continue;

            if (!TryGetValidBoundingBox(seg, out var bboxReason))
            {
                LogSkip(seg, bboxReason);
                skipped++;
                errors.Add($"{seg.SegmentId}: {bboxReason}");
                continue;
            }
            try
            {
                var elementType = ResolveElementType(seg, topHorizontalMaxZ);

                switch (elementType)
                {
                    case "WALL":
                    case "BUND_WALL":
                        {
                            double wallDx = seg.BoundingBox.MaxX - seg.BoundingBox.MinX;
                            double wallDy = seg.BoundingBox.MaxY - seg.BoundingBox.MinY;
                            double wallDz = seg.BoundingBox.MaxZ - seg.BoundingBox.MinZ;
                            double wallLen = Math.Max(wallDx, wallDy);
                            if (wallLen < MinWallLengthMm || wallDz < MinWallHeightMm)
                            {
                                var skipReason = $"Skip very small wall segment (< {MinWallLengthMm / 1000.0:0.0}m length or < {MinWallHeightMm / 1000.0:0.0}m height)";
                                LogSkip(seg, skipReason);
                                skipped++;
                                break;
                            }

                            // Duplicate detection on same storey (within 1.0m Z and 200mm XY)
                            double baseZ = seg.StoreyBaseMm * MmToFeet;
                            XYZ midPt = new XYZ(seg.Centroid.X * MmToFeet, seg.Centroid.Y * MmToFeet, baseZ);
                            XYZ dirVec = XYZ.BasisX;
                            if (seg.TryGetWallEndpoints(out var sx, out var sy, out var ex, out var ey))
                            {
                                midPt = new XYZ((sx + ex) * 0.5 * MmToFeet, (sy + ey) * 0.5 * MmToFeet, baseZ);
                                var diff = new XYZ(ex - sx, ey - sy, 0);
                                if (diff.GetLength() > 1e-6)
                                    dirVec = diff.Normalize();
                            }

                            bool isDuplicate = false;
                            foreach (var existing in createdWalls)
                            {
                                if (Math.Abs(existing.BaseZ - baseZ) < 3.0 &&
                                    midPt.DistanceTo(existing.Midpoint) < (200.0 * MmToFeet) &&
                                    Math.Abs(dirVec.DotProduct(existing.Direction)) > 0.98)
                                {
                                    isDuplicate = true;
                                    break;
                                }
                            }
                            if (isDuplicate)
                            {
                                var skipReason = "Skip duplicate wall segment on same storey";
                                LogSkip(seg, skipReason);
                                skipped++;
                                break;
                            }

                            if (EnableNativeWalls && TryCreateWall(doc, seg, out var wallId))
                            {
                                elementIds.Add(wallId);
                                created++;
                                createdWalls.Add((midPt, baseZ, dirVec));
                                Trace.TraceInformation(
                                    "wall_created_native segment_id={0} bbox={1}",
                                    seg.SegmentId,
                                    FormatBoundingBox(seg.BoundingBox));
                            }
                            else if (TryCreateDirectShape(doc, view, seg, BuiltInCategory.OST_Walls, out var wallDsId))
                            {
                                elementIds.Add(wallDsId);
                                created++;
                                createdWalls.Add((midPt, baseZ, dirVec));
                                Trace.TraceInformation(
                                    "wall_created_directshape segment_id={0} bbox={1}",
                                    seg.SegmentId,
                                    FormatBoundingBox(seg.BoundingBox));
                            }
                            else
                            {
                                var reason = "FAILED_ELEMENT: wall native and oriented direct-shape creation failed (refusing silent AABB cuboid)";
                                LogSkip(seg, reason);
                                skipped++;
                                errors.Add($"{seg.SegmentId}: {reason}");
                            }
                        }
                        break;
                    case "ROOF":
                        {
                            var skipReason = "Skip roof element to keep model open";
                            LogSkip(seg, skipReason);
                            skipped++;
                        }
                        break;
                    case "FLOOR":
                        {

                            if (TryCreateFloor(doc, seg, out var floorId))
                            {
                                elementIds.Add(floorId);
                                created++;
                            }
                            else if (TryCreateDirectShape(doc, view, seg, BuiltInCategory.OST_Floors, out var floorDsId))
                            {
                                elementIds.Add(floorDsId);
                                created++;
                            }
                            else
                            {
                                var reason = "floor creation failed and floor direct-shape fallback failed";
                                LogSkip(seg, reason);
                                skipped++;
                                errors.Add($"{seg.SegmentId}: {reason}");
                            }
                        }
                        break;
                    case "CEILING":
                        if (TryCreateDirectShape(doc, view, seg, BuiltInCategory.OST_Ceilings, out var ceilingId))
                        {
                            elementIds.Add(ceilingId);
                            created++;
                        }
                        else
                        {
                            var reason = "ceiling creation failed";
                            LogSkip(seg, reason);
                            skipped++;
                            errors.Add($"{seg.SegmentId}: {reason}");
                        }
                        break;
                    case "PIPE":
                        if (TryCreatePipe(doc, seg, out var pipeId))
                        {
                            elementIds.Add(pipeId);
                            created++;
                        }
                        else if (TryCreateDirectShape(doc, view, seg, BuiltInCategory.OST_PipeCurves, out var pipeDsId))
                        {
                            elementIds.Add(pipeDsId);
                            created++;
                        }
                        else
                        {
                            var reason = "pipe creation failed and pipe direct-shape fallback failed";
                            LogSkip(seg, reason);
                            skipped++;
                            errors.Add($"{seg.SegmentId}: {reason}");
                        }
                        break;
                    case "DUCT":
                        if (TryCreateDuct(doc, seg, out var ductId))
                        {
                            elementIds.Add(ductId);
                            created++;
                        }
                        else if (TryCreateDirectShape(doc, view, seg, BuiltInCategory.OST_DuctCurves, out var ductDsId))
                        {
                            elementIds.Add(ductDsId);
                            created++;
                        }
                        else
                        {
                            var reason = "duct creation failed and duct direct-shape fallback failed";
                            LogSkip(seg, reason);
                            skipped++;
                            errors.Add($"{seg.SegmentId}: {reason}");
                        }
                        break;
                    case "CONDUIT":
                        if (TryCreateConduit(doc, seg, out var conduitId))
                        {
                            elementIds.Add(conduitId);
                            created++;
                        }
                        else if (TryCreateDirectShape(doc, view, seg, BuiltInCategory.OST_Conduit, out var conduitDsId))
                        {
                            elementIds.Add(conduitDsId);
                            created++;
                        }
                        else
                        {
                            var reason = "conduit creation failed and conduit direct-shape fallback failed";
                            LogSkip(seg, reason);
                            skipped++;
                            errors.Add($"{seg.SegmentId}: {reason}");
                        }
                        break;
                    case "CABLE_TRAY":
                        if (TryCreateCableTray(doc, seg, out var trayId))
                        {
                            elementIds.Add(trayId);
                            created++;
                        }
                        else if (TryCreateDirectShape(doc, view, seg, BuiltInCategory.OST_CableTray, out var trayDsId))
                        {
                            elementIds.Add(trayDsId);
                            created++;
                        }
                        else
                        {
                            var reason = "cable tray creation failed and direct-shape fallback failed";
                            LogSkip(seg, reason);
                            skipped++;
                            errors.Add($"{seg.SegmentId}: {reason}");
                        }
                        break;
                    case "VALVE":
                    case "SAFETY_RELIEF_VALVE":
                    case "STRAINER":
                    case "EXPANSION_JOINT":
                    case "DELUGE_VALVE":
                        if (TryCreateAccessoryInstance(doc, seg, BuiltInCategory.OST_PipeAccessory, out var valveId))
                        {
                            elementIds.Add(valveId);
                            created++;
                        }
                        else if (TryCreateDirectShape(doc, view, seg, BuiltInCategory.OST_PipeAccessory, out var valveDsId))
                        {
                            elementIds.Add(valveDsId);
                            created++;
                        }
                        else
                        {
                            var reason = "valve/accessory creation failed";
                            LogSkip(seg, reason);
                            skipped++;
                            errors.Add($"{seg.SegmentId}: {reason}");
                        }
                        break;
                    case "TANK":
                    case "PUMP":
                    case "HVAC_EQUIPMENT":
                    case "PRESSURE_VESSEL":
                    case "HEAT_EXCHANGER":
                    case "COMPRESSOR":
                    case "PRESSURIZER":
                    case "STEAM_GENERATOR":
                    case "EMERGENCY_DIESEL_GENERATOR":
                        if (TryCreateEquipmentInstance(doc, seg, BuiltInCategory.OST_MechanicalEquipment, out var mechId))
                        {
                            elementIds.Add(mechId);
                            created++;
                        }
                        else if (TryCreateDirectShape(doc, view, seg, BuiltInCategory.OST_MechanicalEquipment, out var mechDsId))
                        {
                            elementIds.Add(mechDsId);
                            created++;
                        }
                        else
                        {
                            var reason = "mechanical equipment creation failed";
                            LogSkip(seg, reason);
                            skipped++;
                            errors.Add($"{seg.SegmentId}: {reason}");
                        }
                        break;
                    case "ELECTRICAL_PANEL":
                    case "TRANSFORMER":
                    case "SWITCHGEAR":
                    case "UPS_SYSTEM":
                    case "JUNCTION_BOX":
                    case "RADIATION_MONITOR":
                    case "FIRE_ALARM_PANEL":
                        if (TryCreateEquipmentInstance(doc, seg, BuiltInCategory.OST_ElectricalEquipment, out var elecId))
                        {
                            elementIds.Add(elecId);
                            created++;
                        }
                        else if (TryCreateDirectShape(doc, view, seg, BuiltInCategory.OST_ElectricalEquipment, out var elecDsId))
                        {
                            elementIds.Add(elecDsId);
                            created++;
                        }
                        else
                        {
                            var reason = "electrical equipment creation failed";
                            LogSkip(seg, reason);
                            skipped++;
                            errors.Add($"{seg.SegmentId}: {reason}");
                        }
                        break;
                    case "SMOKE_DETECTOR":
                    case "SPRINKLER":
                    case "FIRE_EXTINGUISHER":
                    case "LIGHTING_FITTING":
                    case "FIRE_HYDRANT":
                    case "DRAINAGE":
                        var fixtureCat = elementType switch
                        {
                            "LIGHTING_FITTING" => BuiltInCategory.OST_LightingFixtures,
                            "SMOKE_DETECTOR" => BuiltInCategory.OST_FireAlarmDevices,
                            "SPRINKLER" => BuiltInCategory.OST_Sprinklers,
                            "FIRE_HYDRANT" => BuiltInCategory.OST_PlumbingFixtures,
                            "DRAINAGE" => BuiltInCategory.OST_PlumbingFixtures,
                            _ => BuiltInCategory.OST_GenericModel,
                        };
                        if (TryCreateEquipmentInstance(doc, seg, fixtureCat, out var fixtureId))
                        {
                            elementIds.Add(fixtureId);
                            created++;
                        }
                        else if (TryCreateDirectShape(doc, view, seg, fixtureCat, out var fixtureDsId))
                        {
                            elementIds.Add(fixtureDsId);
                            created++;
                        }
                        else if (TryCreateDirectShape(doc, view, seg, BuiltInCategory.OST_GenericModel, out var fixtureGenericDsId))
                        {
                            elementIds.Add(fixtureGenericDsId);
                            created++;
                        }
                        else
                        {
                            var reason = "fixture creation failed";
                            LogSkip(seg, reason);
                            skipped++;
                            errors.Add($"{seg.SegmentId}: {reason}");
                        }
                        break;
                    case "DOOR":
                        if (TryCreateNativeDoorOrOpening(doc, seg, isDoor: true, out var nativeDoorId))
                        {
                            elementIds.Add(nativeDoorId);
                            created++;
                        }
                        else
                        {
                            var reason = "door creation failed (host wall missing or degenerate geometry; proxy fabrication rejected)";
                            LogSkip(seg, reason);
                            skipped++;
                            errors.Add($"{seg.SegmentId}: {reason}");
                        }
                        break;
                    case "WINDOW":
                        if (TryCreateNativeDoorOrOpening(doc, seg, isDoor: false, out var nativeWinId))
                        {
                            elementIds.Add(nativeWinId);
                            created++;
                        }
                        else
                        {
                            var reason = "window creation failed (host wall missing or degenerate geometry; proxy fabrication rejected)";
                            LogSkip(seg, reason);
                            skipped++;
                            errors.Add($"{seg.SegmentId}: {reason}");
                        }
                        break;
                    case "STAIR":
                    case "STAIR_RUN":
                        {
                            var skipStairReason = "Skipping staircase element as requested by user";
                            LogSkip(seg, skipStairReason);
                            skipped++;
                        }
                        break;
                    case "RAMP":
                        {
                            var skipReason = "Skip ramp element";
                            LogSkip(seg, skipReason);
                            skipped++;
                        }
                        break;
                    case "COLUMN":
                        if (TryCreateColumn(doc, seg, out var colNativeId))
                        {
                            elementIds.Add(colNativeId);
                            created++;
                        }
                        else if (TryCreateDirectShape(doc, view, seg, BuiltInCategory.OST_StructuralColumns, out var colId))
                        {
                            elementIds.Add(colId);
                            created++;
                        }
                        else
                        {
                            var reason = "column creation failed";
                            LogSkip(seg, reason);
                            skipped++;
                            errors.Add($"{seg.SegmentId}: {reason}");
                        }
                        break;
                    case "BEAM":
                        if (TryCreateDirectShape(doc, view, seg, BuiltInCategory.OST_StructuralFraming, out var beamId))
                        {
                            elementIds.Add(beamId);
                            created++;
                        }
                        else
                        {
                            var reason = "beam creation failed";
                            LogSkip(seg, reason);
                            skipped++;
                            errors.Add($"{seg.SegmentId}: {reason}");
                        }
                        break;
                    case "RAILING":
                        if (TryCreateDirectShape(doc, view, seg, BuiltInCategory.OST_Railings, out var railingId))
                        {
                            elementIds.Add(railingId);
                            created++;
                        }
                        else
                        {
                            var reason = "railing creation failed";
                            LogSkip(seg, reason);
                            skipped++;
                            errors.Add($"{seg.SegmentId}: {reason}");
                        }
                        break;
                    case "CYLINDER":
                        if (TryCreateDirectShape(doc, view, seg, BuiltInCategory.OST_GenericModel, out var cylDsId))
                        {
                            elementIds.Add(cylDsId);
                            created++;
                        }
                        else
                        {
                            var reason = "cylinder DirectShape creation failed";
                            LogSkip(seg, reason);
                            skipped++;
                            errors.Add($"{seg.SegmentId}: {reason}");
                        }
                        break;
                    default:
                        var fallbackCategory = GetFallbackCategory(elementType);
                        if (TryCreateDirectShape(doc, view, seg, fallbackCategory, out var dsId))
                        {
                            elementIds.Add(dsId);
                            created++;
                        }
                        else
                        {
                            var reason = $"geometry creation failed for {seg.Shape}";
                            LogSkip(seg, reason);
                            skipped++;
                            errors.Add($"{seg.SegmentId}: {reason}");
                        }
                        break;
                }
            }
            catch (Exception ex)
            {
                var reason = $"exception: {ex.Message}";
                LogSkip(seg, reason);
                errors.Add($"{seg.SegmentId}: {reason}");
                skipped++;
            }
        }

        return new BuildResult
        {
            Created = created,
            Skipped = skipped,
            Errors = errors,
            ElementIds = elementIds
        };
    }

    private static BuiltInCategory GetFallbackCategory(string elementType)
    {
        return elementType switch
        {
            "WALL" or "BUND_WALL" => BuiltInCategory.OST_Walls,
            "FLOOR" => BuiltInCategory.OST_Floors,
            "CEILING" => BuiltInCategory.OST_Ceilings,
            "ROOF" => BuiltInCategory.OST_Roofs,
            "DOOR" => BuiltInCategory.OST_Doors,
            "WINDOW" => BuiltInCategory.OST_Windows,
            "STAIR" or "STAIR_RUN" => BuiltInCategory.OST_Stairs,
            "RAMP" => BuiltInCategory.OST_Ramps,
            "COLUMN" => BuiltInCategory.OST_StructuralColumns,
            "BEAM" => BuiltInCategory.OST_StructuralFraming,
            "RAILING" => BuiltInCategory.OST_Railings,
            // "PIPE" => BuiltInCategory.OST_PipeCurves,
            "DUCT" => BuiltInCategory.OST_DuctCurves,
            "CONDUIT" => BuiltInCategory.OST_Conduit,
            "CABLE_TRAY" => BuiltInCategory.OST_CableTray,
            "LADDER" => BuiltInCategory.OST_Stairs,
            "VALVE" or "SAFETY_RELIEF_VALVE" or "STRAINER" or "EXPANSION_JOINT" => BuiltInCategory.OST_PipeAccessory,
            "TANK" or "PUMP" or "HVAC_EQUIPMENT" or "PRESSURE_VESSEL" or "HEAT_EXCHANGER" or "COMPRESSOR" => BuiltInCategory.OST_MechanicalEquipment,
            "ELECTRICAL_PANEL" or "TRANSFORMER" or "SWITCHGEAR" or "UPS_SYSTEM" or "JUNCTION_BOX" or "RADIATION_MONITOR" or "FIRE_ALARM_PANEL" => BuiltInCategory.OST_ElectricalEquipment,
            "SMOKE_DETECTOR" => BuiltInCategory.OST_FireAlarmDevices,
            "SPRINKLER" => BuiltInCategory.OST_Sprinklers,
            "LIGHTING_FITTING" => BuiltInCategory.OST_LightingFixtures,
            "FIRE_HYDRANT" or "DRAINAGE" => BuiltInCategory.OST_PlumbingFixtures,
            _ => BuiltInCategory.OST_GenericModel,
        };
    }

    // ── Geometry dispatch ─────────────────────────────────────────────────────

    // private static IList<GeometryObject> BuildGeometry(SidecarSegment seg) =>
    //     seg.IsCylinder || seg.IsValveCandidate
    //         ? BuildCylinderGeometry(seg)
    //         : BuildBoxGeometry(seg.BoundingBox);
    // private static IList<GeometryObject> BuildGeometry(SidecarSegment seg)
    // {
    //     if (seg.Shape == "void")
    //         return Array.Empty<GeometryObject>();

    //     if (seg.IsCylinder || seg.IsValveCandidate)
    //         return BuildCylinderGeometry(seg);

    //     return BuildBoxGeometry(seg.BoundingBox);
    // }
    private static IList<GeometryObject> BuildGeometry(SidecarSegment seg)
    {
        if (IsLargeTankSegment(seg))
        {
            return BuildTankCylinderGeometry(seg);
        }

        // Voids are openings; they do not tessellate into solid DirectShape cuboids.
        if (string.Equals(seg.Shape, "void", StringComparison.OrdinalIgnoreCase))
            return Array.Empty<GeometryObject>();

        try
        {
            if (seg.IsCylinder || seg.IsValveCandidate)
            {
                // Pipe-like internals should render as enclosed hollow shells,
                // not as solid rods.
                var cylGeom = BuildPipeShellGeometry(seg);
                if (cylGeom != null && cylGeom.Count > 0)
                    return cylGeom;

                cylGeom = BuildCylinderGeometry(seg);
                if (cylGeom != null && cylGeom.Count > 0)
                    return cylGeom;

                // Never silently convert a failed pipe/cylinder into a fake box cuboid
                return Array.Empty<GeometryObject>();
            }

            if (IsWallLikeSegment(seg))
            {
                var wallGeom = BuildWallGeometry(seg);
                if (wallGeom != null && wallGeom.Count > 0)
                    return wallGeom;

                return Array.Empty<GeometryObject>();
            }

            if (string.Equals(seg.ElementType, "FLOOR", StringComparison.OrdinalIgnoreCase))
                return BuildFloorGeometry(seg.BoundingBox);

            if (IsEnclosedMechanicalType(seg))
            {
                var mechGeom = BuildHollowBoxGeometry(seg.BoundingBox, seg);
                if (mechGeom.Count > 0)
                    return mechGeom;
            }

            // Only tessellate verified box shapes
            if (string.Equals(seg.Shape, "box", StringComparison.OrdinalIgnoreCase))
                return BuildBoxGeometry(seg.BoundingBox);

            return Array.Empty<GeometryObject>();
        }
        catch
        {
            return Array.Empty<GeometryObject>();
        }
    }

    private static string ResolveElementType(SidecarSegment seg, double topHorizontalMaxZ)
    {
        if (IsLargeTankSegment(seg))
        {
            return "CYLINDER";
        }

        var explicitType = GetElementType(seg);

        var hasExplicitSemanticType = !string.IsNullOrWhiteSpace(seg.ElementType)
            || (seg.Tags != null && seg.Tags.ContainsKey("semantic_label"));

        if (hasExplicitSemanticType && !string.Equals(explicitType, "GENERIC", StringComparison.OrdinalIgnoreCase))
            return explicitType;

        // Ensure sloped segments are promoted to stair/ramp even when
        // element_type was not populated by upstream classification.
        if (string.Equals(seg.Shape, "plane_sloped", StringComparison.OrdinalIgnoreCase))
        {
            if (TryGetTagDouble(seg, "stair_steps", out var steps) && steps >= 2.0)
                return "STAIR";
            return "RAMP";
        }

        if (seg.IsCylinder || seg.IsValveCandidate)
            return "PIPE";

        if (LooksLikePipeLikeInternalComponent(seg))
            return "PIPE";

        if (seg.IsVerticalPlane)
            return "WALL";

        if (seg.IsHorizontalPlane && IsTopHorizontalPlane(seg, topHorizontalMaxZ))
            return "ROOF";

        var resolved = explicitType;

        if (resolved == "GENERIC")
        {
            var bb = seg.BoundingBox;
            var width = Math.Max(bb.MaxX - bb.MinX, 0.0);
            var depth = Math.Max(bb.MaxY - bb.MinY, 0.0);
            var height = Math.Max(bb.MaxZ - bb.MinZ, 0.0);
            var major = Math.Max(width, Math.Max(depth, height));
            var minor = Math.Min(width, Math.Min(depth, height));

            // Ceiling-level compact object fallback (common for luminaires).
            if (seg.Centroid.Z > 2400.0 && major <= 1200.0 && minor >= 20.0)
                return "LIGHTING_FITTING";
        }

        return resolved;
    }

    private static bool IsTopHorizontalPlane(SidecarSegment seg, double topHorizontalMaxZ)
    {
        if (!seg.IsHorizontalPlane)
            return false;

        return Math.Abs(seg.BoundingBox.MaxZ - topHorizontalMaxZ) <= 0.5;
    }

    private static bool TryGetValidBoundingBox(SidecarSegment seg, out string reason)
    {
        reason = "missing bounding box";
        if (seg.BoundingBox == null)
            return false;

        var width = seg.BoundingBox.MaxX - seg.BoundingBox.MinX;
        var depth = seg.BoundingBox.MaxY - seg.BoundingBox.MinY;
        var height = seg.BoundingBox.MaxZ - seg.BoundingBox.MinZ;
        var isWallLike = seg.IsVerticalPlane ||
                 string.Equals(seg.ElementType, "WALL", StringComparison.OrdinalIgnoreCase) ||
                 string.Equals(seg.ElementType, "BUND_WALL", StringComparison.OrdinalIgnoreCase);
        const double minBuildableMm = 10.0;

        if (double.IsNaN(width) || double.IsNaN(depth) || double.IsNaN(height) ||
            double.IsInfinity(width) || double.IsInfinity(depth) || double.IsInfinity(height))
        {
            reason = "bounding box contains NaN or Infinity";
            return false;
        }

        if (width <= 0 && !isWallLike)
        {
            reason = $"invalid bounding box width={width:0.###}";
            return false;
        }

        if (depth <= 0 && !isWallLike)
        {
            reason = $"invalid bounding box depth={depth:0.###}";
            return false;
        }

        if (height <= 0)
        {
            reason = $"invalid bounding box height={height:0.###}";
            return false;
        }

        if (isWallLike && Math.Max(width, depth) <= 0)
        {
            reason = "wall-like segment has no horizontal span";
            return false;
        }

        if (width < minBuildableMm || depth < minBuildableMm || height < minBuildableMm)
        {
            if (isWallLike)
            {
                reason = string.Empty;
                return true;
            }

            // Pipe-like segments can have very thin AABB extents on one axis at
            // branch/junction centers. Accept them when axis/radius metadata is
            // valid so they are not dropped before pipe creation.
            if (!HasBuildablePipeLikeGeometry(seg))
            {
                reason = $"degenerate bounding box (<{minBuildableMm:0}mm)";
                return false;
            }
        }

        reason = string.Empty;
        return true;
    }

    private static bool HasBuildablePipeLikeGeometry(SidecarSegment seg)
    {
        var pipeLike = seg.IsCylinder || seg.IsValveCandidate || LooksLikePipeLikeInternalComponent(seg);
        if (!pipeLike)
            return false;

        // Need a meaningful centerline length.
        if (!TryGetCylinderEndpoints(seg, out var start, out var end))
            return false;

        if (start.DistanceTo(end) < MinDimFeet)
            return false;

        // Radius metadata is preferred. If missing, fall back to bbox-derived
        // size for cylinders/valve candidates.
        if (TryGetTagDouble(seg, "fitted_radius_mm", out var fittedR) && fittedR > 0.0)
            return true;

        if (TryGetTagDouble(seg, "diameter_mm", out var dMm) && dMm > 0.0)
            return true;

        var approxDiameterMm = Math.Min(Math.Max(seg.WidthMm, 0.0), Math.Max(seg.DepthMm, 0.0));
        return approxDiameterMm > 0.0;
    }

    private static void LogSkip(SidecarSegment seg, string reason)
    {
        Trace.TraceWarning(
            "skip_segment shape={0} bbox={1} reason={2}",
            seg.Shape,
            FormatBoundingBox(seg.BoundingBox),
            reason);
    }

    private static void LogSkipGroup(
        IEnumerable<string> segmentIds,
        IDictionary<string, SidecarSegment> segmentLookup,
        string reason)
    {
        foreach (var segmentId in segmentIds)
        {
            if (segmentLookup.TryGetValue(segmentId, out var segment))
                LogSkip(segment, reason);
        }
    }

    private static SidecarBoundingBox MergeBoundingBoxes(SidecarBoundingBox left, SidecarBoundingBox right)
    {
        return new SidecarBoundingBox
        {
            MinX = Math.Min(left.MinX, right.MinX),
            MinY = Math.Min(left.MinY, right.MinY),
            MinZ = Math.Min(left.MinZ, right.MinZ),
            MaxX = Math.Max(left.MaxX, right.MaxX),
            MaxY = Math.Max(left.MaxY, right.MaxY),
            MaxZ = Math.Max(left.MaxZ, right.MaxZ),
        };
    }

    private static List<MergedSegmentGroup> BuildMergedWallGroups(IList<SidecarSegment> segments)
    {
        var groups = new Dictionary<string, MergedSegmentGroup>(StringComparer.Ordinal);

        foreach (var seg in segments.Where(s => s.IsVerticalPlane))
        {
            var key = GetWallGroupKey(seg);
            if (!groups.TryGetValue(key, out var group))
            {
                group = new MergedSegmentGroup
                {
                    GroupId = $"merged-wall-{groups.Count + 1}",
                    Representative = seg,
                    BoundingBox = seg.BoundingBox,
                    SegmentIds = new List<string> { seg.SegmentId }
                };
                groups[key] = group;
                continue;
            }

            group.BoundingBox = MergeBoundingBoxes(group.BoundingBox, seg.BoundingBox);
            group.SegmentIds.Add(seg.SegmentId);
        }

        return groups.Values.ToList();
    }

    private static MergedSegmentGroup? BuildMergedRoofGroup(IList<SidecarSegment> segments, double topHorizontalMaxZ)
    {
        var roofSegments = segments
            .Where(s => s.IsHorizontalPlane && IsTopHorizontalPlane(s, topHorizontalMaxZ))
            .ToList();

        if (roofSegments.Count == 0)
            return null;

        var group = new MergedSegmentGroup
        {
            GroupId = "merged-roof-01",
            Representative = roofSegments[0],
            BoundingBox = roofSegments[0].BoundingBox,
            SegmentIds = roofSegments.Select(s => s.SegmentId).ToList()
        };

        foreach (var seg in roofSegments.Skip(1))
            group.BoundingBox = MergeBoundingBoxes(group.BoundingBox, seg.BoundingBox);

        return group;
    }

    private static string GetWallGroupKey(SidecarSegment seg)
    {
        if (TryGetTagDouble(seg, "wall_start_x_mm", out var startX) &&
            TryGetTagDouble(seg, "wall_start_y_mm", out var startY) &&
            TryGetTagDouble(seg, "wall_end_x_mm", out var endX) &&
            TryGetTagDouble(seg, "wall_end_y_mm", out var endY))
        {
            // Collapse fragmented wall traces from the same brick wall plane.
            // Use a line-based key (axis + normal offset) so partial segments
            // with different endpoints still merge into one wall group.
            double wallTolMm = 300.0;
            if (TryGetTagDouble(seg, "wall_thickness_mm", out var wallThicknessMm) && wallThicknessMm > 0)
                wallTolMm = Math.Max(wallTolMm, wallThicknessMm * 3.0);

            static double Q(double value, double step) => Math.Round(value / step) * step;

            // Prefer axis tags for stable grouping across fragmented traces.
            if (TryGetTagDouble(seg, "wall_axis_x", out var axisX) &&
                TryGetTagDouble(seg, "wall_axis_y", out var axisY))
            {
                var len = Math.Sqrt(axisX * axisX + axisY * axisY);
                if (len > 1e-6)
                {
                    var ax = axisX / len;
                    var ay = axisY / len;

                    // Canonical direction for deterministic keys.
                    if (ax < 0 || (Math.Abs(ax) < 1e-9 && ay < 0))
                    {
                        ax = -ax;
                        ay = -ay;
                    }

                    var mx = (startX + endX) * 0.5;
                    var my = (startY + endY) * 0.5;

                    // Signed normal offset of line from origin.
                    var nx = -ay;
                    var ny = ax;
                    var offset = nx * mx + ny * my;

                    var bboxForZ = seg.BoundingBox;
                    var zMid = (bboxForZ.MinZ + bboxForZ.MaxZ) * 0.5;
                    var zBandStep = Math.Max(400.0, wallTolMm);

                    return string.Join("|",
                        "axis",
                        Q(ax, 0.02),
                        Q(ay, 0.02),
                        Q(offset, wallTolMm),
                        Q(zMid, zBandStep));
                }
            }

            var sx = Q(startX, wallTolMm);
            var sy = Q(startY, wallTolMm);
            var ex = Q(endX, wallTolMm);
            var ey = Q(endY, wallTolMm);

            // Canonical direction so A->B and B->A map to the same group.
            bool reverse = (sx > ex) || (Math.Abs(sx - ex) < 1e-6 && sy > ey);
            if (reverse)
            {
                (sx, ex) = (ex, sx);
                (sy, ey) = (ey, sy);
            }

            return string.Join("|", sx, sy, ex, ey);
        }

        var bb = seg.BoundingBox;
        const double bboxTolMm = 450.0;
        static double QB(double value) => Math.Round(value / bboxTolMm) * bboxTolMm;
        return string.Join("|",
            QB(bb.MinX),
            QB(bb.MaxX),
            QB(bb.MinY),
            QB(bb.MaxY));
    }

    private static SidecarSegment CreateSyntheticSegment(MergedSegmentGroup group, string shape)
    {
        return new SidecarSegment
        {
            SegmentId = group.GroupId,
            ZoneId = group.Representative.ZoneId,
            Shape = shape,
            Normal = group.Representative.Normal,
            Centroid = group.Representative.Centroid,
            BoundingBox = group.BoundingBox,
            PointCount = group.Representative.PointCount,
            Confidence = group.Representative.Confidence,
            ScanStationId = group.Representative.ScanStationId,
            SourceFile = group.Representative.SourceFile,
            Tags = group.Representative.Tags != null
                ? new Dictionary<string, object>(group.Representative.Tags)
                : new Dictionary<string, object>(),
            ElementType = group.Representative.ElementType
        };
    }

    private static string FormatBoundingBox(SidecarBoundingBox? bb)
    {
        if (bb == null)
            return "null";

        return $"({bb.MinX:0.###},{bb.MinY:0.###},{bb.MinZ:0.###})-({bb.MaxX:0.###},{bb.MaxY:0.###},{bb.MaxZ:0.###})";
    }

    private static bool TryCreateDirectShape(
        Document doc,
        View? view,
        SidecarSegment seg,
        BuiltInCategory category,
        out ElementId elementId)
    {
        elementId = ElementId.InvalidElementId;

        try
        {
            var geom = BuildGeometry(seg);
            if (geom == null || geom.Count == 0)
            {
                WriteErrorLog($"TryCreateDirectShape: BuildGeometry returned null or empty for segment {seg.SegmentId} (shape: {seg.Shape})");
                return false;
            }

            var ds = DirectShape.CreateElement(doc, new ElementId((long)category));
            ds.ApplicationId = "ScanToBIM.Processor";
            ds.ApplicationDataId = seg.SegmentId;

            string name = $"STB_{seg.Shape}_{seg.SegmentId.Substring(0, Math.Min(8, seg.SegmentId.Length))}";
            if (string.Equals(seg.Shape, "cylinder", StringComparison.OrdinalIgnoreCase) || IsLargeTankSegment(seg))
            {
                name = $"Mechanical Container_{seg.SegmentId.Substring(0, Math.Min(8, seg.SegmentId.Length))}";
            }
            ds.Name = name;

            ds.SetShape(geom);
            WriteSegmentParam(ds, seg);

            if (view != null)
            {
                try
                {
                    ApplyColourOverride(doc, view, ds.Id, seg);
                }
                catch (Exception ex)
                {
                    WriteErrorLog($"ApplyColourOverride warning for segment {seg.SegmentId}: {ex.Message}");
                }
            }

            elementId = ds.Id;
            return true;
        }
        catch (Exception ex)
        {
            WriteErrorLog($"TryCreateDirectShape failed for segment {seg.SegmentId} (shape: {seg.Shape}, category: {category}): {ex.Message}\n{ex.StackTrace}");
            return false;
        }
    }

    // ── Box tessellation ──────────────────────────────────────────────────────

    // private static IList<GeometryObject> BuildBoxGeometry(SidecarBoundingBox bb)
    // {
    //     // Apply inset to each face to reduce overlap / z-fighting with neighbours
    //     var x0 = bb.MinX * MmToFeet + InsetFeet; var x1 = bb.MaxX * MmToFeet - InsetFeet;
    //     var y0 = bb.MinY * MmToFeet + InsetFeet; var y1 = bb.MaxY * MmToFeet - InsetFeet;
    //     var z0 = bb.MinZ * MmToFeet + InsetFeet; var z1 = bb.MaxZ * MmToFeet - InsetFeet;

    //     if (x1 - x0 < MinDimFeet || y1 - y0 < MinDimFeet || z1 - z0 < MinDimFeet)
    //         return Array.Empty<GeometryObject>();

    //     var tsb = new TessellatedShapeBuilder
    //     {
    //         Target   = TessellatedShapeBuilderTarget.Solid,
    //         Fallback = TessellatedShapeBuilderFallback.Mesh,
    //     };
    //     tsb.OpenConnectedFaceSet(false);

    //     // 6 faces — vertices listed CCW when viewed from outside
    //     AddQuad(tsb, new XYZ(x0,y0,z0), new XYZ(x1,y0,z0), new XYZ(x1,y1,z0), new XYZ(x0,y1,z0)); // bottom
    //     AddQuad(tsb, new XYZ(x0,y0,z1), new XYZ(x0,y1,z1), new XYZ(x1,y1,z1), new XYZ(x1,y0,z1)); // top
    //     AddQuad(tsb, new XYZ(x0,y0,z0), new XYZ(x0,y0,z1), new XYZ(x1,y0,z1), new XYZ(x1,y0,z0)); // front (-Y)
    //     AddQuad(tsb, new XYZ(x0,y1,z0), new XYZ(x1,y1,z0), new XYZ(x1,y1,z1), new XYZ(x0,y1,z1)); // back  (+Y)
    //     AddQuad(tsb, new XYZ(x0,y0,z0), new XYZ(x0,y1,z0), new XYZ(x0,y1,z1), new XYZ(x0,y0,z1)); // left  (-X)
    //     AddQuad(tsb, new XYZ(x1,y0,z0), new XYZ(x1,y0,z1), new XYZ(x1,y1,z1), new XYZ(x1,y1,z0)); // right (+X)

    //     tsb.CloseConnectedFaceSet();
    //     tsb.Build();
    //     return
    //     tsb.GetBuildResult().GetGeometricalObjects();
    // }
    private static IList<GeometryObject> BuildBoxGeometry(SidecarBoundingBox bb)
    {
        double x0 = bb.MinX * MmToFeet;
        double y0 = bb.MinY * MmToFeet;
        double z0 = bb.MinZ * MmToFeet;
        double x1 = bb.MaxX * MmToFeet;
        double y1 = bb.MaxY * MmToFeet;
        double z1 = bb.MaxZ * MmToFeet;

        ApplyVisualScale(ref x0, ref x1, VisualScaleXY);
        ApplyVisualScale(ref y0, ref y1, VisualScaleXY);
        ApplyVisualScale(ref z0, ref z1, VisualScaleZ);

        double insetX = Math.Min(InsetFeet, Math.Max(0.0, (x1 - x0) / 2.0 - 1e-6));
        double insetY = Math.Min(InsetFeet, Math.Max(0.0, (y1 - y0) / 2.0 - 1e-6));
        double insetZ = Math.Min(InsetFeet, Math.Max(0.0, (z1 - z0) / 2.0 - 1e-6));

        x0 += insetX;
        y0 += insetY;
        z0 += insetZ;
        x1 -= insetX;
        y1 -= insetY;
        z1 -= insetZ;

        double width = x1 - x0;
        double depth = y1 - y0;
        double height = z1 - z0;

        if (width <= 0 || depth <= 0 || height <= 0)
            return Array.Empty<GeometryObject>();

        var solid = BuildRectSolid(x0, y0, z0, x1, y1, z1);
        if (solid == null)
            return Array.Empty<GeometryObject>();

        return new List<GeometryObject> { solid };
    }

    private static void ApplyVisualScale(ref double min, ref double max, double scale)
    {
        if (scale <= 0)
            return;

        var center = (min + max) * 0.5;
        var half = (max - min) * 0.5 * scale;
        min = center - half;
        max = center + half;
    }

    private static IList<GeometryObject> BuildHollowBoxGeometry(SidecarBoundingBox bb, SidecarSegment seg)
    {
        double x0 = bb.MinX * MmToFeet;
        double y0 = bb.MinY * MmToFeet;
        double z0 = bb.MinZ * MmToFeet;
        double x1 = bb.MaxX * MmToFeet;
        double y1 = bb.MaxY * MmToFeet;
        double z1 = bb.MaxZ * MmToFeet;

        ApplyVisualScale(ref x0, ref x1, VisualScaleXY);
        ApplyVisualScale(ref y0, ref y1, VisualScaleXY);
        ApplyVisualScale(ref z0, ref z1, VisualScaleZ);

        var outer = BuildRectSolid(x0, y0, z0, x1, y1, z1);
        if (outer == null)
            return Array.Empty<GeometryObject>();

        double shellMm = 0.0;
        if (!TryGetTagDouble(seg, "shell_thickness_mm", out shellMm) || shellMm <= 0)
        {
            var dimsMm = new[] { seg.WidthMm, seg.DepthMm, seg.HeightMm }
                .Select(v => Math.Max(v, 1.0))
                .OrderBy(v => v)
                .ToArray();
            shellMm = Math.Clamp(dimsMm[0] * 0.10, 20.0, 120.0);
        }

        double shellFt = shellMm * MmToFeet;
        double maxShellFt = Math.Max(
            MinDimFeet,
            Math.Min(Math.Min(x1 - x0, y1 - y0), z1 - z0) / 2.0 - MinDimFeet);
        shellFt = Math.Clamp(shellFt, MinDimFeet, maxShellFt);

        if (shellFt <= MinDimFeet)
            return new List<GeometryObject> { outer };

        var inner = BuildRectSolid(
            x0 + shellFt,
            y0 + shellFt,
            z0 + shellFt,
            x1 - shellFt,
            y1 - shellFt,
            z1 - shellFt);

        if (inner == null)
            return new List<GeometryObject> { outer };

        try
        {
            var shell = BooleanOperationsUtils.ExecuteBooleanOperation(
                outer,
                inner,
                BooleanOperationsType.Difference);
            return new List<GeometryObject> { shell };
        }
        catch
        {
            return new List<GeometryObject> { outer };
        }
    }

    // ── Cylinder prism tessellation ───────────────────────────────────────────

    // private static IList<GeometryObject> BuildCylinderGeometry(SidecarSegment seg)
    // {
    //     var bb  = seg.BoundingBox;
    //     var cx  = seg.Centroid.X * MmToFeet;
    //     var cy  = seg.Centroid.Y * MmToFeet;
    //     var z0  = bb.MinZ * MmToFeet + InsetFeet;
    //     var z1  = bb.MaxZ * MmToFeet - InsetFeet;

    //     // Prefer RANSAC-fitted radius (stored in tags by scan_tools) over AABB heuristic.
    //     // The AABB min(width,depth)/2 overestimates radius for diagonal pipes.
    //     double radiusMm = seg.WidthMm < seg.DepthMm ? seg.WidthMm : seg.DepthMm;
    //     if (seg.Tags != null
    //         && seg.Tags.TryGetValue("fitted_radius_mm", out var fittedObj)
    //         && fittedObj is double fittedR && fittedR > 0)
    //     {
    //         radiusMm = fittedR;
    //     }
    //     var r = (radiusMm / 2.0 - InsetMm) * MmToFeet;

    //     if (r < MinDimFeet || z1 - z0 < MinDimFeet)
    //         return Array.Empty<GeometryObject>();

    //     // Build circle points at top and bottom
    //     var bot = new XYZ[CylinderSides];
    //     var top = new XYZ[CylinderSides];
    //     for (int i = 0; i < CylinderSides; i++)
    //     {
    //         var a = 2.0 * Math.PI * i / CylinderSides;
    //         bot[i] = new XYZ(cx + r * Math.Cos(a), cy + r * Math.Sin(a), z0);
    //         top[i] = new XYZ(cx + r * Math.Cos(a), cy + r * Math.Sin(a), z1);
    //     }

    //     var tsb = new TessellatedShapeBuilder
    //     {
    //         Target   = TessellatedShapeBuilderTarget.Solid,
    //         Fallback = TessellatedShapeBuilderFallback.Mesh,
    //     };
    //     tsb.OpenConnectedFaceSet(false);

    //     // Bottom cap — CCW viewed from below (reverse order)
    //     tsb.AddFace(new TessellatedFace(
    //         bot.Reverse().ToList(), ElementId.InvalidElementId));

    //     // Top cap — CCW viewed from above
    //     tsb.AddFace(new TessellatedFace(
    //         top.ToList(), ElementId.InvalidElementId));

    //     // Side quads
    //     for (int i = 0; i < CylinderSides; i++)
    //     {
    //         int j = (i + 1) % CylinderSides;
    //         AddQuad(tsb, bot[i], bot[j], top[j], top[i]);
    //     }

    //     tsb.CloseConnectedFaceSet();
    //     tsb.Build();
    //     return
    //     tsb.GetBuildResult().GetGeometricalObjects();
    // }
    private static IList<GeometryObject> BuildCylinderGeometry(SidecarSegment seg)
    {
        if (!TryGetCylinderEndpoints(seg, out var start, out var end))
            return Array.Empty<GeometryObject>();

        double radiusMm = Math.Min(seg.WidthMm, seg.DepthMm) / 2.0;
        if (!TryGetTagDouble(seg, "fitted_radius_mm", out var fittedRadius) || fittedRadius <= 0)
        {
            if (TryGetTagDouble(seg, "diameter_mm", out var diameterMm) && diameterMm > 0)
                radiusMm = diameterMm / 2.0;
        }
        else
        {
            radiusMm = fittedRadius;
        }

        double r = Math.Max(radiusMm - InsetMm, 0.001) * MmToFeet;
        r *= CylinderRadiusVisualBoost;

        if (r <= 0)
            return Array.Empty<GeometryObject>();

        var solid = RevitElementFactory.BuildCylinderSolid(start, end, r);
        if (solid == null)
            return Array.Empty<GeometryObject>();

        return new List<GeometryObject> { solid };
    }

    private static IList<GeometryObject> BuildPipeShellGeometry(SidecarSegment seg)
    {
        if (!TryGetCylinderEndpoints(seg, out var start, out var end))
            return Array.Empty<GeometryObject>();

        var axis = end - start;
        var axisLen = axis.GetLength();
        if (axisLen < MinDimFeet)
            return Array.Empty<GeometryObject>();
        axis = axis.Normalize();

        double outerRadiusMm = Math.Min(seg.WidthMm, seg.DepthMm) / 2.0;
        if (TryGetTagDouble(seg, "fitted_radius_mm", out var fittedRadius) && fittedRadius > 0)
            outerRadiusMm = fittedRadius;
        else if (TryGetTagDouble(seg, "diameter_mm", out var diameterMm) && diameterMm > 0)
            outerRadiusMm = diameterMm / 2.0;

        if (outerRadiusMm <= 0)
            return Array.Empty<GeometryObject>();

        double wallMm = 0.0;
        if (!TryGetTagDouble(seg, "pipe_wall_thickness_mm", out wallMm) || wallMm <= 0)
            wallMm = Math.Clamp(outerRadiusMm * 0.12, 3.0, 25.0);

        double innerRadiusMm = outerRadiusMm - wallMm;
        if (innerRadiusMm <= 1.0)
            return BuildCylinderGeometry(seg);

        double outerR = Math.Max((outerRadiusMm - InsetMm) * MmToFeet, MinDimFeet / 2.0);
        outerR *= CylinderRadiusVisualBoost;
        double innerR = Math.Max((innerRadiusMm - InsetMm) * MmToFeet, MinDimFeet / 4.0);
        innerR *= CylinderRadiusVisualBoost;
        if (innerR >= outerR - 1e-6)
            return BuildCylinderGeometry(seg);

        XYZ refDir = Math.Abs(axis.DotProduct(XYZ.BasisZ)) > 0.95 ? XYZ.BasisX : XYZ.BasisZ;
        XYZ u = axis.CrossProduct(refDir).Normalize();
        XYZ v = axis.CrossProduct(u).Normalize();

        var outerBot = new XYZ[CylinderSides];
        var outerTop = new XYZ[CylinderSides];
        var innerBot = new XYZ[CylinderSides];
        var innerTop = new XYZ[CylinderSides];

        for (int i = 0; i < CylinderSides; i++)
        {
            double a = 2 * Math.PI * i / CylinderSides;
            var outerOffset = u.Multiply(outerR * Math.Cos(a)) + v.Multiply(outerR * Math.Sin(a));
            var innerOffset = u.Multiply(innerR * Math.Cos(a)) + v.Multiply(innerR * Math.Sin(a));

            outerBot[i] = start + outerOffset;
            outerTop[i] = end + outerOffset;
            innerBot[i] = start + innerOffset;
            innerTop[i] = end + innerOffset;
        }

        var tsb = new TessellatedShapeBuilder
        {
            Target = TessellatedShapeBuilderTarget.AnyGeometry,
            Fallback = TessellatedShapeBuilderFallback.Mesh,
        };

        tsb.OpenConnectedFaceSet(false);

        for (int i = 0; i < CylinderSides; i++)
        {
            int j = (i + 1) % CylinderSides;

            // Outer cylindrical skin
            AddQuad(tsb, outerBot[i], outerBot[j], outerTop[j], outerTop[i]);
            // Inner cylindrical skin (reverse winding)
            AddQuad(tsb, innerBot[j], innerBot[i], innerTop[i], innerTop[j]);
            // Start annulus cap
            AddQuad(tsb, outerBot[j], outerBot[i], innerBot[i], innerBot[j]);
            // End annulus cap
            AddQuad(tsb, outerTop[i], outerTop[j], innerTop[j], innerTop[i]);
        }

        tsb.CloseConnectedFaceSet();
        tsb.Build();
        return tsb.GetBuildResult().GetGeometricalObjects();
    }

    private static bool IsEnclosedMechanicalType(SidecarSegment seg)
    {
        if (string.IsNullOrWhiteSpace(seg.ElementType))
            return false;
        return EnclosedMechanicalTypes.Contains(seg.ElementType);
    }

    private static bool IsWallLikeSegment(SidecarSegment seg)
    {
        if (seg.IsVerticalPlane)
            return true;

        return string.Equals(seg.ElementType, "WALL", StringComparison.OrdinalIgnoreCase)
            || string.Equals(seg.ElementType, "BUND_WALL", StringComparison.OrdinalIgnoreCase);
    }

    private static IList<GeometryObject> BuildWallGeometry(SidecarSegment seg)
    {
        var bb = seg.BoundingBox;
        if (bb == null)
            return Array.Empty<GeometryObject>();

        if (!TryGetWallPoints(seg, out var p0Raw, out var p1Raw))
            return Array.Empty<GeometryObject>();

        var z0Mm = bb.MinZ;
        var z1Mm = bb.MaxZ;
        if (TryGetTagDouble(seg, "wall_start_z_mm", out var wsZ))
            z0Mm = wsZ;
        if (TryGetTagDouble(seg, "wall_end_z_mm", out var weZ))
            z1Mm = weZ;

        if (z1Mm < z0Mm)
        {
            var t = z0Mm;
            z0Mm = z1Mm;
            z1Mm = t;
        }

        var z0 = z0Mm * MmToFeet;
        var z1 = z1Mm * MmToFeet;
        var height = z1 - z0;
        if (height < MinDimFeet)
            return Array.Empty<GeometryObject>();

        var p0 = new XYZ(p0Raw.X, p0Raw.Y, z0);
        var p1 = new XYZ(p1Raw.X, p1Raw.Y, z0);

        var axis = p1 - p0;
        var axisLen = axis.GetLength();
        if (axisLen < MinDimFeet)
            return Array.Empty<GeometryObject>();
        axis = axis.Normalize();

        var thicknessMm = seg.WallThicknessMm;
        if (thicknessMm <= 0.0)
            thicknessMm = 200.0;

        var thicknessFt = Math.Max(thicknessMm * MmToFeet, MinDimFeet);

        var perp = new XYZ(-axis.Y, axis.X, 0.0);
        if (perp.GetLength() < 1e-9)
            return Array.Empty<GeometryObject>();
        perp = perp.Normalize();

        var halfT = thicknessFt * 0.5;
        var a = p0 + perp.Multiply(halfT);
        var b = p1 + perp.Multiply(halfT);
        var c = p1 - perp.Multiply(halfT);
        var d = p0 - perp.Multiply(halfT);

        var loop = new CurveLoop();
        loop.Append(Line.CreateBound(a, b));
        loop.Append(Line.CreateBound(b, c));
        loop.Append(Line.CreateBound(c, d));
        loop.Append(Line.CreateBound(d, a));

        try
        {
            var solid = GeometryCreationUtilities.CreateExtrusionGeometry(
                new List<CurveLoop> { loop },
                XYZ.BasisZ,
                height);
            return new List<GeometryObject> { solid };
        }
        catch
        {
            return Array.Empty<GeometryObject>();
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static bool TryGetTagDouble(SidecarSegment seg, string key, out double value)
    {
        var d = seg.GetTagDouble(key);
        if (d.HasValue)
        {
            value = d.Value;
            return true;
        }
        value = 0.0;
        return false;
    }

    private static bool TryGetWallPoints(SidecarSegment seg, out XYZ p0, out XYZ p1)
    {
        var bb = seg.BoundingBox;
        p0 = new XYZ(bb.MinX * MmToFeet, bb.MinY * MmToFeet, bb.MinZ * MmToFeet);
        p1 = new XYZ(bb.MaxX * MmToFeet, bb.MaxY * MmToFeet, bb.MinZ * MmToFeet);

        if (seg.TryGetWallEndpoints(out var startX, out var startY, out var endX, out var endY))
        {
            p0 = new XYZ(startX * MmToFeet, startY * MmToFeet, bb.MinZ * MmToFeet);
            p1 = new XYZ(endX * MmToFeet, endY * MmToFeet, bb.MinZ * MmToFeet);
            return true;
        }

        var nrm = seg.Normal;
        if (nrm != null && Math.Abs(nrm.Z) < 0.7 &&
            (Math.Abs(nrm.X) > 1e-6 || Math.Abs(nrm.Y) > 1e-6))
        {
            var axisX = -nrm.Y;
            var axisY = nrm.X;
            var axisLen = Math.Sqrt(axisX * axisX + axisY * axisY);
            axisX /= axisLen;
            axisY /= axisLen;

            var cxMm = (bb.MinX + bb.MaxX) * 0.5;
            var cyMm = (bb.MinY + bb.MaxY) * 0.5;

            var cornersX = new[] { bb.MinX, bb.MinX, bb.MaxX, bb.MaxX };
            var cornersY = new[] { bb.MinY, bb.MaxY, bb.MinY, bb.MaxY };
            double minProj = double.MaxValue, maxProj = double.MinValue;
            for (int ci = 0; ci < 4; ci++)
            {
                var proj = cornersX[ci] * axisX + cornersY[ci] * axisY;
                minProj = Math.Min(minProj, proj);
                maxProj = Math.Max(maxProj, proj);
            }
            var halfMm = Math.Max((maxProj - minProj) * 0.5, 100.0);

            p0 = new XYZ((cxMm - axisX * halfMm) * MmToFeet, (cyMm - axisY * halfMm) * MmToFeet, bb.MinZ * MmToFeet);
            p1 = new XYZ((cxMm + axisX * halfMm) * MmToFeet, (cyMm + axisY * halfMm) * MmToFeet, bb.MinZ * MmToFeet);
            return true;
        }

        return false;
    }

    private static bool TryGetCylinderEndpoints(SidecarSegment seg, out XYZ p0, out XYZ p1)
    {
        var bb = seg.BoundingBox;
        p0 = new XYZ(seg.Centroid.X * MmToFeet, seg.Centroid.Y * MmToFeet, bb.MinZ * MmToFeet);
        p1 = new XYZ(seg.Centroid.X * MmToFeet, seg.Centroid.Y * MmToFeet, bb.MaxZ * MmToFeet);

        bool success = false;

        if (TryGetTagDouble(seg, "cyl_start_x_mm", out var sx) &&
            TryGetTagDouble(seg, "cyl_start_y_mm", out var sy) &&
            TryGetTagDouble(seg, "cyl_start_z_mm", out var sz) &&
            TryGetTagDouble(seg, "cyl_end_x_mm", out var ex) &&
            TryGetTagDouble(seg, "cyl_end_y_mm", out var ey) &&
            TryGetTagDouble(seg, "cyl_end_z_mm", out var ez))
        {
            p0 = new XYZ(sx * MmToFeet, sy * MmToFeet, sz * MmToFeet);
            p1 = new XYZ(ex * MmToFeet, ey * MmToFeet, ez * MmToFeet);
            success = true;
        }
        else if (TryGetTagDouble(seg, "cyl_axis_x", out var ax) &&
            TryGetTagDouble(seg, "cyl_axis_y", out var ay) &&
            TryGetTagDouble(seg, "cyl_axis_z", out var az))
        {
            var axis = new XYZ(ax, ay, az);
            if (axis.GetLength() > 1e-6)
            {
                axis = axis.Normalize();
                var c = new XYZ(seg.Centroid.X * MmToFeet, seg.Centroid.Y * MmToFeet, seg.Centroid.Z * MmToFeet);
                var bbDx = Math.Max(seg.WidthMm, 0.0);
                var bbDy = Math.Max(seg.DepthMm, 0.0);
                var bbDz = Math.Max(seg.HeightMm, 0.0);
                var lengthFt = Math.Max(Math.Max(bbDx, Math.Max(bbDy, bbDz)) * MmToFeet, MinDimFeet);
                var half = lengthFt / 2.0;
                p0 = c - axis.Multiply(half);
                p1 = c + axis.Multiply(half);
                success = true;
            }
        }
        else
        {
            // Fallback to longest BB axis
            var dx = bb.MaxX - bb.MinX;
            var dy = bb.MaxY - bb.MinY;
            var dz = bb.MaxZ - bb.MinZ;
            var cx = (bb.MinX + bb.MaxX) * 0.5 * MmToFeet;
            var cy = (bb.MinY + bb.MaxY) * 0.5 * MmToFeet;
            var cz = (bb.MinZ + bb.MaxZ) * 0.5 * MmToFeet;

            if (dx >= dy && dx >= dz)
            {
                p0 = new XYZ(bb.MinX * MmToFeet, cy, cz);
                p1 = new XYZ(bb.MaxX * MmToFeet, cy, cz);
            }
            else if (dy >= dx && dy >= dz)
            {
                p0 = new XYZ(cx, bb.MinY * MmToFeet, cz);
                p1 = new XYZ(cx, bb.MaxY * MmToFeet, cz);
            }
            else
            {
                p0 = new XYZ(cx, cy, bb.MinZ * MmToFeet);
                p1 = new XYZ(cx, cy, bb.MaxZ * MmToFeet);
            }
            success = true;
        }

        if (success)
        {
            // Snapping to vertical / horizontal axes
            var delta = p1 - p0;
            double dxVal = Math.Abs(delta.X);
            double dyVal = Math.Abs(delta.Y);
            double dzVal = Math.Abs(delta.Z);

            if (dzVal >= dxVal && dzVal >= dyVal)
            {
                // Vertical
                double avgX = (p0.X + p1.X) / 2.0;
                double avgY = (p0.Y + p1.Y) / 2.0;
                p0 = new XYZ(avgX, avgY, p0.Z);
                p1 = new XYZ(avgX, avgY, p1.Z);
            }
            else if (dxVal >= dyVal)
            {
                // Horizontal along X
                double avgY = (p0.Y + p1.Y) / 2.0;
                double avgZ = (p0.Z + p1.Z) / 2.0;
                p0 = new XYZ(p0.X, avgY, avgZ);
                p1 = new XYZ(p1.X, avgY, avgZ);
            }
            else
            {
                // Horizontal along Y
                double avgX = (p0.X + p1.X) / 2.0;
                double avgZ = (p0.Z + p1.Z) / 2.0;
                p0 = new XYZ(avgX, p0.Y, avgZ);
                p1 = new XYZ(avgX, p1.Y, avgZ);
            }

            ExtendEndpoints(ref p0, ref p1, PipeJoinExtensionMm * MmToFeet);
            return p0.DistanceTo(p1) >= MinDimFeet;
        }

        return false;
    }

    private static void ExtendEndpoints(ref XYZ p0, ref XYZ p1, double extensionFt)
    {
        if (extensionFt <= 0)
            return;

        var axis = p1 - p0;
        var len = axis.GetLength();
        if (len < MinDimFeet * 2)
            return;

        axis = axis.Normalize();
        p0 = p0 - axis.Multiply(extensionFt);
        p1 = p1 + axis.Multiply(extensionFt);
    }

    private static SidecarSegment BuildOpeningProxySegment(SidecarSegment seg, bool isDoor)
    {
        var bb = seg.BoundingBox;
        var cx = (bb.MinX + bb.MaxX) * 0.5;
        var cy = (bb.MinY + bb.MaxY) * 0.5;
        var cz = (bb.MinZ + bb.MaxZ) * 0.5;

        var width = Math.Max(bb.MaxX - bb.MinX, 1.0);
        var depth = Math.Max(bb.MaxY - bb.MinY, 1.0);
        var height = Math.Max(bb.MaxZ - bb.MinZ, 1.0);

        bool alongX = width >= depth;
        var major = alongX ? width : depth;
        var targetMajor = isDoor
            ? Math.Clamp(major, 700.0, 1400.0)
            : Math.Clamp(major, 800.0, 2400.0);
        var targetThk = Math.Clamp(Math.Min(width, depth), 120.0, 300.0);
        var targetHeight = isDoor
            ? Math.Clamp(Math.Max(height, 2000.0), 2000.0, 2400.0)
            : Math.Clamp(Math.Max(height, 900.0), 900.0, 1700.0);

        var minX = alongX ? cx - targetMajor / 2.0 : cx - targetThk / 2.0;
        var maxX = alongX ? cx + targetMajor / 2.0 : cx + targetThk / 2.0;
        var minY = alongX ? cy - targetThk / 2.0 : cy - targetMajor / 2.0;
        var maxY = alongX ? cy + targetThk / 2.0 : cy + targetMajor / 2.0;

        double minZ;
        double maxZ;
        if (isDoor)
        {
            var baseZ = bb.MinZ < 400.0 ? bb.MinZ : Math.Min(bb.MinZ, cz - targetHeight / 2.0);
            minZ = baseZ;
            maxZ = baseZ + targetHeight;
        }
        else
        {
            minZ = cz - targetHeight / 2.0;
            maxZ = cz + targetHeight / 2.0;
        }

        var adjBb = new SidecarBoundingBox
        {
            MinX = minX,
            MaxX = maxX,
            MinY = minY,
            MaxY = maxY,
            MinZ = minZ,
            MaxZ = maxZ,
        };

        return new SidecarSegment
        {
            SegmentId = seg.SegmentId,
            ZoneId = seg.ZoneId,
            Shape = seg.Shape,
            Normal = seg.Normal,
            Centroid = new SidecarPoint3D { X = cx, Y = cy, Z = (minZ + maxZ) * 0.5 },
            BoundingBox = adjBb,
            PointCount = seg.PointCount,
            Confidence = seg.Confidence,
            ScanStationId = seg.ScanStationId,
            SourceFile = seg.SourceFile,
            Tags = seg.Tags != null ? new Dictionary<string, object>(seg.Tags) : new Dictionary<string, object>(),
            ElementType = seg.ElementType,
        };
    }

    private static void AddQuad(TessellatedShapeBuilder tsb,
        XYZ p0, XYZ p1, XYZ p2, XYZ p3)
    {
        tsb.AddFace(new TessellatedFace(
            new List<XYZ> { p0, p1, p2, p3 },
            ElementId.InvalidElementId));
    }

    private static void ApplyColourOverride(
        Document doc, View view, ElementId elementId, SidecarSegment seg)
    {
        var elem = GetElementType(seg);

        var colour = elem switch
        {
            "WALL" or "BUND_WALL" => new Color(198, 198, 198),
            "DOOR" => new Color(140, 100, 70),
            "WINDOW" => new Color(130, 170, 195),
            _ => seg.Confidence >= 0.85 ? ColourHigh
               : seg.Confidence >= 0.60 ? ColourMedium
               : ColourLow,
        };

        var ogs = new OverrideGraphicSettings();
        ogs.SetProjectionLineColor(colour);

        // Surface pattern colour requires Revit 2019+; guard with try/catch
        try
        {
            var solidFill = GetSolidFillPatternId(doc);
            if (solidFill != ElementId.InvalidElementId)
            {
                ogs.SetSurfaceForegroundPatternId(solidFill);
                ogs.SetSurfaceForegroundPatternColor(colour);
                var transparency = elem switch
                {
                    "WALL" or "BUND_WALL" => 5,
                    "DOOR" or "WINDOW" => 10,
                    _ => 20,
                };
                ogs.SetSurfaceTransparency(transparency);
            }
        }
        catch { /* older Revit version — line colour only */ }

        try
        {
            view.SetElementOverrides(elementId, ogs);
        }
        catch (Exception ex)
        {
            WriteErrorLog($"view.SetElementOverrides warning for element {elementId}: {ex.Message}");
        }
    }

    private static bool TryGetDominantColor(SidecarSegment seg, out Color color)
    {
        color = ColourMedium;
        if (seg.Tags == null)
            return false;

        object? raw = null;
        if (seg.Tags.TryGetValue("trace_color_hex", out var traceObj) && traceObj != null)
            raw = traceObj;
        else if (seg.Tags.TryGetValue("dominant_color", out var domObj) && domObj != null)
            raw = domObj;

        if (raw == null)
            return false;

        string? hex = raw switch
        {
            JsonElement e when e.ValueKind == JsonValueKind.String => e.GetString(),
            _ => raw.ToString(),
        };

        if (string.IsNullOrWhiteSpace(hex))
            return false;

        var s = hex.Trim();
        if (!s.StartsWith("#", StringComparison.Ordinal))
            s = "#" + s;
        if (s.Length != 7)
            return false;

        if (!byte.TryParse(s.Substring(1, 2), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var r))
            return false;
        if (!byte.TryParse(s.Substring(3, 2), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var g))
            return false;
        if (!byte.TryParse(s.Substring(5, 2), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var b))
            return false;

        color = new Color(r, g, b);
        return true;
    }

    private static ElementId GetSolidFillPatternId(Document doc)
    {
        // Find the built-in <Solid Fill> pattern (invariant across locales)
        var collector = new FilteredElementCollector(doc)
            .OfClass(typeof(FillPatternElement))
            .Cast<FillPatternElement>();

        foreach (var fp in collector)
        {
            if (fp.GetFillPattern().IsSolidFill)
                return fp.Id;
        }
        return ElementId.InvalidElementId;
    }

    private static void WriteSegmentParam(Element element, SidecarSegment seg)
    {
        // Write ScanToBIM_SegmentId if the shared parameter was loaded
        var p = element.LookupParameter(SharedParamNames.SegmentId);
        p?.Set(seg.SegmentId);

        // SafetyCategory set to NS until backend classifies via Sprint P4
        var ps = element.LookupParameter(SharedParamNames.SafetyCategory);
        ps?.Set("NS");

        // ── USIBD LOA v3.1 declaration ──────────────────────────────────────
        // Every DirectShape carries its accuracy tier + σ so downstream
        // consumers know the allowed use (fabrication? design? schematic?).
        var pTier = element.LookupParameter(SharedParamNames.LoaTier);
        pTier?.Set(seg.LoaTier);

        var pSigma = element.LookupParameter(SharedParamNames.LoaSigmaMm);
        pSigma?.Set(seg.LoaSigmaMm);

        var pStd = element.LookupParameter(SharedParamNames.LoaStandardVersion);
        pStd?.Set("USIBD LOA v3.1");
    }
    // ── Native element creation stubs ─────────────────────────────────────

    private static bool TryCreateWall(Document doc, SidecarSegment seg, out ElementId wallId)
    {
        wallId = ElementId.InvalidElementId;
        try
        {
            var bb = seg.BoundingBox;

            // Semantic base elevation and height (P0.4)
            double baseMm = seg.StoreyBaseMm;
            double topMm = seg.StoreyTopMm;
            double heightMm = seg.StoreyHeightMm;

            double z0 = baseMm * MmToFeet;
            double height = heightMm * MmToFeet;

            double thicknessMm = seg.WallThicknessMm;
            var wallType = SelectClosestWallType(doc, thicknessMm * MmToFeet);
            var level = GetNearestLevel(doc, z0);

            if (wallType == null || level == null)
            {
                WriteErrorLog($"TryCreateWall: wallType is {(wallType == null ? "null" : "not null")}, level is {(level == null ? "null" : "not null")}");
                return false;
            }

            XYZ p0, p1;
            if (!TryGetWallPoints(seg, out p0, out p1))
            {
                // Prefer an angled centerline derived from the face normal over
                // the axis-aligned AABB fallback: for walls at e.g. 46° the
                // AABB longest-edge fallback would snap the wall to X or Y and
                // destroy the true orientation.
                var nrm = seg.Normal;
                if (nrm != null && Math.Abs(nrm.Z) < 0.7 &&
                    (Math.Abs(nrm.X) > 1e-6 || Math.Abs(nrm.Y) > 1e-6))
                {
                    var axisX = -nrm.Y;
                    var axisY = nrm.X;
                    var axisLen = Math.Sqrt(axisX * axisX + axisY * axisY);
                    axisX /= axisLen;
                    axisY /= axisLen;

                    var cxMm = (bb.MinX + bb.MaxX) * 0.5;
                    var cyMm = (bb.MinY + bb.MaxY) * 0.5;

                    // Length = extent of the AABB projected onto the wall axis.
                    var cornersX = new[] { bb.MinX, bb.MinX, bb.MaxX, bb.MaxX };
                    var cornersY = new[] { bb.MinY, bb.MaxY, bb.MinY, bb.MaxY };
                    double minProj = double.MaxValue, maxProj = double.MinValue;
                    for (int ci = 0; ci < 4; ci++)
                    {
                        var proj = cornersX[ci] * axisX + cornersY[ci] * axisY;
                        minProj = Math.Min(minProj, proj);
                        maxProj = Math.Max(maxProj, proj);
                    }
                    var halfMm = Math.Max((maxProj - minProj) * 0.5, 100.0);

                    p0 = new XYZ((cxMm - axisX * halfMm) * MmToFeet, (cyMm - axisY * halfMm) * MmToFeet, level.Elevation);
                    p1 = new XYZ((cxMm + axisX * halfMm) * MmToFeet, (cyMm + axisY * halfMm) * MmToFeet, level.Elevation);
                }
                else
                {
                    double dxMm = Math.Max(bb.MaxX - bb.MinX, 0.0);
                    double dyMm = Math.Max(bb.MaxY - bb.MinY, 0.0);
                    if (dxMm >= dyMm)
                    {
                        var cy = (bb.MinY + bb.MaxY) * 0.5 * MmToFeet;
                        p0 = new XYZ(bb.MinX * MmToFeet, cy, level.Elevation);
                        p1 = new XYZ(bb.MaxX * MmToFeet, cy, level.Elevation);
                    }
                    else
                    {
                        var cx = (bb.MinX + bb.MaxX) * 0.5 * MmToFeet;
                        p0 = new XYZ(cx, bb.MinY * MmToFeet, level.Elevation);
                        p1 = new XYZ(cx, bb.MaxY * MmToFeet, level.Elevation);
                    }
                }
            }
            else
            {
                // Baseline must lie in the plane of the host level
                p0 = new XYZ(p0.X, p0.Y, level.Elevation);
                p1 = new XYZ(p1.X, p1.Y, level.Elevation);
            }

            // Minimum length check (>= 300 mm)
            if (p0.DistanceTo(p1) < MinDimFeet || p0.DistanceTo(p1) < (MinWallLengthMm * MmToFeet))
            {
                WriteErrorLog($"TryCreateWall: wall segment {seg.SegmentId} length {p0.DistanceTo(p1) / MmToFeet:F1}mm below minimum {MinWallLengthMm:F0}mm");
                return false;
            }

            var line = Line.CreateBound(p0, p1);
            double baseOffset = z0 - level.Elevation;

            var wall = Wall.Create(doc, line, wallType.Id, level.Id, height, baseOffset, false, false);
            wallId = wall.Id;

            // Disallow end joins to prevent Revit geometry corruption / overlap failures (P0.4)
            WallUtils.DisallowWallJoinAtEnd(wall, 0);
            WallUtils.DisallowWallJoinAtEnd(wall, 1);

            WriteSegmentParam(wall, seg);
            return true;
        }
        catch (Exception ex)
        {
            WriteErrorLog($"TryCreateWall failed for {seg.SegmentId}: {ex.Message}\n{ex.StackTrace}");
            return false;
        }
    }

    private static bool IsFoundationOrFooting(WallType wt)
    {
        var name = wt.Name;
        if (name.IndexOf("fnd", StringComparison.OrdinalIgnoreCase) >= 0 ||
            name.IndexOf("footing", StringComparison.OrdinalIgnoreCase) >= 0 ||
            name.IndexOf("foundation", StringComparison.OrdinalIgnoreCase) >= 0 ||
            name.IndexOf("retaining", StringComparison.OrdinalIgnoreCase) >= 0 ||
            name.IndexOf("bearing", StringComparison.OrdinalIgnoreCase) >= 0)
            return true;

        try
        {
            var funcParam = wt.get_Parameter(BuiltInParameter.FUNCTION_PARAM);
            if (funcParam != null && funcParam.HasValue && funcParam.AsInteger() == (int)WallFunction.Foundation)
                return true;
        }
        catch { }

        return false;
    }

    private static WallType? SelectClosestWallType(Document doc, double targetWidthFt)
    {
        // Clamp target width to realistic architectural wall range [100mm, 500mm]
        double clampedFt = Math.Min(Math.Max(targetWidthFt, 100.0 * MmToFeet), 500.0 * MmToFeet);

        var basicTypes = new FilteredElementCollector(doc)
            .OfClass(typeof(WallType))
            .Cast<WallType>()
            .Where(wt => wt.Kind == WallKind.Basic)
            .ToList();

        // 1. Filter out foundation/footing/retaining types
        var nonFoundation = basicTypes
            .Where(wt => !IsFoundationOrFooting(wt))
            .ToList();

        var pool = nonFoundation.Count > 0 ? nonFoundation : basicTypes;

        // 2. Prefer standard architectural types (Generic, Basic Wall, Interior, Exterior)
        var preferred = pool
            .Where(wt =>
            {
                var n = wt.Name;
                return n.IndexOf("generic", StringComparison.OrdinalIgnoreCase) >= 0 ||
                       n.IndexOf("basic", StringComparison.OrdinalIgnoreCase) >= 0 ||
                       n.IndexOf("interior", StringComparison.OrdinalIgnoreCase) >= 0 ||
                       n.IndexOf("exterior", StringComparison.OrdinalIgnoreCase) >= 0;
            })
            .ToList();

        var candidates = preferred.Count > 0 ? preferred : pool;

        return candidates
            .OrderBy(wt => Math.Abs(GetWallTypeWidthFeet(wt) - clampedFt))
            .FirstOrDefault()
            ?? basicTypes.FirstOrDefault();
    }

    private static double GetWallTypeWidthFeet(WallType wallType)
    {
        return wallType.Width;
    }


    private static bool TryCreateFloor(Document doc, SidecarSegment seg, out ElementId floorId)
    {
        floorId = ElementId.InvalidElementId;
        try
        {
            var bb = seg.BoundingBox;
            double x0 = bb.MinX * MmToFeet;
            double x1 = bb.MaxX * MmToFeet;
            double y0 = bb.MinY * MmToFeet;
            double y1 = bb.MaxY * MmToFeet;
            double z0 = bb.MinZ * MmToFeet;

            var floorType = new FilteredElementCollector(doc)
                .OfClass(typeof(FloorType)).Cast<FloorType>()
                .FirstOrDefault();
            var level = new FilteredElementCollector(doc)
                .OfClass(typeof(Level)).Cast<Level>()
                .OrderBy(l => Math.Abs(l.Elevation - z0)).FirstOrDefault();

            if (floorType == null || level == null)
            {
                WriteErrorLog($"TryCreateFloor: floorType is {(floorType == null ? "null" : "not null")}, level is {(level == null ? "null" : "not null")}");
                return false;
            }

            double levelElevation = level.Elevation;
            var loop = new CurveLoop();
            bool polySuccess = false;

            if (seg.Tags != null && seg.Tags.TryGetValue("boundary_polygon_mm", out var polyObj))
            {
                var pts = ParseBoundaryPolygon(polyObj, levelElevation);
                if (pts.Count >= 3)
                {
                    try
                    {
                        for (int i = 0; i < pts.Count; i++)
                        {
                            var pCurrent = pts[i];
                            var pNext = pts[(i + 1) % pts.Count];
                            if (pCurrent.DistanceTo(pNext) >= MinDimFeet)
                            {
                                loop.Append(Line.CreateBound(pCurrent, pNext));
                            }
                        }
                        if (!loop.IsOpen())
                        {
                            polySuccess = true;
                        }
                    }
                    catch
                    {
                        loop = new CurveLoop();
                        polySuccess = false;
                    }
                }
            }

            if (!polySuccess)
            {
                loop = new CurveLoop();
                loop.Append(Line.CreateBound(new XYZ(x0, y0, levelElevation), new XYZ(x1, y0, levelElevation)));
                loop.Append(Line.CreateBound(new XYZ(x1, y0, levelElevation), new XYZ(x1, y1, levelElevation)));
                loop.Append(Line.CreateBound(new XYZ(x1, y1, levelElevation), new XYZ(x0, y1, levelElevation)));
                loop.Append(Line.CreateBound(new XYZ(x0, y1, levelElevation), new XYZ(x0, y0, levelElevation)));
            }
            var profile = new List<CurveLoop> { loop };

            var floor = Floor.Create(doc, profile, floorType.Id, level.Id);
            floorId = floor.Id;

            // Set height offset if there is a difference
            double offset = z0 - levelElevation;
            if (Math.Abs(offset) > 1e-5)
            {
                var candidates = new[]
                {
                    BuiltInParameter.INSTANCE_FREE_HOST_OFFSET_PARAM,
                    BuiltInParameter.INSTANCE_ELEVATION_PARAM,
                    BuiltInParameter.FAMILY_BASE_LEVEL_OFFSET_PARAM,
                    BuiltInParameter.WALL_BASE_OFFSET,
                    BuiltInParameter.FLOOR_HEIGHTABOVELEVEL_PARAM,
                    BuiltInParameter.CEILING_HEIGHTABOVELEVEL_PARAM,
                };
                foreach (var bip in candidates)
                {
                    var p = floor.get_Parameter(bip);
                    if (p != null && !p.IsReadOnly && p.StorageType == StorageType.Double)
                    {
                        try
                        {
                            p.Set(offset);
                            break;
                        }
                        catch { }
                    }
                }
            }

            WriteSegmentParam(floor, seg);
            return true;
        }
        catch (Exception ex)
        {
            WriteErrorLog($"TryCreateFloor failed: {ex.Message}\n{ex.StackTrace}");
            return false;
        }
    }

    private static List<XYZ> ParseBoundaryPolygon(object polyObj, double elevation)
    {
        var pts = new List<XYZ>();
        if (polyObj == null) return pts;

        try
        {
            if (polyObj is JsonElement elem && elem.ValueKind == JsonValueKind.Array)
            {
                foreach (var ptElem in elem.EnumerateArray())
                {
                    if (ptElem.ValueKind == JsonValueKind.Array)
                    {
                        var coords = ptElem.EnumerateArray().ToList();
                        if (coords.Count >= 2)
                        {
                            double xMm = coords[0].GetDouble();
                            double yMm = coords[1].GetDouble();
                            pts.Add(new XYZ(xMm * MmToFeet, yMm * MmToFeet, elevation));
                        }
                    }
                }
            }
            else if (polyObj is System.Collections.IEnumerable list)
            {
                foreach (var item in list)
                {
                    if (item is System.Collections.IEnumerable subList && !(item is string))
                    {
                        var coords = new List<double>();
                        foreach (var c in subList)
                        {
                            coords.Add(Convert.ToDouble(c, CultureInfo.InvariantCulture));
                        }
                        if (coords.Count >= 2)
                        {
                            pts.Add(new XYZ(coords[0] * MmToFeet, coords[1] * MmToFeet, elevation));
                        }
                    }
                    else if (item is JsonElement subElem && subElem.ValueKind == JsonValueKind.Array)
                    {
                        var coords = subElem.EnumerateArray().ToList();
                        if (coords.Count >= 2)
                        {
                            pts.Add(new XYZ(coords[0].GetDouble() * MmToFeet, coords[1].GetDouble() * MmToFeet, elevation));
                        }
                    }
                }
            }

            if (pts.Count > 2 && pts[0].DistanceTo(pts[pts.Count - 1]) < MinDimFeet)
            {
                pts.RemoveAt(pts.Count - 1);
            }
        }
        catch (Exception ex)
        {
            WriteErrorLog($"ParseBoundaryPolygon error: {ex.Message}");
        }

        return pts;
    }

    private static bool TryCreateColumn(Document doc, SidecarSegment seg, out ElementId colId)
    {
        colId = ElementId.InvalidElementId;
        try
        {
            var bb = seg.BoundingBox;
            double z0 = seg.StoreyBaseMm * MmToFeet;
            double z1 = (seg.StoreyTopMm > seg.StoreyBaseMm ? seg.StoreyTopMm : bb.MaxZ) * MmToFeet;
            double height = z1 - z0;
            if (height < MinDimFeet) height = (bb.MaxZ - bb.MinZ) * MmToFeet;
            if (height < MinDimFeet) return false;

            var level = GetNearestLevel(doc, z0);
            if (level == null) return false;

            var colSymbol = new FilteredElementCollector(doc)
                .OfCategory(BuiltInCategory.OST_StructuralColumns)
                .OfClass(typeof(FamilySymbol))
                .Cast<FamilySymbol>()
                .FirstOrDefault();

            if (colSymbol != null)
            {
                if (!colSymbol.IsActive)
                {
                    colSymbol.Activate();
                    doc.Regenerate();
                }

                XYZ location = new XYZ(seg.Centroid.X * MmToFeet, seg.Centroid.Y * MmToFeet, level.Elevation);
                var colInstance = doc.Create.NewFamilyInstance(location, colSymbol, level, Autodesk.Revit.DB.Structure.StructuralType.Column);
                if (colInstance != null)
                {
                    colId = colInstance.Id;

                    double baseOffset = z0 - level.Elevation;
                    var baseParam = colInstance.get_Parameter(BuiltInParameter.FAMILY_BASE_LEVEL_OFFSET_PARAM);
                    if (baseParam != null && !baseParam.IsReadOnly)
                    {
                        try { baseParam.Set(baseOffset); } catch { }
                    }

                    var topLevelParam = colInstance.get_Parameter(BuiltInParameter.FAMILY_TOP_LEVEL_PARAM);
                    var topLevel = GetNearestLevel(doc, z1);
                    if (topLevelParam != null && !topLevelParam.IsReadOnly && topLevel != null && topLevel.Id != level.Id)
                    {
                        try { topLevelParam.Set(topLevel.Id); } catch { }
                        var topOffsetParam = colInstance.get_Parameter(BuiltInParameter.FAMILY_TOP_LEVEL_OFFSET_PARAM);
                        if (topOffsetParam != null && !topOffsetParam.IsReadOnly)
                        {
                            try { topOffsetParam.Set(z1 - topLevel.Elevation); } catch { }
                        }
                    }

                    WriteSegmentParam(colInstance, seg);
                    return true;
                }
            }
            return false;
        }
        catch (Exception ex)
        {
            WriteErrorLog($"TryCreateColumn failed for segment {seg.SegmentId}: {ex.Message}");
            return false;
        }
    }


    private static bool TryCreatePipe(Document doc, SidecarSegment seg, out ElementId pipeId)
    {
        pipeId = ElementId.InvalidElementId;
        try
        {
            if (!TryGetCylinderEndpoints(seg, out var start, out var end))
                return false;

            if (start.DistanceTo(end) < MinDimFeet)
                return false;

            var systemType = new FilteredElementCollector(doc)
                .OfClass(typeof(PipingSystemType)).Cast<PipingSystemType>().FirstOrDefault();
            var pipeType = new FilteredElementCollector(doc)
                .OfClass(typeof(PipeType)).Cast<PipeType>().FirstOrDefault();
            var level = GetNearestLevel(doc, Math.Min(start.Z, end.Z));

            if (systemType == null || pipeType == null || level == null)
                return false;

            var pipe = Pipe.Create(doc, systemType.Id, pipeType.Id, level.Id, start, end);

            double diameterMm = Math.Min(seg.WidthMm, seg.DepthMm);
            if (TryGetTagDouble(seg, "fitted_radius_mm", out var fittedRadiusMm) && fittedRadiusMm > 0)
                diameterMm = fittedRadiusMm * 2.0;
            else if (TryGetTagDouble(seg, "diameter_mm", out var taggedDiameterMm) && taggedDiameterMm > 0)
                diameterMm = taggedDiameterMm;

            double diameterFt = Math.Max(diameterMm * MmToFeet, MinDimFeet);
            var diameterParam = pipe.get_Parameter(BuiltInParameter.RBS_PIPE_DIAMETER_PARAM);
            diameterParam?.Set(diameterFt);

            pipeId = pipe.Id;
            WriteSegmentParam(pipe, seg);
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool TryCreateDuct(Document doc, SidecarSegment seg, out ElementId ductId)
    {
        ductId = ElementId.InvalidElementId;
        try
        {
            if (!TryGetCylinderEndpoints(seg, out var start, out var end))
                return false;
            if (start.DistanceTo(end) < MinDimFeet)
                return false;

            var ductType = new FilteredElementCollector(doc)
                .OfClass(typeof(DuctType)).Cast<DuctType>().FirstOrDefault();
            var sysType = new FilteredElementCollector(doc)
                .OfClass(typeof(MechanicalSystemType)).Cast<MechanicalSystemType>().FirstOrDefault();
            var level = GetNearestLevel(doc, Math.Min(start.Z, end.Z));
            if (ductType == null || sysType == null || level == null)
                return false;

            var duct = Duct.Create(doc, ductType.Id, sysType.Id, level.Id, start, end);

            var spans = new[] { seg.WidthMm, seg.DepthMm, seg.HeightMm }.OrderBy(v => v).ToArray();
            var widthFt = Math.Max(spans[1] * MmToFeet, 100.0 * MmToFeet);
            var heightFt = Math.Max(spans[0] * MmToFeet, 100.0 * MmToFeet);
            duct.get_Parameter(BuiltInParameter.RBS_CURVE_WIDTH_PARAM)?.Set(widthFt);
            duct.get_Parameter(BuiltInParameter.RBS_CURVE_HEIGHT_PARAM)?.Set(heightFt);

            ductId = duct.Id;
            WriteSegmentParam(duct, seg);
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool TryCreateConduit(Document doc, SidecarSegment seg, out ElementId conduitId)
    {
        conduitId = ElementId.InvalidElementId;
        try
        {
            if (!TryGetCylinderEndpoints(seg, out var start, out var end))
                return false;
            if (start.DistanceTo(end) < MinDimFeet)
                return false;

            var conduitType = new FilteredElementCollector(doc)
                .OfClass(typeof(ConduitType)).Cast<ConduitType>().FirstOrDefault();
            var level = GetNearestLevel(doc, Math.Min(start.Z, end.Z));
            if (conduitType == null || level == null)
                return false;

            var conduit = Conduit.Create(doc, conduitType.Id, start, end, level.Id);
            double diameterMm = Math.Max(Math.Min(seg.WidthMm, seg.DepthMm), 20.0);
            if (TryGetTagDouble(seg, "fitted_radius_mm", out var fittedRadiusMm) && fittedRadiusMm > 0)
                diameterMm = Math.Max(fittedRadiusMm * 2.0, 20.0);
            else if (TryGetTagDouble(seg, "diameter_mm", out var taggedDiameterMm) && taggedDiameterMm > 0)
                diameterMm = Math.Max(taggedDiameterMm, 20.0);
            conduit.get_Parameter(BuiltInParameter.RBS_CONDUIT_DIAMETER_PARAM)?.Set(diameterMm * MmToFeet);

            conduitId = conduit.Id;
            WriteSegmentParam(conduit, seg);
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool TryCreateCableTray(Document doc, SidecarSegment seg, out ElementId trayId)
    {
        trayId = ElementId.InvalidElementId;
        try
        {
            if (!TryGetCylinderEndpoints(seg, out var start, out var end))
                return false;
            if (start.DistanceTo(end) < MinDimFeet)
                return false;

            var trayType = new FilteredElementCollector(doc)
                .OfClass(typeof(CableTrayType)).Cast<CableTrayType>().FirstOrDefault();
            var level = GetNearestLevel(doc, Math.Min(start.Z, end.Z));
            if (trayType == null || level == null)
                return false;

            var tray = CableTray.Create(doc, trayType.Id, start, end, level.Id);
            var spans = new[] { seg.WidthMm, seg.DepthMm, seg.HeightMm }.OrderBy(v => v).ToArray();
            double widthMm = Math.Max(spans[1], 50.0);
            double heightMm = Math.Max(spans[0], 25.0);

            if (TryGetTagDouble(seg, "width_mm", out var taggedWidthMm) && taggedWidthMm > 0)
                widthMm = taggedWidthMm;
            if (TryGetTagDouble(seg, "height_mm", out var taggedHeightMm) && taggedHeightMm > 0)
                heightMm = taggedHeightMm;

            tray.get_Parameter(BuiltInParameter.RBS_CABLETRAY_WIDTH_PARAM)?.Set(Math.Max(widthMm * MmToFeet, 50.0 * MmToFeet));
            tray.get_Parameter(BuiltInParameter.RBS_CABLETRAY_HEIGHT_PARAM)?.Set(Math.Max(heightMm * MmToFeet, 25.0 * MmToFeet));

            trayId = tray.Id;
            WriteSegmentParam(tray, seg);
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool TryCreateNativeDoorOrOpening(Document doc, SidecarSegment seg, bool isDoor, out ElementId elementId)
    {
        elementId = ElementId.InvalidElementId;
        try
        {
            var bb = seg.BoundingBox;
            double z0 = bb.MinZ * MmToFeet;
            double z1 = bb.MaxZ * MmToFeet;
            var level = GetNearestLevel(doc, z0);
            if (level == null) return false;

            XYZ insertPt = new XYZ(seg.Centroid.X * MmToFeet, seg.Centroid.Y * MmToFeet, z0);

            var walls = new FilteredElementCollector(doc)
                .OfClass(typeof(Wall))
                .Cast<Wall>()
                .ToList();

            Wall? hostWall = null;
            double minDist = double.MaxValue;
            foreach (var w in walls)
            {
                if (w.Location is LocationCurve lc)
                {
                    double dist = lc.Curve.Distance(insertPt);
                    if (dist < minDist && dist < 3.0)
                    {
                        minDist = dist;
                        hostWall = w;
                    }
                }
            }

            if (hostWall == null) return false;

            var cat = isDoor ? BuiltInCategory.OST_Doors : BuiltInCategory.OST_Windows;
            var symbol = new FilteredElementCollector(doc)
                .OfCategory(cat)
                .OfClass(typeof(FamilySymbol))
                .Cast<FamilySymbol>()
                .FirstOrDefault();

            if (symbol != null)
            {
                if (!symbol.IsActive)
                {
                    symbol.Activate();
                    doc.Regenerate();
                }

                XYZ fitPt = insertPt;
                if (hostWall.Location is LocationCurve hostLc)
                {
                    var proj = hostLc.Curve.Project(insertPt);
                    if (proj != null)
                    {
                        fitPt = new XYZ(proj.XYZPoint.X, proj.XYZPoint.Y, z0);
                    }
                }

                var instance = doc.Create.NewFamilyInstance(fitPt, symbol, hostWall, level, Autodesk.Revit.DB.Structure.StructuralType.NonStructural);
                if (instance != null)
                {
                    elementId = instance.Id;
                    WriteSegmentParam(instance, seg);
                    return true;
                }
            }

            // Native wall opening rectangular cut
            XYZ p0 = new XYZ(bb.MinX * MmToFeet, bb.MinY * MmToFeet, z0);
            XYZ p1 = new XYZ(bb.MaxX * MmToFeet, bb.MaxY * MmToFeet, z1);
            var opening = doc.Create.NewOpening(hostWall, p0, p1);
            if (opening != null)
            {
                elementId = opening.Id;
                return true;
            }

            return false;
        }
        catch
        {
            return false;
        }
    }

    private static bool TryCreateAccessoryInstance(Document doc, SidecarSegment seg, BuiltInCategory category, out ElementId id)
    {
        id = ElementId.InvalidElementId;
        try
        {
            var symbol = new FilteredElementCollector(doc)
                .OfClass(typeof(FamilySymbol))
                .OfCategory(category)
                .Cast<FamilySymbol>()
                .FirstOrDefault();
            if (symbol == null)
                return false;

            var level = GetNearestLevel(doc, seg.BoundingBox.MinZ * MmToFeet);
            if (level == null)
                return false;

            if (!symbol.IsActive)
                symbol.Activate();

            var pt = new XYZ(seg.Centroid.X * MmToFeet, seg.Centroid.Y * MmToFeet, seg.Centroid.Z * MmToFeet);
            var inst = doc.Create.NewFamilyInstance(pt, symbol, level, Autodesk.Revit.DB.Structure.StructuralType.NonStructural);
            id = inst.Id;
            WriteSegmentParam(inst, seg);
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool TryCreateEquipmentInstance(Document doc, SidecarSegment seg, BuiltInCategory category, out ElementId id)
    {
        id = ElementId.InvalidElementId;
        try
        {
            var symbol = new FilteredElementCollector(doc)
                .OfClass(typeof(FamilySymbol))
                .OfCategory(category)
                .Cast<FamilySymbol>()
                .FirstOrDefault();
            if (symbol == null)
                return false;

            var level = GetNearestLevel(doc, seg.BoundingBox.MinZ * MmToFeet);
            if (level == null)
                return false;

            if (!symbol.IsActive)
                symbol.Activate();

            var pt = new XYZ(seg.Centroid.X * MmToFeet, seg.Centroid.Y * MmToFeet, seg.BoundingBox.MinZ * MmToFeet);
            var inst = doc.Create.NewFamilyInstance(pt, symbol, level, Autodesk.Revit.DB.Structure.StructuralType.NonStructural);
            id = inst.Id;
            WriteSegmentParam(inst, seg);
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static Level? GetNearestLevel(Document doc, double elevationFt)
    {
        var allLevels = new FilteredElementCollector(doc)
            .OfClass(typeof(Level))
            .Cast<Level>()
            .ToList();

        if (allLevels.Count == 0)
        {
            try
            {
                var created = Level.Create(doc, elevationFt);
                if (created != null)
                {
                    try
                    {
                        created.Name = $"Scan Level {elevationFt * 304.8:F0}mm";
                    }
                    catch
                    {
                        // Naming can fail when another level already uses that name.
                    }

                    WriteErrorLog($"GetNearestLevel: created initial level at {elevationFt:0.###} ft ({elevationFt * 304.8:F0} mm)");
                    return created;
                }
            }
            catch (Exception ex)
            {
                WriteErrorLog($"GetNearestLevel: failed to create initial level at {elevationFt:0.###} ft: {ex.Message}");
            }

            return null;
        }

        // Snap to existing level if within 3.5 feet (~1.0 m)
        double snapToleranceFt = 3.5;
        var closeLevels = allLevels
            .Where(l => Math.Abs(l.Elevation - elevationFt) <= snapToleranceFt)
            .OrderBy(l => Math.Abs(l.Elevation - elevationFt))
            .ToList();

        if (closeLevels.Count > 0)
        {
            return closeLevels.First();
        }

        // If no existing level is close enough, create a dedicated storey level
        try
        {
            var created = Level.Create(doc, elevationFt);
            if (created != null)
            {
                try
                {
                    created.Name = $"Scan Level {elevationFt * 304.8:F0}mm";
                }
                catch
                {
                    try
                    {
                        created.Name = $"Scan Level {elevationFt * 304.8:F0}mm_{Guid.NewGuid().ToString("N")[..4]}";
                    }
                    catch { }
                }

                WriteErrorLog($"GetNearestLevel: created storey level at {elevationFt:0.###} ft ({elevationFt * 304.8:F0} mm)");
                return created;
            }
        }
        catch (Exception ex)
        {
            WriteErrorLog($"GetNearestLevel: failed to create level at {elevationFt:0.###} ft: {ex.Message}");
        }

        return allLevels.OrderBy(l => Math.Abs(l.Elevation - elevationFt)).FirstOrDefault();
    }

    private static bool IsYellowPipeColor(SidecarSegment seg)
    {
        if (seg.Tags == null)
            return false;

        object? colorObj = null;
        if (seg.Tags.TryGetValue("trace_color_hex", out var traceObj) && traceObj != null)
            colorObj = traceObj;
        else if (seg.Tags.TryGetValue("dominant_color", out var domObj) && domObj != null)
            colorObj = domObj;

        if (colorObj == null)
            return false;

        var color = colorObj switch
        {
            JsonElement e when e.ValueKind == JsonValueKind.String => e.GetString(),
            _ => colorObj.ToString(),
        };

        if (string.IsNullOrWhiteSpace(color))
            return false;

        var c = color.ToUpperInvariant();
        return c == "#FFFF00" || c == "#FFC000";
    }

    private static Wall? FindNearestWall(Document doc, XYZ point, double toleranceFt)
    {
        return new FilteredElementCollector(doc)
            .OfClass(typeof(Wall))
            .Cast<Wall>()
            .Where(w => w.Location is LocationCurve lc && lc.Curve.Distance(point) <= toleranceFt)
            .OrderBy(w => ((LocationCurve)w.Location).Curve.Distance(point))
            .FirstOrDefault();
    }

    private static XYZ AlignPointToWallHost(Wall wall, XYZ desiredPoint)
    {
        if (wall.Location is not LocationCurve lc)
            return desiredPoint;

        var projection = lc.Curve.Project(new XYZ(desiredPoint.X, desiredPoint.Y, lc.Curve.GetEndPoint(0).Z));
        var onCurve = projection?.XYZPoint ?? desiredPoint;

        var bb = wall.get_BoundingBox(null);
        if (bb == null)
            return new XYZ(onCurve.X, onCurve.Y, desiredPoint.Z);

        const double insetFt = 0.02;
        double x = Math.Max(bb.Min.X + insetFt, Math.Min(bb.Max.X - insetFt, onCurve.X));
        double y = Math.Max(bb.Min.Y + insetFt, Math.Min(bb.Max.Y - insetFt, onCurve.Y));
        double z = Math.Max(bb.Min.Z + insetFt, Math.Min(bb.Max.Z - insetFt, desiredPoint.Z));
        return new XYZ(x, y, z);
    }


    private static bool TryCreateStair(Document doc, SidecarSegment seg, out ElementId stairId)
    {
        stairId = ElementId.InvalidElementId;
        try
        {
            var geom = BuildSteppedStairGeometry(seg);
            if (geom == null || geom.Count == 0)
                return false;

            // Prefer stair category, but fall back to GenericModel if the
            // project/category rejects stair DirectShape creation.
            try
            {
                var ds = DirectShape.CreateElement(doc, new ElementId(BuiltInCategory.OST_Stairs));
                ds.SetShape(geom);
                WriteSegmentParam(ds, seg);
                stairId = ds.Id;
                return true;
            }
            catch
            {
                var ds = DirectShape.CreateElement(doc, new ElementId(BuiltInCategory.OST_GenericModel));
                ds.SetShape(geom);
                WriteSegmentParam(ds, seg);
                stairId = ds.Id;
                return true;
            }
        }
        catch
        {
            return false;
        }
    }

    private static IList<GeometryObject> BuildSteppedStairGeometry(SidecarSegment seg)
    {
        var bb = seg.BoundingBox;

        double minX = bb.MinX * MmToFeet;
        double maxX = bb.MaxX * MmToFeet;
        double minY = bb.MinY * MmToFeet;
        double maxY = bb.MaxY * MmToFeet;
        double minZ = bb.MinZ * MmToFeet;
        double maxZ = bb.MaxZ * MmToFeet;

        // Use reconstructed storey levels if available for multi-level stairs
        bool hasLevelSpan = false;
        if (seg.Tags != null)
        {
            if (TryGetTagDouble(seg, "reconstructed_storey_base_mm", out var baseZ) &&
                TryGetTagDouble(seg, "reconstructed_storey_top_mm", out var topZ))
            {
                minZ = baseZ * MmToFeet;
                maxZ = topZ * MmToFeet;
                hasLevelSpan = true;
            }
        }

        double runXRaw = Math.Max(maxX - minX, 0.0);
        double runYRaw = Math.Max(maxY - minY, 0.0);
        double riseRaw = Math.Max(maxZ - minZ, 0.0);

        double runX = Math.Max(runXRaw, 900.0 * MmToFeet);
        double runY = Math.Max(runYRaw, 900.0 * MmToFeet);
        double rise = Math.Max(riseRaw, 600.0 * MmToFeet);

        bool alongX = runX >= runY;
        double runLength = alongX ? runX : runY;
        double runWidth = Math.Max(alongX ? runY : runX, 900.0 * MmToFeet);
        runWidth = Math.Min(runWidth, 3500.0 * MmToFeet);
        runWidth *= VisualScaleXY;

        int stepCount = 0;
        bool hasTaggedSteps = TryGetTagDouble(seg, "stair_steps", out var taggedSteps) && taggedSteps > 0;
        if (hasTaggedSteps)
            stepCount = (int)Math.Round(taggedSteps);
        if (stepCount <= 0)
            stepCount = (int)Math.Round(rise / (170.0 * MmToFeet));
        // Prevent undercounted steps from collapsing the stair into a flattened wedge.
        // If scan tags report too few steps for the measured rise, enforce a realistic minimum.
        int minStepCountByRise = (int)Math.Ceiling(riseRaw / (210.0 * MmToFeet));
        if (!hasTaggedSteps && minStepCountByRise > 0)
            stepCount = Math.Max(stepCount, minStepCountByRise);
        stepCount = Math.Max(3, Math.Min(stepCount, 40));

        // Keep proportions near real stair ergonomics even when AABB extents are noisy.
        double minModeledRise = stepCount * 150.0 * MmToFeet;
        rise = Math.Max(rise, minModeledRise);
        
        // For multi-level stairs, use the computed rise directly without extra scaling
        if (!hasLevelSpan)
            rise *= VisualScaleZ;

        double runLengthRaw = alongX ? runXRaw : runYRaw;
        double minRunLength = stepCount * 240.0 * MmToFeet;
        double maxRunLength = stepCount * 450.0 * MmToFeet;
        runLength = Math.Max(runLength, minRunLength);
        runLength = Math.Min(runLength, maxRunLength);
        if (runLengthRaw > 0)
            runLength = Math.Min(Math.Max(runLength, runLengthRaw), maxRunLength);
        runLength *= VisualScaleXY;

        // Use centroid as the starting point for stairs from same location
        double centerX = seg.Centroid.X * MmToFeet;
        double centerY = seg.Centroid.Y * MmToFeet;
        double axisStart = alongX
            ? centerX - runLength / 2.0
            : centerY - runLength / 2.0;

        double treadDepth = runLength / stepCount;
        double riser = rise / stepCount;

        var solids = new List<GeometryObject>(stepCount);
        for (int i = 0; i < stepCount; i++)
        {
            double x0 = alongX ? axisStart + i * treadDepth : centerX - runWidth / 2.0;
            double x1 = alongX ? axisStart + (i + 1) * treadDepth : centerX + runWidth / 2.0;
            double y0 = alongX ? centerY - runWidth / 2.0 : axisStart + i * treadDepth;
            double y1 = alongX ? centerY + runWidth / 2.0 : axisStart + (i + 1) * treadDepth;
            double z0 = minZ + i * riser;
            double z1 = z0 + riser;

            var solid = BuildRectSolid(x0, y0, z0, x1, y1, z1);
            if (solid != null)
                solids.Add(solid);
        }

        return solids;
    }

    private static Solid? BuildRectSolid(double x0, double y0, double z0, double x1, double y1, double z1)
    {
        double width = x1 - x0;
        double depth = y1 - y0;
        double height = z1 - z0;
        if (width <= 0.0 || depth <= 0.0 || height <= 0.0)
            return null;

        var loop = new CurveLoop();
        loop.Append(Line.CreateBound(new XYZ(0, 0, 0), new XYZ(width, 0, 0)));
        loop.Append(Line.CreateBound(new XYZ(width, 0, 0), new XYZ(width, depth, 0)));
        loop.Append(Line.CreateBound(new XYZ(width, depth, 0), new XYZ(0, depth, 0)));
        loop.Append(Line.CreateBound(new XYZ(0, depth, 0), new XYZ(0, 0, 0)));

        var solid = GeometryCreationUtilities.CreateExtrusionGeometry(
            new List<CurveLoop> { loop },
            XYZ.BasisZ,
            height);

        var move = Transform.CreateTranslation(new XYZ(x0, y0, z0));
        return SolidUtils.CreateTransformed(solid, move);
    }

    private static long MapToCategory(string elementType)
    {
        // TODO: Map elementType string to Revit BuiltInCategory value
        // Example: return (long)BuiltInCategory.OST_Walls;
        return (long)BuiltInCategory.OST_GenericModel;
    }

    private static bool IsLargeTankSegment(SidecarSegment seg)
    {
        if (seg == null) return false;

        if (_selectedTankIds.Count > 0)
        {
            return _selectedTankIds.Contains(seg.SegmentId);
        }

        // Generic matches: any cylinder that is large and vertical (diameter >= _maxCylDiameter * 0.8 and height > 1000mm)
        var shape = (seg.Shape ?? "").ToLowerInvariant();
        if (shape == "cylinder" || shape == "tanks")
        {
            double diameter = Math.Min(seg.WidthMm, seg.DepthMm);
            if (_maxCylDiameter > 0 && diameter >= _maxCylDiameter * 0.8 && seg.HeightMm > 1000.0)
            {
                return true;
            }
        }

        return false;
    }

    private static bool IsVerticalCylinderCandidate(SidecarSegment seg)
    {
        if (seg == null) return false;
        var shape = (seg.Shape ?? "").ToLowerInvariant();
        if (shape == "cylinder" || shape == "tanks")
        {
            double diameter = Math.Min(seg.WidthMm, seg.DepthMm);
            // Large silos or tanks have diameter >= 800mm and height > 1000mm
            if (diameter >= 800.0 && seg.HeightMm > 1000.0)
            {
                return true;
            }
        }
        return false;
    }


    private static IList<GeometryObject> BuildTankCylinderGeometry(SidecarSegment seg)
    {
        // All values come from the Python backend dynamically.
        // min_z = floor level (base of tank), max_z = top of the tank.
        double botZ = seg.BoundingBox.MinZ;
        double topZ = seg.BoundingBox.MaxZ;

        // Radius = half of the smaller horizontal dimension, at least 400 mm.
        double radiusMm = Math.Max(Math.Min(seg.WidthMm, seg.DepthMm) / 2.0, 400.0);

        // Prefer the fitted_radius_mm tag if the Python side provided it.
        if (seg.Tags != null &&
            seg.Tags.TryGetValue("fitted_radius_mm", out var fitRadObj) &&
            double.TryParse(fitRadObj?.ToString(), System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out double fitRad) &&
            fitRad > 200.0)
        {
            radiusMm = fitRad;
        }

        // Align with taller tank to keep identical heights is disabled to prevent overriding actual scanned dimensions and positioning.

        var start = new XYZ(seg.Centroid.X * MmToFeet, seg.Centroid.Y * MmToFeet, botZ * MmToFeet);
        var end   = new XYZ(seg.Centroid.X * MmToFeet, seg.Centroid.Y * MmToFeet, topZ * MmToFeet);
        double r = radiusMm * MmToFeet;

        var solid = RevitElementFactory.BuildCylinderSolid(start, end, r);
        if (solid == null)
            return Array.Empty<GeometryObject>();

        return new List<GeometryObject> { solid };
    }


    private static IList<GeometryObject> BuildFloorGeometry(SidecarBoundingBox bb)
    {
        double x0 = bb.MinX * MmToFeet;
        double y0 = bb.MinY * MmToFeet;
        double z0 = bb.MinZ * MmToFeet;
        double x1 = bb.MaxX * MmToFeet;
        double y1 = bb.MaxY * MmToFeet;
        double z1 = bb.MaxZ * MmToFeet;

        var solid = BuildRectSolid(x0, y0, z0, x1, y1, z1);
        if (solid == null)
            return Array.Empty<GeometryObject>();

        return new List<GeometryObject> { solid };
    }

    private static void AlignHorizontalPipeToFloor(SidecarSegment seg, double floorMinX, double floorMaxX)
    {
        if (seg == null) return;
        if (!TryGetCylinderEndpoints(seg, out var start, out var end))
            return;

        var axis = (end - start).Normalize();
        double dotX = Math.Abs(axis.DotProduct(XYZ.BasisX));
        if (dotX <= 0.8)
            return;

        double span = Math.Abs(end.X - start.X);
        double floorSpan = Math.Max(floorMaxX - floorMinX, 0.0) * MmToFeet;
        if (floorSpan <= 0.0 || span < Math.Min(0.5 * floorSpan, 0.5))
            return;

        if (seg.BoundingBox != null)
        {
            seg.BoundingBox.MinX = floorMinX;
            seg.BoundingBox.MaxX = floorMaxX;
        }
        if (seg.Tags != null)
        {
            seg.Tags["cyl_start_x_mm"] = floorMinX;
            seg.Tags["cyl_end_x_mm"] = floorMaxX;
        }
        if (seg.Centroid != null)
        {
            seg.Centroid.X = (floorMinX + floorMaxX) * 0.5;
        }
    }

    private static void AlignVerticalPipeToWallHeight(SidecarSegment seg, double floorZ, double wallMaxZ)
    {
        if (seg == null) return;
        if (!TryGetCylinderEndpoints(seg, out var start, out var end))
            return;

        var axis = (end - start).Normalize();
        double dotZ = Math.Abs(axis.DotProduct(XYZ.BasisZ));
        if (dotZ <= 0.8)
            return;

        double span = Math.Abs(end.Z - start.Z);
        double wallSpan = Math.Max(wallMaxZ - floorZ, 0.0) * MmToFeet;
        if (wallSpan <= 0.0 || span < Math.Min(0.5 * wallSpan, 0.5))
            return;

        if (seg.BoundingBox != null)
        {
            seg.BoundingBox.MinZ = floorZ;
            seg.BoundingBox.MaxZ = wallMaxZ;
        }
        if (seg.Tags != null)
        {
            seg.Tags["cyl_start_z_mm"] = floorZ;
            seg.Tags["cyl_end_z_mm"] = wallMaxZ;
        }
        if (seg.Centroid != null)
        {
            seg.Centroid.Z = (floorZ + wallMaxZ) * 0.5;
        }
    }


    private static void WriteErrorLog(string message)
    {
        try
        {
            var errorLogPath = System.IO.Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "ScanToBIM", "Results", "native_build_errors.log");
            System.IO.File.AppendAllText(errorLogPath, $"[{DateTime.Now}] {message}\n\n");
        }
        catch {}
    }

}
