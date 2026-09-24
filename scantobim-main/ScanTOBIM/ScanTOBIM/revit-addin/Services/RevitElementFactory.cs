// ─────────────────────────────────────────────────────────────────────────────
// RevitElementFactory.cs
//
// Creates Revit elements (Wall, Floor, Column) from ElementInstructions
// received from the Python AI agent.
//
// SAFETY CONTRACT — non-negotiable execution order for every create method:
//   1. SafetyGateService.AssertNotSC1(instruction)   ← SC1 hard block
//   2. SafetyGateService.AssertSC2HasApproval(...)   ← SC2 signature check
//   3. Open Transaction
//   4. Create element via Revit API
//   5. SafetyGateService.WriteComplianceParameters() ← write 4 shared params
//   6. Commit Transaction
//
// All methods MUST be called on the Revit main thread.
// All dimensions from the agent are in millimetres; Revit uses decimal feet.
// ─────────────────────────────────────────────────────────────────────────────

using System.Collections.Generic;
using System.Globalization;
using System.Diagnostics;
using System.Linq;
using Autodesk.Revit.DB;
using Autodesk.Revit.DB.Architecture;
using Autodesk.Revit.DB.Electrical;
using Autodesk.Revit.DB.Mechanical;
using Autodesk.Revit.DB.Plumbing;
using Autodesk.Revit.DB.Structure;
using ScanToBIM.Models;
using ElementType = ScanToBIM.Models.ElementType;
using Element     = Autodesk.Revit.DB.Element;
using Wall        = Autodesk.Revit.DB.Wall;
using Floor       = Autodesk.Revit.DB.Floor;
using Ceiling     = Autodesk.Revit.DB.Ceiling;
using ScanToBIM.SafetyGate;

namespace ScanToBIM.Services;

public sealed class RevitElementFactory
{
    private readonly Document      _doc;
    private readonly FamilyResolver _resolver;
    private readonly double? _envZScale;
    private readonly double? _envZOffsetMm;
    private double _instructionZScale = 1.0;
    private double _instructionZOffsetMm = 0.0;
    private static readonly HashSet<ElementType> _enclosedShellTypes = new()
    {
        ElementType.tank,
        ElementType.hvac_equipment,
        ElementType.pressure_vessel,
        ElementType.heat_exchanger,
        ElementType.compressor,
        ElementType.pressurizer,
        ElementType.steam_generator,
        ElementType.emergency_diesel_generator,
        ElementType.transformer,
        ElementType.switchgear,
        ElementType.ups_system,
    };

    public RevitElementFactory(Document doc, string? familyMapPath = null)
    {
        _doc      = doc ?? throw new ArgumentNullException(nameof(doc));
        _resolver = new FamilyResolver(doc, familyMapPath);
        _envZScale = ReadVerticalScaleFromEnv();
        _envZOffsetMm = ReadVerticalOffsetFromEnv();
    }

    // ── Public dispatcher ─────────────────────────────────────────────────────

    /// <summary>
    /// Routes an ElementInstruction to the correct create method.
    /// Enforces SC1 hard block and SC2 approval gate before any Revit API call.
    /// Returns an ActionResult regardless of success or failure — never throws to caller.
    /// </summary>
    public ActionResult Execute(ElementInstruction instruction)
    {
        var sw = Stopwatch.StartNew();
        try
        {
            // ── SAFETY GATES — ORDER IS NON-NEGOTIABLE ────────────────────────
            SafetyGateService.AssertNotSC1(instruction);
            SafetyGateService.AssertSC2HasApproval(instruction);
            // ──────────────────────────────────────────────────────────────────
            _instructionZScale = ResolveInstructionZScale(instruction);
            _instructionZOffsetMm = ResolveInstructionZOffsetMm(instruction);

            var revitId = instruction.ElementType switch
            {
                ElementType.wall    => CreateWall(instruction),
                ElementType.floor   => CreateFloor(instruction),
                ElementType.ceiling => CreateCeiling(instruction),
                ElementType.column  => CreateColumn(instruction),
                ElementType.beam    => CreateBeam(instruction),
                ElementType.stair   => CreateStair(instruction),
                ElementType.ramp    => CreateRamp(instruction),
                ElementType.door           => CreateDoor(instruction),
                ElementType.window         => CreateWindow(instruction),
                ElementType.railing        => CreateRailing(instruction),
                ElementType.duct           => CreateDuct(instruction),
                ElementType.conduit        => CreateConduit(instruction),
                ElementType.cable_tray     => CreateCableTray(instruction),
                ElementType.hvac_equipment   => CreateMepEquipment(instruction, BuiltInCategory.OST_MechanicalEquipment),
                ElementType.tank             => CreateMepEquipment(instruction, BuiltInCategory.OST_MechanicalEquipment),
                ElementType.pump             => CreateMepEquipment(instruction, BuiltInCategory.OST_MechanicalEquipment),
                ElementType.valve            => CreateValve(instruction),
                ElementType.sprinkler        => CreateFireFixture(instruction),
                ElementType.drainage         => CreatePlumbingFixture(instruction),
                ElementType.slab_opening     => CreateSlabOpening(instruction),
                ElementType.electrical_panel => CreateElectricalPanel(instruction),
                ElementType.kerb             => CreateKerb(instruction),
                // ── New structural / access types ─────────────────────────────
                ElementType.ladder           => CreateGenericStructural(instruction, "Ladder"),
                ElementType.grating          => CreateFloor(instruction),       // flat grating → floor
                ElementType.overhead_crane   => CreateGenericStructural(instruction, "Overhead Crane"),
                ElementType.hatch            => CreateSlabOpening(instruction), // small void in slab
                ElementType.trench           => CreateSlabOpening(instruction), // elongated slab void
                ElementType.bund_wall        => CreateKerb(instruction),        // low wall
                ElementType.seismic_isolator => CreateMepEquipment(instruction, BuiltInCategory.OST_MechanicalEquipment),
                ElementType.containment_penetration => CreateSlabOpening(instruction),
                ElementType.penetration_seal        => CreateSlabOpening(instruction),
                // ── New MEP inline types ───────────────────────────────────────
                // ElementType.pipe                => CreatePipe(instruction),
                ElementType.safety_relief_valve => CreateValve(instruction),
                ElementType.strainer            => CreateValve(instruction),
                ElementType.expansion_joint     => CreateValve(instruction),
                ElementType.fire_damper         => CreateDuct(instruction),    // inline in duct
                ElementType.pipe_support        => CreateGenericStructural(instruction, "Pipe Support"),
                // ── New MEP vessels / equipment ───────────────────────────────
                ElementType.pressure_vessel  => CreateMepEquipment(instruction, BuiltInCategory.OST_MechanicalEquipment),
                ElementType.heat_exchanger   => CreateMepEquipment(instruction, BuiltInCategory.OST_MechanicalEquipment),
                ElementType.compressor       => CreateMepEquipment(instruction, BuiltInCategory.OST_MechanicalEquipment),
                ElementType.pressurizer      => CreateMepEquipment(instruction, BuiltInCategory.OST_MechanicalEquipment),
                ElementType.steam_generator  => CreateMepEquipment(instruction, BuiltInCategory.OST_MechanicalEquipment),
                ElementType.emergency_diesel_generator => CreateMepEquipment(instruction, BuiltInCategory.OST_MechanicalEquipment),
                ElementType.radiation_monitor => CreateElectricalPanel(instruction),
                // ── New electrical types ───────────────────────────────────────
                ElementType.transformer    => CreateMepEquipment(instruction, BuiltInCategory.OST_ElectricalEquipment),
                ElementType.switchgear     => CreateMepEquipment(instruction, BuiltInCategory.OST_ElectricalEquipment),
                ElementType.ups_system     => CreateMepEquipment(instruction, BuiltInCategory.OST_ElectricalEquipment),
                ElementType.junction_box   => CreateElectricalPanel(instruction),
                ElementType.lighting_fitting => CreateFireFixture(instruction),
                // ── New fire / life safety types ──────────────────────────────
                ElementType.fire_extinguisher => CreateFireFixture(instruction),
                ElementType.fire_hydrant     => CreatePlumbingFixture(instruction),
                ElementType.deluge_valve     => CreateValve(instruction),
                ElementType.fire_alarm_panel => CreateElectricalPanel(instruction),
                ElementType.smoke_detector   => CreateFireFixture(instruction),
                ElementType.pipe             => CreatePipe(instruction),
                _ => throw new NotSupportedException(
                    $"Element type '{instruction.ElementType}' is not supported.")
            };

            // Stamp semantic metadata in a follow-up transaction so every
            // created element has consistent label/embedding fields.
            var created = _doc.GetElement(revitId);
            if (created != null)
            {
                using var txMeta = new Transaction(_doc, $"ScanToBIM: Metadata [{ShortId(instruction)}]");
                txMeta.Start();
                ApplySemanticMetadata(created, instruction);
                txMeta.Commit();
            }

            return new ActionResult
            {
                Success       = true,
                ElementId     = $"revit-{revitId.Value}",
                InstructionId = instruction.InstructionId,
                DurationMs    = sw.ElapsedMilliseconds,
            };
        }
        catch (SafetyCategoryViolationException ex)
        {
            return new ActionResult
            {
                Success       = false,
                InstructionId = instruction.InstructionId,
                Error         = $"SC1_HARD_BLOCKED: {ex.Message}",
                DurationMs    = sw.ElapsedMilliseconds,
            };
        }
        catch (SC2ApprovalRequiredException ex)
        {
            return new ActionResult
            {
                Success       = false,
                InstructionId = instruction.InstructionId,
                Error         = $"SC2_PENDING_APPROVAL: {ex.Message}",
                DurationMs    = sw.ElapsedMilliseconds,
            };
        }
        catch (Exception ex)
        {
            return new ActionResult
            {
                Success       = false,
                InstructionId = instruction.InstructionId,
                Error         = ex.Message,
                DurationMs    = sw.ElapsedMilliseconds,
            };
        }
        finally
        {
            _instructionZScale = 1.0;
            _instructionZOffsetMm = 0.0;
        }
    }


    private ElementId CreateWall(ElementInstruction instruction)
    {
        var bb = instruction.BoundingBox;
        double x0 = MmToFt(bb.MinX), y0 = MmToFt(bb.MinY);
        double x1 = MmToFt(bb.MaxX), y1 = MmToFt(bb.MaxY);

        // Semantic storey base and top (P0.4)
        double baseMm = bb.MinZ;
        if (TryGetParamDouble(instruction, "reconstructed_storey_base_mm", out var rbase))
            baseMm = rbase;
        else if (TryGetParamDouble(instruction, "source_min_z_mm", out var sminz))
            baseMm = sminz;

        double topMm = bb.MaxZ;
        if (TryGetParamDouble(instruction, "reconstructed_storey_top_mm", out var rtop))
            topMm = rtop;
        else if (TryGetParamDouble(instruction, "source_max_z_mm", out var smaxz))
            topMm = smaxz;

        double heightMm = topMm - baseMm;
        if (heightMm <= 500.0) heightMm = 3000.0;
        else if (heightMm > 6000.0) heightMm = 3500.0;

        double z0 = MmToFtZ(baseMm);
        double wallHeight = MmToFt(heightMm);

        // Resolve wall centreline
        XYZ start, end;
        string axisSource;
        double thicknessMm = 200.0;
        var axisGeom = WallGeometryHelper.TryExtract(instruction, z0);
        if (axisGeom.HasValue)
        {
            start       = axisGeom.Value.Start;
            end         = axisGeom.Value.End;
            thicknessMm = axisGeom.Value.ThicknessMm;
            axisSource  = instruction.Parameters!.ContainsKey("wall_start_x_mm") ? "ransac_explicit" : "ransac_axis";
        }
        else
        {
            double dx = bb.MaxX - bb.MinX;
            double dy = bb.MaxY - bb.MinY;
            double cx = MmToFt((bb.MinX + bb.MaxX) / 2.0);
            double cy = MmToFt((bb.MinY + bb.MaxY) / 2.0);
            if (dx >= dy)
            {
                start = new XYZ(x0, cy, z0);
                end   = new XYZ(x1, cy, z0);
            }
            else
            {
                start = new XYZ(cx, y0, z0);
                end   = new XYZ(cx, y1, z0);
            }
            axisSource = "aabb_longest_edge";
        }
        if (start.DistanceTo(end) < MmToFt(100.0))
        {
            Debug.WriteLine($"[ScanToBIM] Degenerate wall length {start.DistanceTo(end) * 304.8:F1}mm for segment {instruction.SegmentId}. Rejected.");
            return ElementId.InvalidElementId;
        }

        // Geometry cleanup (future: wall merging, opening subtraction)
        // If wall geometry is invalid, fallback to DirectShape
        try
        {
            var level    = GetOrCreateLevel(z0, "Scan Level Wall");
            var wallType = GetWallTypeForThickness(thicknessMm);
            double baseOffset = z0 - level.Elevation;
            using var tx = new Transaction(_doc, $"ScanToBIM: Wall [{ShortId(instruction)}]");
            tx.Start();
            var wall = Wall.Create(
                _doc,
                Line.CreateBound(start, end),
                wallType.Id,
                level.Id,
                wallHeight,
                /*offset*/ baseOffset,
                /*flip*/   false,
                /*structural*/ false);

            // Disallow end joins to prevent Revit geometry corruption / overlap failures (P0.4)
            WallUtils.DisallowWallJoinAtEnd(wall, 0);
            WallUtils.DisallowWallJoinAtEnd(wall, 1);

            var comments = $"AxisSource={axisSource} Angle={Math.Round(Math.Atan2(end.Y-start.Y, end.X-start.X)*(180.0/Math.PI),2)}° " +
                          $"Length={Math.Round((end-start).GetLength()*304.8,1)}mm Thickness={thicknessMm:F0}mm";
            if (instruction.Parameters?.TryGetValue("dominant_color", out var color) == true)
                comments += $" Color={color}";
            wall.get_Parameter(BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS)?.Set(comments);
            TryJoinAndCutOverlappingWalls(wall);
            SafetyGateService.WriteComplianceParameters(wall, instruction);
            tx.Commit();
            return wall.Id;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[ScanToBIM] Native Wall.Create failed for segment {instruction.SegmentId}: {ex.Message}. Proxy fabrication rejected.");
            return ElementId.InvalidElementId;
        }
    }


    // ── Floor ─────────────────────────────────────────────────────────────────


    private ElementId CreateFloor(ElementInstruction instruction)
    {
        var bb = instruction.BoundingBox;
        double z = MmToFtZ(bb.MinZ);
        var obbCorners = FloorGeometryHelper.TryExtract(instruction, z);
        XYZ[] corners;
        if (obbCorners.HasValue)
        {
            corners = new[] { obbCorners.Value.P1, obbCorners.Value.P2, obbCorners.Value.P3, obbCorners.Value.P4 };
        }
        else
        {
            double x0 = MmToFt(bb.MinX), y0 = MmToFt(bb.MinY);
            double x1 = MmToFt(bb.MaxX), y1 = MmToFt(bb.MaxY);
            corners = new[]
            {
                new XYZ(x0, y0, z),
                new XYZ(x1, y0, z),
                new XYZ(x1, y1, z),
                new XYZ(x0, y1, z),
            };
        }
        var mainLoop = CurveLoop.Create(
            corners.Select((pt, i) => (Curve)Line.CreateBound(pt, corners[(i + 1) % corners.Length])).ToList());
        // Geometry cleanup
        var cleanedMain = GeometryUtils.SimplifyCurveLoop(mainLoop);
        if (!GeometryUtils.IsValidCurveLoop(cleanedMain))
            throw new InvalidOperationException("Floor profile is invalid after cleanup.");
        // Openings (voids)
        var openingLoops = new List<CurveLoop>();
        if (instruction.Parameters != null && instruction.Parameters.TryGetValue("openings", out var openingsObj) && openingsObj is List<List<XYZ>> openingPtsList)
        {
            foreach (var pts in openingPtsList)
            {
                if (pts.Count < 3) continue;
                var curves = new List<Curve>();
                for (int i = 0; i < pts.Count; i++)
                    curves.Add(Line.CreateBound(pts[i], pts[(i + 1) % pts.Count]));
                var loop = new CurveLoop();
                foreach (var c in curves) loop.Append(c);
                var cleaned = GeometryUtils.SimplifyCurveLoop(loop);
                if (GeometryUtils.IsValidCurveLoop(cleaned))
                    openingLoops.Add(cleaned);
            }
        }
        var allLoops = GeometryUtils.SubtractOpenings(cleanedMain, openingLoops);
        var level     = GetOrCreateLevel(z, "Scan Level Floor");
        var floorType = GetDefaultFloorType();
        using var tx = new Transaction(_doc, $"ScanToBIM: Floor [{ShortId(instruction)}]");
        tx.Start();
        var floor = Floor.Create(_doc, allLoops, floorType.Id, level.Id);
        TrySetLevelOffset(floor, level, z);
        SafetyGateService.WriteComplianceParameters(floor, instruction);
        tx.Commit();
        return floor.Id;
    }

    // ── Ceiling ───────────────────────────────────────────────────────────────

    private ElementId CreateCeiling(ElementInstruction instruction)
    {
        var bb = instruction.BoundingBox;
        double z  = MmToFtZ(bb.MaxZ);
        var obbCorners = FloorGeometryHelper.TryExtract(instruction, z);
        XYZ[] corners;
        if (obbCorners.HasValue)
        {
            corners = new[] { obbCorners.Value.P1, obbCorners.Value.P2, obbCorners.Value.P3, obbCorners.Value.P4 };
        }
        else
        {
            double x0 = MmToFt(bb.MinX), y0 = MmToFt(bb.MinY);
            double x1 = MmToFt(bb.MaxX), y1 = MmToFt(bb.MaxY);
            corners = new[]
            {
                new XYZ(x0, y0, z),
                new XYZ(x1, y0, z),
                new XYZ(x1, y1, z),
                new XYZ(x0, y1, z),
            };
        }
        var mainLoop = CurveLoop.Create(
            corners.Select((pt, i) => (Curve)Line.CreateBound(pt, corners[(i + 1) % corners.Length])).ToList());
        var cleanedMain = GeometryUtils.SimplifyCurveLoop(mainLoop);
        if (!GeometryUtils.IsValidCurveLoop(cleanedMain))
            throw new InvalidOperationException("Ceiling profile is invalid after cleanup.");
        var openingLoops = new List<CurveLoop>();
        if (instruction.Parameters != null && instruction.Parameters.TryGetValue("openings", out var openingsObj) && openingsObj is List<List<XYZ>> openingPtsList)
        {
            foreach (var pts in openingPtsList)
            {
                if (pts.Count < 3) continue;
                var curves = new List<Curve>();
                for (int i = 0; i < pts.Count; i++)
                    curves.Add(Line.CreateBound(pts[i], pts[(i + 1) % pts.Count]));
                var loop = new CurveLoop();
                foreach (var c in curves) loop.Append(c);
                var cleaned = GeometryUtils.SimplifyCurveLoop(loop);
                if (GeometryUtils.IsValidCurveLoop(cleaned))
                    openingLoops.Add(cleaned);
            }
        }
        var allLoops = GeometryUtils.SubtractOpenings(cleanedMain, openingLoops);
        var level       = GetOrCreateLevel(z, "Scan Level Ceiling");
        var ceilingType = GetDefaultCeilingType();
        using var tx = new Transaction(_doc, $"ScanToBIM: Ceiling [{ShortId(instruction)}]");
        tx.Start();
        var ceiling = Ceiling.Create(_doc, allLoops, ceilingType.Id, level.Id);
        TrySetLevelOffset(ceiling, level, z);
        SafetyGateService.WriteComplianceParameters(ceiling, instruction);
        tx.Commit();
        return ceiling.Id;
    }

    // ── Beam ──────────────────────────────────────────────────────────────────

    private ElementId CreateBeam(ElementInstruction instruction)
    {
        var bb = instruction.BoundingBox;
        var c  = instruction.Centroid;

        // Beam runs along its longest horizontal axis; use bb extents to find it
        double dx = bb.MaxX - bb.MinX;
        double dy = bb.MaxY - bb.MinY;

        double z = MmToFtZ(c.Z);

        XYZ start, end;
        if (dx >= dy)
        {
            start = new XYZ(MmToFt(bb.MinX), MmToFt(c.Y), z);
            end   = new XYZ(MmToFt(bb.MaxX), MmToFt(c.Y), z);
        }
        else
        {
            start = new XYZ(MmToFt(c.X), MmToFt(bb.MinY), z);
            end   = new XYZ(MmToFt(c.X), MmToFt(bb.MaxY), z);
        }

        if (start.DistanceTo(end) < MmToFt(100.0))
        {
            Debug.WriteLine($"[ScanToBIM] Degenerate beam length for segment {instruction.SegmentId}. Rejected.");
            return ElementId.InvalidElementId;
        }

        var level     = GetOrCreateLevel(MmToFtZ(bb.MinZ), "Scan Level Beam");
        var beamSymbol = GetDefaultBeamSymbol();

        using var tx = new Transaction(_doc, $"ScanToBIM: Beam [{ShortId(instruction)}]");
        tx.Start();

        if (!beamSymbol.IsActive)
            beamSymbol.Activate();

        var beam = _doc.Create.NewFamilyInstance(
            Line.CreateBound(start, end),
            beamSymbol,
            level,
            StructuralType.Beam);
        TrySetLevelOffset(beam, level, z);

        SafetyGateService.WriteComplianceParameters(beam, instruction);

        tx.Commit();
        return beam.Id;
    }

    // ── Stair ─────────────────────────────────────────────────────────────────

    private ElementId CreateStair(ElementInstruction instruction)
    {
        try
        {
            var bb = instruction.BoundingBox;

            double minZ = MmToFtZ(bb.MinZ);
            double maxZ = MmToFtZ(bb.MaxZ);
            double riseFt = Math.Max(maxZ - minZ, MmToFt(700.0));

            double minX = MmToFt(bb.MinX);
            double maxX = MmToFt(bb.MaxX);
            double minY = MmToFt(bb.MinY);
            double maxY = MmToFt(bb.MaxY);

            double runX = Math.Abs(maxX - minX);
            double runY = Math.Abs(maxY - minY);
            bool alongX = runX >= runY;

            // Use detected run length, but keep within practical stair bounds.
            double runLengthFt = Math.Max(alongX ? runX : runY, MmToFt(1200.0));
            runLengthFt = Math.Min(runLengthFt, MmToFt(12000.0));

            double cx = (minX + maxX) * 0.5;
            double cy = (minY + maxY) * 0.5;
            double zBase = minZ;

            XYZ start = alongX
                ? new XYZ(cx - (runLengthFt * 0.5), cy, zBase)
                : new XYZ(cx, cy - (runLengthFt * 0.5), zBase);
            XYZ end = alongX
                ? new XYZ(cx + (runLengthFt * 0.5), cy, zBase)
                : new XYZ(cx, cy + (runLengthFt * 0.5), zBase);

            var baseLevel = GetOrCreateLevel(zBase, "Scan Level Stair Base");
            var topLevel = GetOrCreateLevel(zBase + riseFt, "Scan Level Stair Top");

            // Stair API requires two distinct levels; force-create top level if needed.
            if (topLevel.Id == baseLevel.Id)
            {
                using var txTop = new Transaction(_doc, $"ScanToBIM: Stair Top Level [{ShortId(instruction)}]");
                txTop.Start();
                topLevel = Level.Create(_doc, zBase + riseFt);
                topLevel.Name = $"Scan Level Stair Top {((zBase + riseFt) * 304.8):F0}mm";
                txTop.Commit();
            }

            ElementId stairsId;
            using (var scope = new StairsEditScope(_doc, $"ScanToBIM: Stair [{ShortId(instruction)}]"))
            {
                stairsId = scope.Start(baseLevel.Id, topLevel.Id);

                using var txRun = new Transaction(_doc, $"ScanToBIM: Stair Run [{ShortId(instruction)}]");
                txRun.Start();

                var run = StairsRun.CreateStraightRun(
                    _doc,
                    stairsId,
                    Line.CreateBound(start, end),
                    StairsRunJustification.Center);

                var widthFt = Math.Max(Math.Min(alongX ? runY : runX, MmToFt(3500.0)), MmToFt(900.0));
                run.get_Parameter(BuiltInParameter.STAIRS_RUN_ACTUAL_RUN_WIDTH)?.Set(widthFt);

                txRun.Commit();
                scope.Commit(new StairFailureSilencer());
            }

            using var txMeta = new Transaction(_doc, $"ScanToBIM: Stair Metadata [{ShortId(instruction)}]");
            txMeta.Start();
            var stair = _doc.GetElement(stairsId);
            if (stair != null)
                SafetyGateService.WriteComplianceParameters(stair, instruction);
            txMeta.Commit();

            return stairsId;
        }
        catch
        {
            // Fallback keeps stair output visible when native stair constraints fail.
            var geom = BuildSteppedStairGeometry(instruction);
            if (geom.Count == 0)
                throw new InvalidOperationException("Unable to build stair geometry from instruction bounds.");

            using var tx = new Transaction(_doc, $"ScanToBIM: Stair Fallback [{ShortId(instruction)}]");
            tx.Start();

            DirectShape stairElement;
            try
            {
                stairElement = DirectShape.CreateElement(_doc, new ElementId(BuiltInCategory.OST_Stairs));
            }
            catch
            {
                stairElement = DirectShape.CreateElement(_doc, new ElementId(BuiltInCategory.OST_GenericModel));
            }

            stairElement.Name = $"STB_Stair_{ShortId(instruction)}";
            stairElement.SetShape(geom);
            SafetyGateService.WriteComplianceParameters(stairElement, instruction);

            tx.Commit();
            return stairElement.Id;
        }
    }

    // ── Ramp ──────────────────────────────────────────────────────────────────

    private ElementId CreateRamp(ElementInstruction instruction)
    {
        // Ramps are modelled as sloped floors using the Floor API.
        // The agent has confirmed no discrete stair treads (stair_steps < 3),
        // so we create a thin sloped slab from min-Z to max-Z.
        var bb = instruction.BoundingBox;

        double x0 = MmToFt(bb.MinX);
        double x1 = MmToFt(bb.MaxX);
        double y0 = MmToFt(bb.MinY);
        double y1 = MmToFt(bb.MaxY);
        double z0 = MmToFtZ(bb.MinZ);
        double z1 = MmToFtZ(bb.MaxZ);

        double riseFt = Math.Abs(z1 - z0);
        double runFt = Math.Max(Math.Abs(x1 - x0), Math.Abs(y1 - y0));
        if (ShouldPromoteRampToStair(instruction, riseFt, runFt))
            return CreateStair(instruction);

        var level = GetOrCreateLevel(z0, "Scan Level Ramp");

        using var tx = new Transaction(_doc, $"ScanToBIM: Ramp [{ShortId(instruction)}]");
        tx.Start();

        // Build a four-point profile: low end at z0, high end at z1
        var profile = new CurveLoop();
        profile.Append(Line.CreateBound(new XYZ(x0, y0, z0), new XYZ(x1, y0, z0)));
        profile.Append(Line.CreateBound(new XYZ(x1, y0, z0), new XYZ(x1, y1, z1)));
        profile.Append(Line.CreateBound(new XYZ(x1, y1, z1), new XYZ(x0, y1, z1)));
        profile.Append(Line.CreateBound(new XYZ(x0, y1, z1), new XYZ(x0, y0, z0)));

        var floorType = GetDefaultFloorType();
        var floor = Floor.Create(_doc, new List<CurveLoop> { profile }, floorType.Id, level.Id);
        TrySetLevelOffset(floor, level, z0);

        SafetyGateService.WriteComplianceParameters(floor, instruction);

        tx.Commit();
        return floor.Id;
    }

    // ── Column ────────────────────────────────────────────────────────────────

    private ElementId CreateColumn(ElementInstruction instruction)
    {
        var c  = instruction.Centroid;
        var bb = instruction.BoundingBox;

        double x  = MmToFt(c.X);
        double y  = MmToFt(c.Y);
        double z0 = MmToFtZ(bb.MinZ);
        double z1 = MmToFtZ(bb.MaxZ);

        var level     = GetOrCreateLevel(z0, "Scan Level Column");
        var colSymbol = GetDefaultColumnSymbol();

        using var tx = new Transaction(_doc, $"ScanToBIM: Column [{ShortId(instruction)}]");
        tx.Start();

        if (!colSymbol.IsActive)
            colSymbol.Activate();

        var col = _doc.Create.NewFamilyInstance(
            new XYZ(x, y, z0),
            colSymbol,
            level,
            StructuralType.Column);
        TrySetLevelOffset(col, level, z0);

        // Set top offset to match scan height
        double colHeight = Math.Abs(z1 - z0);
        if (colHeight > 0.01)
            col.get_Parameter(BuiltInParameter.FAMILY_TOP_LEVEL_OFFSET_PARAM)?.Set(colHeight);

        // Parameterize cross-section dimensions to match fitted scan geometry
        double colWidthFt = MmToFt(Math.Max(bb.MaxX - bb.MinX, 100.0));
        double colDepthFt = MmToFt(Math.Max(bb.MaxY - bb.MinY, 100.0));
        if (TryGetParamDouble(instruction, "column_width_mm", out var cw) && cw > 10.0) colWidthFt = MmToFt(cw);
        if (TryGetParamDouble(instruction, "column_depth_mm", out var cd) && cd > 10.0) colDepthFt = MmToFt(cd);

        col.LookupParameter("b")?.Set(colWidthFt);
        col.LookupParameter("h")?.Set(colDepthFt);
        col.LookupParameter("Width")?.Set(colWidthFt);
        col.LookupParameter("Depth")?.Set(colDepthFt);
        if (TryGetParamDouble(instruction, "column_diameter_mm", out var cDiam) && cDiam > 10.0)
        {
            col.LookupParameter("Diameter")?.Set(MmToFt(cDiam));
            col.LookupParameter("d")?.Set(MmToFt(cDiam));
        }

        SafetyGateService.WriteComplianceParameters(col, instruction);

        tx.Commit();
        return col.Id;
    }

    // ── Duct ──────────────────────────────────────────────────────────────────

    private ElementId CreateDuct(ElementInstruction instruction)
    {
        var bb = instruction.BoundingBox;
        var c  = instruction.Centroid;
        var spansMm = GetOrderedBoundingBoxSpansMm(bb);
        
        var cyl = CylinderGeometryHelper.TryExtract(instruction);
        XYZ start, end;
        
        if (cyl.HasValue)
        {
            start = cyl.Value.Start;
            end = cyl.Value.End;
        }
        else
        {
            double dx = bb.MaxX - bb.MinX;
            double dy = bb.MaxY - bb.MinY;
            double z  = MmToFt(c.Z);

            if (dx >= dy)
            {
                start = new XYZ(MmToFt(bb.MinX), MmToFt(c.Y), z);
                end   = new XYZ(MmToFt(bb.MaxX), MmToFt(c.Y), z);
            }
            else
            {
                start = new XYZ(MmToFt(c.X), MmToFt(bb.MinY), z);
                end   = new XYZ(MmToFt(c.X), MmToFt(bb.MaxY), z);
            }
        }

        if (start.DistanceTo(end) < MmToFt(100.0))
        {
            Debug.WriteLine($"[ScanToBIM] Degenerate duct length for segment {instruction.SegmentId}. Rejected.");
            return ElementId.InvalidElementId;
        }

        var runElevation = (start.Z + end.Z) * 0.5;
        var level       = GetOrCreateLevel(runElevation, "Scan Level Duct");
        var ductType    = GetDefaultDuctType();
        var mepSysType  = GetMechanicalSystemTypeByColor(instruction);

        using var tx = new Transaction(_doc, $"ScanToBIM: Duct [{ShortId(instruction)}]");
        tx.Start();

        var duct = Duct.Create(_doc, ductType.Id, mepSysType.Id, level.Id, start, end);

    // Use the two smallest spans as the cross-section; the largest is the run length.
    double widthFt  = MmToFt(ClampMm(spansMm[1], 100.0));
    double heightFt = MmToFt(ClampMm(spansMm[0], 100.0));
    duct.get_Parameter(BuiltInParameter.RBS_CURVE_WIDTH_PARAM)?.Set(widthFt);
    duct.get_Parameter(BuiltInParameter.RBS_CURVE_HEIGHT_PARAM)?.Set(heightFt);

        SafetyGateService.WriteComplianceParameters(duct, instruction);

        tx.Commit();
        return duct.Id;
    }

    // ── Conduit ───────────────────────────────────────────────────────────────

    private ElementId CreateConduit(ElementInstruction instruction)
    {
        var bb = instruction.BoundingBox;
        var c  = instruction.Centroid;
        var spansMm = GetOrderedBoundingBoxSpansMm(bb);
        
        var cyl = CylinderGeometryHelper.TryExtract(instruction);
        XYZ start, end;
        double diamFt;

        if (cyl.HasValue)
        {
            start = cyl.Value.Start;
            end = cyl.Value.End;
            diamFt = cyl.Value.RadiusFt * 2.0;
        }
        else
        {
            double dx = bb.MaxX - bb.MinX;
            double dy = bb.MaxY - bb.MinY;
            double dz = bb.MaxZ - bb.MinZ;
            double z  = MmToFt(c.Z);

            // Fallback keeps linear runs horizontal while scaling length from
            // the largest observed span (including Z) so output is not hardcoded.
            double majorMm = Math.Max(dx, Math.Max(dy, dz));
            double halfMajorFt = MmToFt(Math.Max(majorMm, 500.0)) / 2.0;
            double cx = MmToFt(c.X);
            double cy = MmToFt(c.Y);

            if (dx >= dy)
            {
                start = new XYZ(cx - halfMajorFt, cy, z);
                end   = new XYZ(cx + halfMajorFt, cy, z);
            }
            else
            {
                start = new XYZ(cx, cy - halfMajorFt, z);
                end   = new XYZ(cx, cy + halfMajorFt, z);
            }

            diamFt = MmToFt(ClampMm((spansMm[0] + spansMm[1]) / 2.0, 20.0));
        }

        if (start.DistanceTo(end) < MmToFt(100.0))
        {
            Debug.WriteLine($"[ScanToBIM] Degenerate conduit length for segment {instruction.SegmentId}. Rejected.");
            return ElementId.InvalidElementId;
        }

        var runElevation = (start.Z + end.Z) * 0.5;
        var level        = GetOrCreateLevel(runElevation, "Scan Level Conduit");
        var conduitType  = GetDefaultConduitType();

        using var tx = new Transaction(_doc, $"ScanToBIM: Conduit [{ShortId(instruction)}]");
        tx.Start();

        var conduit = Conduit.Create(_doc, conduitType.Id, start, end, level.Id);

        conduit.get_Parameter(BuiltInParameter.RBS_CONDUIT_DIAMETER_PARAM)?.Set(diamFt);

        SafetyGateService.WriteComplianceParameters(conduit, instruction);

        tx.Commit();
        return conduit.Id;
    }

    // ── Pipe ──────────────────────────────────────────────────────────────────

    private ElementId CreatePipe(ElementInstruction instruction)
    {
        var bb = instruction.BoundingBox;
        var c  = instruction.Centroid;
        var spansMm = GetOrderedBoundingBoxSpansMm(bb);

        var cyl = CylinderGeometryHelper.TryExtract(instruction);
        XYZ start, end;
        double diamFt;

        if (cyl.HasValue)
        {
            start = cyl.Value.Start;
            end = cyl.Value.End;
            diamFt = cyl.Value.RadiusFt * 2.0;
        }
        else
        {
            double dx = bb.MaxX - bb.MinX;
            double dy = bb.MaxY - bb.MinY;
            double dz = bb.MaxZ - bb.MinZ;
            double z  = MmToFt(c.Z);

            double majorMm = Math.Max(dx, Math.Max(dy, dz));
            double halfMajorFt = MmToFt(Math.Max(majorMm, 500.0)) / 2.0;
            double cx = MmToFt(c.X);
            double cy = MmToFt(c.Y);

            if (dx >= dy)
            {
                start = new XYZ(cx - halfMajorFt, cy, z);
                end   = new XYZ(cx + halfMajorFt, cy, z);
            }
            else
            {
                start = new XYZ(cx, cy - halfMajorFt, z);
                end   = new XYZ(cx, cy + halfMajorFt, z);
            }

            diamFt = MmToFt(ClampMm((spansMm[0] + spansMm[1]) / 2.0, 20.0));
        }

        if (start.DistanceTo(end) < 0.01)
            end = new XYZ(start.X + MmToFt(500), start.Y, start.Z);

        bool isYellowPipe = IsYellowPipeColor(instruction);
        double runLengthFt = start.DistanceTo(end);

        if (isYellowPipe)
        {
            double minYellowMm = runLengthFt >= MmToFt(3000.0) ? 180.0 : 120.0;
            diamFt = Math.Max(diamFt, MmToFt(minYellowMm));

            if (diamFt >= MmToFt(100.0))
            {
                var wallTolFt = MmToFt(1200.0);
                var startWall = FindNearestWall(start, wallTolFt);
                var endWall = FindNearestWall(end, wallTolFt);

                if (startWall != null)
                    start = AlignPointToWallHost(startWall, start, start.Z);
                if (endWall != null)
                    end = AlignPointToWallHost(endWall, end, end.Z);

                if (startWall != null && endWall != null && startWall.Id == endWall.Id && runLengthFt > MmToFt(800.0))
                {
                    var axis = (end - start);
                    if (axis.GetLength() > 1e-6)
                    {
                        axis = axis.Normalize();
                        end = start + axis.Multiply(runLengthFt);
                    }
                }
            }
        }

        var runElevation = (start.Z + end.Z) * 0.5;
        var level       = GetOrCreateLevel(runElevation, "Scan Level Pipe");
        var pipeType    = GetDefaultPipeType();
        var pipeSysType = GetPipingSystemTypeByColor(instruction);

        using var tx = new Transaction(_doc, $"ScanToBIM: Pipe [{ShortId(instruction)}]");
        tx.Start();

        var pipe = Pipe.Create(_doc, pipeSysType.Id, pipeType.Id, level.Id, start, end);
        pipe.get_Parameter(BuiltInParameter.RBS_PIPE_DIAMETER_PARAM)?.Set(diamFt);

        TryConnectPipeToNearbyNetwork(pipe, start, end, diamFt);

        SafetyGateService.WriteComplianceParameters(pipe, instruction);

        tx.Commit();
        return pipe.Id;
    }
    // ── Cable Tray ────────────────────────────────────────────────────────────

    private ElementId CreateCableTray(ElementInstruction instruction)
    {
        var bb = instruction.BoundingBox;
        var c  = instruction.Centroid;
        var spansMm = GetOrderedBoundingBoxSpansMm(bb);

        double dx = bb.MaxX - bb.MinX;
        double dy = bb.MaxY - bb.MinY;
        double z  = MmToFt(c.Z);

        XYZ start = dx >= dy
            ? new XYZ(MmToFt(bb.MinX), MmToFt(c.Y), z)
            : new XYZ(MmToFt(c.X), MmToFt(bb.MinY), z);
        XYZ end = dx >= dy
            ? new XYZ(MmToFt(bb.MaxX), MmToFt(c.Y), z)
            : new XYZ(MmToFt(c.X), MmToFt(bb.MaxY), z);

        if (start.DistanceTo(end) < 0.01)
            end = new XYZ(start.X + MmToFt(500), start.Y, start.Z);

        var level         = GetOrCreateLevel(z, "Scan Level CableTray");
        var cableTrayType = GetDefaultCableTrayType();

        using var tx = new Transaction(_doc, $"ScanToBIM: CableTray [{ShortId(instruction)}]");
        tx.Start();

        var tray = CableTray.Create(_doc, cableTrayType.Id, start, end, level.Id);

    double widthFt  = MmToFt(ClampMm(spansMm[1], 50.0));
    double heightFt = MmToFt(ClampMm(spansMm[0], 25.0));
    tray.get_Parameter(BuiltInParameter.RBS_CABLETRAY_WIDTH_PARAM)?.Set(widthFt);
    tray.get_Parameter(BuiltInParameter.RBS_CABLETRAY_HEIGHT_PARAM)?.Set(heightFt);

        SafetyGateService.WriteComplianceParameters(tray, instruction);

        tx.Commit();
        return tray.Id;
    }

    // ── MEP Equipment (HVAC unit / Tank / Pump / Vessel / Nuclear) ───────────

    private ElementId CreateMepEquipment(ElementInstruction instruction, BuiltInCategory category)
    {
        var c  = instruction.Centroid;
        var bb = instruction.BoundingBox;

        if (_enclosedShellTypes.Contains(instruction.ElementType))
            return CreateEnclosedMechanicalShell(instruction, category);

        var insertPt  = AnchorPointToEnvelope(new XYZ(MmToFt(c.X), MmToFt(c.Y), MmToFtZ(bb.MinZ)), MmToFtZ(bb.MinZ));
        var level     = GetOrCreateLevel(insertPt.Z, $"Scan Level {instruction.ElementType}");
        var resolved  = _resolver.Resolve(instruction, category);
        var symbol    = resolved.Symbol ?? GetDefaultMepEquipmentSymbol(category);

        using var tx = new Transaction(_doc, $"ScanToBIM: {instruction.ElementType} [{ShortId(instruction)}]");
        tx.Start();

        if (!symbol.IsActive)
            symbol.Activate();

        var instance = _doc.Create.NewFamilyInstance(
            insertPt, symbol, level, StructuralType.NonStructural);
        TrySetLevelOffset(instance, level, insertPt.Z);

        SafetyGateService.WriteComplianceParameters(instance, instruction);
        // Record which resolution path was used for audit traceability
        instance.get_Parameter(BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS)
            ?.Set($"FamilyResolution={resolved.Path} Family={resolved.FamilyName ?? "fallback"} Type={resolved.TypeName ?? "default"}");

        tx.Commit();
        return instance.Id;
    }

    private ElementId CreateEnclosedMechanicalShell(ElementInstruction instruction, BuiltInCategory category)
    {
        var bb = instruction.BoundingBox;
        double x0 = MmToFt(bb.MinX);
        double y0 = MmToFt(bb.MinY);
        double z0 = MmToFtZ(bb.MinZ);
        double x1 = MmToFt(bb.MaxX);
        double y1 = MmToFt(bb.MaxY);
        double z1 = MmToFtZ(bb.MaxZ);

        var anchor = AnchorPointToEnvelope(
            new XYZ((x0 + x1) / 2.0, (y0 + y1) / 2.0, z0),
            z0);
        double zShift = anchor.Z - z0;
        z0 += zShift;
        z1 += zShift;

        // Try extracting cylinder parameters first
        var cyl = CylinderGeometryHelper.TryExtract(instruction);
        Solid shellSolid = null;
        double shellMm = 0.0;

        if (cyl.HasValue)
        {
            var start = cyl.Value.Start;
            var end = cyl.Value.End;
            double radius = cyl.Value.RadiusFt;

            // Apply the same z-shift to keep it anchored to level
            start = new XYZ(start.X, start.Y, start.Z + zShift);
            end = new XYZ(end.X, end.Y, end.Z + zShift);

            // Force orthogonal snapping: horizontal (along X or Y) or vertical (along Z)
            var axis = (end - start).Normalize();
            double len = start.DistanceTo(end);
            
            double dotX = Math.Abs(axis.DotProduct(XYZ.BasisX));
            double dotY = Math.Abs(axis.DotProduct(XYZ.BasisY));
            double dotZ = Math.Abs(axis.DotProduct(XYZ.BasisZ));

            XYZ snappedAxis;
            if (dotZ > 0.85) snappedAxis = XYZ.BasisZ;
            else if (dotX > dotY) snappedAxis = XYZ.BasisX;
            else snappedAxis = XYZ.BasisY;

            // Re-center around centroid (shifted for level anchoring)
            var centroid = new XYZ(MmToFt(instruction.Centroid.X), MmToFt(instruction.Centroid.Y), MmToFtZ(instruction.Centroid.Z) + zShift);
            start = centroid - snappedAxis.Multiply(len / 2.0);
            end = centroid + snappedAxis.Multiply(len / 2.0);

            var cylSolid = BuildCylinderSolid(start, end, radius);
            if (cylSolid != null)
            {
                double minDimMm = Math.Max(1.0, radius * 2.0 * 304.8);
                shellMm = Math.Clamp(minDimMm * 0.10, 20.0, 120.0);
                if (TryGetParamDouble(instruction, "shell_thickness_mm", out var shellFromParam) && shellFromParam > 0)
                    shellMm = Math.Clamp(shellFromParam, 10.0, minDimMm / 2.0);

                double shellFt = MmToFt(shellMm);
                if (shellFt > MmToFt(2.0) && shellFt < radius - MmToFt(5.0))
                {
                    var innerCyl = BuildCylinderSolid(
                        start + snappedAxis.Multiply(shellFt),
                        end - snappedAxis.Multiply(shellFt),
                        radius - shellFt
                    );
                    if (innerCyl != null)
                    {
                        try
                        {
                            shellSolid = BooleanOperationsUtils.ExecuteBooleanOperation(
                                cylSolid,
                                innerCyl,
                                BooleanOperationsType.Difference
                            );
                        }
                        catch
                        {
                            shellSolid = cylSolid;
                        }
                    }
                    else
                    {
                        shellSolid = cylSolid;
                    }
                }
                else
                {
                    shellSolid = cylSolid;
                }
            }
        }

        // Fallback to standard box solid construction if cylinder creation is not applicable or failed
        if (shellSolid == null)
        {
            var outer = BuildBoxSolid(x0, y0, z0, x1, y1, z1)
                ?? throw new InvalidOperationException("Mechanical shell outer geometry is invalid.");

            double minDimMm = Math.Max(
                1.0,
                Math.Min(bb.MaxX - bb.MinX, Math.Min(bb.MaxY - bb.MinY, bb.MaxZ - bb.MinZ))
            );
            shellMm = Math.Clamp(minDimMm * 0.10, 20.0, 120.0);
            if (TryGetParamDouble(instruction, "shell_thickness_mm", out var shellFromParam) && shellFromParam > 0)
                shellMm = Math.Clamp(shellFromParam, 10.0, minDimMm / 2.0);

            double shellFt = MmToFt(shellMm);
            double maxShellFt = Math.Max(0.0, Math.Min(x1 - x0, Math.Min(y1 - y0, z1 - z0)) / 2.0 - MmToFt(5.0));

            shellSolid = outer;
            if (shellFt > MmToFt(2.0) && shellFt < maxShellFt)
            {
                var inner = BuildBoxSolid(x0 + shellFt, y0 + shellFt, z0 + shellFt, x1 - shellFt, y1 - shellFt, z1 - shellFt);
                if (inner != null)
                {
                    try
                    {
                        shellSolid = BooleanOperationsUtils.ExecuteBooleanOperation(
                            outer,
                            inner,
                            BooleanOperationsType.Difference
                        );
                    }
                    catch
                    {
                        shellSolid = outer;
                    }
                }
            }
        }

        using var tx = new Transaction(_doc, $"ScanToBIM: {instruction.ElementType} Shell [{ShortId(instruction)}]");
        tx.Start();

        var ds = DirectShape.CreateElement(_doc, new ElementId(category));
        var namePrefix = cyl.HasValue ? "Cylinder" : (string.Equals(instruction.ElementType.ToString(), "pressure_vessel", StringComparison.OrdinalIgnoreCase) || string.Equals(instruction.ElementType.ToString(), "tank", StringComparison.OrdinalIgnoreCase) ? "Container" : instruction.ElementType.ToString());
        ds.Name = $"STB_{namePrefix}_{ShortId(instruction)}";
        ds.SetShape(new List<GeometryObject> { shellSolid });
        ds.get_Parameter(BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS)
            ?.Set($"EnclosedShell=True ShellThicknessMm={shellMm:F1}");

        SafetyGateService.WriteComplianceParameters(ds, instruction);

        tx.Commit();
        return ds.Id;
    }

    // ── Door ──────────────────────────────────────────────────────────────────

    private ElementId CreateDoor(ElementInstruction instruction)
    {
        var bb = instruction.BoundingBox;
        var c  = instruction.Centroid;

        // Insert point is at the bottom-centre of the opening
        var insertPt = new XYZ(MmToFt(c.X), MmToFt(c.Y), MmToFtZ(bb.MinZ));
        var level     = GetOrCreateLevel(insertPt.Z, "Scan Level Door");
        var doorSymbol = GetDefaultDoorSymbol();
        var hostWall   = FindNearestWall(insertPt);
        var fitInsertPt = AlignPointToWallHost(hostWall, insertPt, MmToFtZ(bb.MinZ));

        using var tx = new Transaction(_doc, $"ScanToBIM: Door [{ShortId(instruction)}]");
        tx.Start();

        if (!doorSymbol.IsActive)
            doorSymbol.Activate();

        FamilyInstance door;
        if (hostWall != null)
            door = _doc.Create.NewFamilyInstance(fitInsertPt, doorSymbol, hostWall, level, StructuralType.NonStructural);
        else
            door = _doc.Create.NewFamilyInstance(insertPt, doorSymbol, level, StructuralType.NonStructural);
        TrySetLevelOffset(door, level, insertPt.Z);

        // Derive door width along host wall axis rather than assuming X-axis span
        double widthFt;
        if (TryGetParamDouble(instruction, "opening_width_mm", out var dwMm) && dwMm > 10.0)
        {
            widthFt = MmToFt(dwMm);
        }
        else if (hostWall?.Location is LocationCurve hostCurve && hostCurve.Curve is Line wallLine)
        {
            var wallDir = wallLine.Direction.Normalize();
            var dx = MmToFt(bb.MaxX - bb.MinX);
            var dy = MmToFt(bb.MaxY - bb.MinY);
            widthFt = Math.Max(Math.Abs(dx * wallDir.X) + Math.Abs(dy * wallDir.Y), Math.Min(dx, dy));
            if (widthFt < 0.1) widthFt = Math.Max(dx, dy);
        }
        else
        {
            widthFt = MmToFt(Math.Max(bb.MaxX - bb.MinX, bb.MaxY - bb.MinY));
        }

        double heightFt = MmToFt(bb.MaxZ - bb.MinZ);
        if (hostWall?.Location is LocationCurve hostCurveRef)
            widthFt = Math.Min(widthFt, hostCurveRef.Curve.Length * 0.95);
        door.get_Parameter(BuiltInParameter.DOOR_WIDTH)?.Set(widthFt);
        door.get_Parameter(BuiltInParameter.DOOR_HEIGHT)?.Set(heightFt);

        SafetyGateService.WriteComplianceParameters(door, instruction);

        tx.Commit();
        return door.Id;
    }

    // ── Window ────────────────────────────────────────────────────────────────

    private ElementId CreateWindow(ElementInstruction instruction)
    {
        var bb = instruction.BoundingBox;
        var c  = instruction.Centroid;

        // Insert at sill centre (bottom of void)
        var insertPt     = new XYZ(MmToFt(c.X), MmToFt(c.Y), MmToFtZ(bb.MinZ));
        var level        = GetOrCreateLevel(insertPt.Z, "Scan Level Window");
        var windowSymbol = GetDefaultWindowSymbol();
        var hostWall     = FindNearestWall(insertPt);
        var fitInsertPt  = AlignPointToWallHost(hostWall, insertPt, MmToFtZ(bb.MinZ));

        using var tx = new Transaction(_doc, $"ScanToBIM: Window [{ShortId(instruction)}]");
        tx.Start();

        if (!windowSymbol.IsActive)
            windowSymbol.Activate();

        FamilyInstance win;
        if (hostWall != null)
            win = _doc.Create.NewFamilyInstance(fitInsertPt, windowSymbol, hostWall, level, StructuralType.NonStructural);
        else
            win = _doc.Create.NewFamilyInstance(insertPt, windowSymbol, level, StructuralType.NonStructural);
        TrySetLevelOffset(win, level, insertPt.Z);

        // Derive window width along host wall axis rather than assuming X-axis span
        double widthFt;
        if (TryGetParamDouble(instruction, "opening_width_mm", out var wwMm) && wwMm > 10.0)
        {
            widthFt = MmToFt(wwMm);
        }
        else if (hostWall?.Location is LocationCurve hostCurve && hostCurve.Curve is Line wallLine)
        {
            var wallDir = wallLine.Direction.Normalize();
            var dx = MmToFt(bb.MaxX - bb.MinX);
            var dy = MmToFt(bb.MaxY - bb.MinY);
            widthFt = Math.Max(Math.Abs(dx * wallDir.X) + Math.Abs(dy * wallDir.Y), Math.Min(dx, dy));
            if (widthFt < 0.1) widthFt = Math.Max(dx, dy);
        }
        else
        {
            widthFt = MmToFt(Math.Max(bb.MaxX - bb.MinX, bb.MaxY - bb.MinY));
        }

        double heightFt = MmToFt(bb.MaxZ - bb.MinZ);
        if (hostWall?.Location is LocationCurve hostCurveWin)
            widthFt = Math.Min(widthFt, hostCurveWin.Curve.Length * 0.95);
        win.get_Parameter(BuiltInParameter.WINDOW_WIDTH)?.Set(widthFt);
        win.get_Parameter(BuiltInParameter.WINDOW_HEIGHT)?.Set(heightFt);

        SafetyGateService.WriteComplianceParameters(win, instruction);

        tx.Commit();
        return win.Id;
    }

    // ── Railing ───────────────────────────────────────────────────────────────

    private ElementId CreateRailing(ElementInstruction instruction)
    {
        var bb = instruction.BoundingBox;
        var c  = instruction.Centroid;

        // Railing path along its longest horizontal axis
        double dx = bb.MaxX - bb.MinX;
        double dy = bb.MaxY - bb.MinY;
        double z  = MmToFtZ(bb.MinZ);

        XYZ start, end;
        if (dx >= dy)
        {
            start = new XYZ(MmToFt(bb.MinX), MmToFt(c.Y), z);
            end   = new XYZ(MmToFt(bb.MaxX), MmToFt(c.Y), z);
        }
        else
        {
            start = new XYZ(MmToFt(c.X), MmToFt(bb.MinY), z);
            end   = new XYZ(MmToFt(c.X), MmToFt(bb.MaxY), z);
        }

        if (start.DistanceTo(end) < MmToFt(100.0))
        {
            Debug.WriteLine($"[ScanToBIM] Degenerate railing length for segment {instruction.SegmentId}. Rejected.");
            return ElementId.InvalidElementId;
        }

        var level       = GetOrCreateLevel(z, "Scan Level Railing");
        var railingType = GetDefaultRailingType();
        var path        = new CurveLoop();
        path.Append(Line.CreateBound(start, end));

        using var tx = new Transaction(_doc, $"ScanToBIM: Railing [{ShortId(instruction)}]");
        tx.Start();

        var railing = Railing.Create(_doc, path, railingType.Id, level.Id);
        TrySetLevelOffset(railing, level, z);

        SafetyGateService.WriteComplianceParameters(railing, instruction);

        tx.Commit();
        return railing.Id;
    }

    // ── Valve ─────────────────────────────────────────────────────────────────

    private ElementId CreateValve(ElementInstruction instruction)
    {
        var c        = instruction.Centroid;
        var bb       = instruction.BoundingBox;
        var spansMm  = GetOrderedBoundingBoxSpansMm(bb);
        var pt       = AnchorPointToEnvelope(new XYZ(MmToFt(c.X), MmToFt(c.Y), MmToFtZ(c.Z)), MmToFtZ(c.Z));
        var level    = GetOrCreateLevel(pt.Z, "Scan Level Valve");
        var resolved = _resolver.Resolve(instruction, BuiltInCategory.OST_PipeAccessory);
        var sym      = resolved.Symbol ?? GetDefaultValveSymbol();

        using var tx = new Transaction(_doc, $"ScanToBIM: Valve [{ShortId(instruction)}]");
        tx.Start();

        if (!sym.IsActive)
            sym.Activate();

        var inst = _doc.Create.NewFamilyInstance(
            pt, sym, level, StructuralType.NonStructural);
        TrySetLevelOffset(inst, level, pt.Z);

        // Set nominal diameter from the valve body's smallest cross-section extent
        double minDimFt = MmToFt(ClampMm((spansMm[0] + spansMm[1]) / 2.0, 20.0));
        inst.get_Parameter(BuiltInParameter.RBS_PIPE_DIAMETER_PARAM)?.Set(minDimFt);

        SafetyGateService.WriteComplianceParameters(inst, instruction);

        tx.Commit();
        return inst.Id;
    }

    // ── Slab Opening ──────────────────────────────────────────────────────────

    private ElementId CreateSlabOpening(ElementInstruction instruction)
    {
        var bb = instruction.BoundingBox;
        double z = MmToFtZ((bb.MinZ + bb.MaxZ) / 2.0);

        // Build rectangular profile at slab mid-elevation
        double x0 = MmToFt(bb.MinX), y0 = MmToFt(bb.MinY);
        double x1 = MmToFt(bb.MaxX), y1 = MmToFt(bb.MaxY);

        var corners = new[] {
            new XYZ(x0, y0, z), new XYZ(x1, y0, z),
            new XYZ(x1, y1, z), new XYZ(x0, y1, z),
        };
        var profile = new CurveArray();
        for (int i = 0; i < corners.Length; i++)
            profile.Append(Line.CreateBound(corners[i], corners[(i + 1) % corners.Length]));

        // Find the nearest floor element to host the opening
        var hostFloor = new FilteredElementCollector(_doc)
            .OfClass(typeof(Floor))
            .Cast<Floor>()
            .OrderBy(f =>
            {
                var bb2 = f.get_BoundingBox(null);
                return bb2 == null ? double.MaxValue
                    : Math.Abs(bb2.Min.Z + (bb2.Max.Z - bb2.Min.Z) / 2 - z);
            })
            .FirstOrDefault();

        using var tx = new Transaction(_doc, $"ScanToBIM: SlabOpening [{ShortId(instruction)}]");
        tx.Start();

        Element opening;
        if (hostFloor != null)
            opening = _doc.Create.NewOpening(hostFloor, profile, true);
        else
        {
            // No host floor — create a shaft opening spanning slab thickness
            var level = GetOrCreateLevel(MmToFtZ(bb.MinZ), "Scan Level Slab Opening");
            var topLevel = GetOrCreateLevel(MmToFtZ(bb.MaxZ), "Scan Level Slab Opening Top");
            opening   = _doc.Create.NewOpening(level, topLevel, profile);
        }

        SafetyGateService.WriteComplianceParameters(opening, instruction);

        tx.Commit();
        return opening.Id;
    }

    // ── Sprinkler / Smoke detector / Lighting ─────────────────────────────────

    private ElementId CreateFireFixture(ElementInstruction instruction)
    {
        var c        = instruction.Centroid;
        var bb       = instruction.BoundingBox;
        var pt       = AnchorPointToEnvelope(new XYZ(MmToFt(c.X), MmToFt(c.Y), MmToFtZ(bb.MaxZ)), MmToFtZ(bb.MaxZ));
        var level    = GetOrCreateLevel(pt.Z, "Scan Level Sprinkler");
        var cat      = instruction.ElementType == ElementType.lighting_fitting
            ? BuiltInCategory.OST_LightingFixtures
            : instruction.ElementType == ElementType.smoke_detector
                ? BuiltInCategory.OST_FireAlarmDevices
                : BuiltInCategory.OST_Sprinklers;
        var resolved = _resolver.Resolve(instruction, cat);
        var sym      = resolved.Symbol ?? GetDefaultMepEquipmentSymbol(cat);

        using var tx = new Transaction(_doc, $"ScanToBIM: Sprinkler [{ShortId(instruction)}]");
        tx.Start();

        if (!sym.IsActive) sym.Activate();
        var inst = _doc.Create.NewFamilyInstance(pt, sym, level, StructuralType.NonStructural);
        TrySetLevelOffset(inst, level, pt.Z);

        SafetyGateService.WriteComplianceParameters(inst, instruction);

        tx.Commit();
        return inst.Id;
    }

    // ── Drainage ──────────────────────────────────────────────────────────────

    private ElementId CreatePlumbingFixture(ElementInstruction instruction)
    {
        var c        = instruction.Centroid;
        var bb       = instruction.BoundingBox;
        var pt       = AnchorPointToEnvelope(new XYZ(MmToFt(c.X), MmToFt(c.Y), MmToFtZ(bb.MinZ)), MmToFtZ(bb.MinZ));
        var level    = GetOrCreateLevel(pt.Z, "Scan Level Drainage");
        var resolved = _resolver.Resolve(instruction, BuiltInCategory.OST_PlumbingFixtures);
        var sym      = resolved.Symbol ?? GetDefaultMepEquipmentSymbol(BuiltInCategory.OST_PlumbingFixtures);

        using var tx = new Transaction(_doc, $"ScanToBIM: Drainage [{ShortId(instruction)}]");
        tx.Start();

        if (!sym.IsActive) sym.Activate();
        var inst = _doc.Create.NewFamilyInstance(pt, sym, level, StructuralType.NonStructural);
        TrySetLevelOffset(inst, level, pt.Z);

        SafetyGateService.WriteComplianceParameters(inst, instruction);

        tx.Commit();
        return inst.Id;
    }

    // ── Electrical Panel ──────────────────────────────────────────────────────

    private ElementId CreateElectricalPanel(ElementInstruction instruction)
    {
        var c        = instruction.Centroid;
        var bb       = instruction.BoundingBox;
        var pt       = AnchorPointToEnvelope(new XYZ(MmToFt(c.X), MmToFt(c.Y), MmToFtZ(bb.MinZ)), MmToFtZ(bb.MinZ));
        var level    = GetOrCreateLevel(pt.Z, "Scan Level ElecPanel");
        var resolved = _resolver.Resolve(instruction, BuiltInCategory.OST_ElectricalEquipment);
        var sym      = resolved.Symbol ?? GetDefaultMepEquipmentSymbol(BuiltInCategory.OST_ElectricalEquipment);
        var hostWall = FindNearestWall(pt);
        var fitPt    = AlignPointToWallHost(hostWall, pt, MmToFtZ(bb.MinZ));

        using var tx = new Transaction(_doc, $"ScanToBIM: ElecPanel [{ShortId(instruction)}]");
        tx.Start();

        if (!sym.IsActive) sym.Activate();

        FamilyInstance inst;
        if (hostWall != null)
            inst = _doc.Create.NewFamilyInstance(fitPt, sym, hostWall, level, StructuralType.NonStructural);
        else
            inst = _doc.Create.NewFamilyInstance(pt, sym, level, StructuralType.NonStructural);
        TrySetLevelOffset(inst, level, pt.Z);

        SafetyGateService.WriteComplianceParameters(inst, instruction);

        tx.Commit();
        return inst.Id;
    }

    // ── Kerb ──────────────────────────────────────────────────────────────────

    private ElementId CreateKerb(ElementInstruction instruction)
    {
        // Kerbs are modelled as very low walls using the standard Wall API.
        var bb = instruction.BoundingBox;

        double x0 = MmToFt(bb.MinX), y0 = MmToFt(bb.MinY), z0 = MmToFtZ(bb.MinZ);
        double x1 = MmToFt(bb.MaxX), y1 = MmToFt(bb.MaxY);

        var start = new XYZ(x0, y0, z0);
        var end   = new XYZ(x1, y1, z0);
        if (start.DistanceTo(end) < MmToFt(100.0))
        {
            Debug.WriteLine($"[ScanToBIM] Degenerate kerb length for segment {instruction.SegmentId}. Rejected.");
            return ElementId.InvalidElementId;
        }

        double kerbHeight = MmToFt(bb.MaxZ - bb.MinZ);
        if (kerbHeight < MmToFt(50)) kerbHeight = MmToFt(150);

        var level    = GetOrCreateLevel(z0, "Scan Level Kerb");
        var wallType = GetDefaultWallType();

        using var tx = new Transaction(_doc, $"ScanToBIM: Kerb [{ShortId(instruction)}]");
        tx.Start();

        var kerb = Wall.Create(_doc, Line.CreateBound(start, end), level.Id, false);
        TrySetLevelOffset(kerb, level, z0);
        kerb.get_Parameter(BuiltInParameter.WALL_USER_HEIGHT_PARAM)?.Set(kerbHeight);

        SafetyGateService.WriteComplianceParameters(kerb, instruction);

        tx.Commit();
        return kerb.Id;
    }

    // ── Generic Structural Family Instance ────────────────────────────────────
    // Used for element types that have no dedicated Revit API (ladder, pipe support,
    // overhead crane) — places a generic structural framing family at the centroid.

    private ElementId CreateGenericStructural(ElementInstruction instruction, string label)
    {
        var c        = instruction.Centroid;
        var bb       = instruction.BoundingBox;
        var pt       = AnchorPointToEnvelope(new XYZ(MmToFt(c.X), MmToFt(c.Y), MmToFtZ(bb.MinZ)), MmToFtZ(bb.MinZ));
        var level    = GetOrCreateLevel(pt.Z, $"Scan Level {label}");
        // Prefer structural framing (ladder, pipe support, crane); fallback to column
        var resolved = _resolver.Resolve(instruction, BuiltInCategory.OST_StructuralFraming);
        var sym      = resolved.Symbol ?? GetDefaultColumnSymbol();

        using var tx = new Transaction(_doc, $"ScanToBIM: {label} [{ShortId(instruction)}]");
        tx.Start();

        if (!sym.IsActive)
            sym.Activate();

        var inst = _doc.Create.NewFamilyInstance(
            pt, sym, level, StructuralType.NonStructural);
        TrySetLevelOffset(inst, level, pt.Z);

        SafetyGateService.WriteComplianceParameters(inst, instruction);

        tx.Commit();
        return inst.Id;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private Level GetOrCreateLevel(double elevationFt, string nameHint = "Scan Level")
    {
        var levels = new FilteredElementCollector(_doc)
            .OfClass(typeof(Level))
            .Cast<Level>()
            .OrderBy(l => l.Elevation)
            .ToList();

        if (levels.Count == 0)
        {
            using var txCreateFirst = new Transaction(_doc, $"ScanToBIM: Create {nameHint}");
            txCreateFirst.Start();
            var first = Level.Create(_doc, elevationFt);
            first.Name = $"{nameHint} {elevationFt * 304.8:F0}mm";
            txCreateFirst.Commit();
            return first;
        }

        var nearest = levels
            .OrderBy(l => Math.Abs(l.Elevation - elevationFt))
            .First();

        // Default behavior: reuse existing project levels (e.g., Level 0/Level 1)
        // so scan-derived elements follow the project level structure.
        var preferExistingLevels = ReadBoolEnv("STB_PREFER_EXISTING_LEVELS", true);
        if (preferExistingLevels)
            return nearest;

        // Prefer existing project levels (for example Level 0 / Level 1) when the
        // elevation is reasonably close, then use a per-element offset to keep Z exact.
        var deltas = levels
            .Zip(levels.Skip(1), (a, b) => b.Elevation - a.Elevation)
            .Where(d => d > MmToFt(100))
            .OrderBy(d => d)
            .ToList();
        double typicalSpacingFt = deltas.Count > 0 ? deltas[deltas.Count / 2] : MmToFt(3000);
        double snapToleranceFt = Math.Max(MmToFt(100), typicalSpacingFt * 0.45);
        if (Math.Abs(nearest.Elevation - elevationFt) <= snapToleranceFt)
            return nearest;

        // Create a new level for this elevation
        using var tx = new Transaction(_doc, $"ScanToBIM: Create {nameHint}");
        tx.Start();
        var level = Level.Create(_doc, elevationFt);
        level.Name = $"{nameHint} {elevationFt * 304.8:F0}mm";
        tx.Commit();
        return level;
    }

    private static bool ReadBoolEnv(string key, bool fallback)
    {
        var raw = Environment.GetEnvironmentVariable(key);
        if (string.IsNullOrWhiteSpace(raw))
            return fallback;
        raw = raw.Trim().ToLowerInvariant();
        return raw is "1" or "true" or "yes" or "on";
    }

    private static void TrySetLevelOffset(Element element, Level level, double targetElevationFt)
    {
        double offsetFt = targetElevationFt - level.Elevation;
        if (Math.Abs(offsetFt) < 1e-6)
            return;

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
            var p = element.get_Parameter(bip);
            if (p == null || p.IsReadOnly || p.StorageType != StorageType.Double)
                continue;

            try
            {
                p.Set(offsetFt);
                return;
            }
            catch
            {
                // Keep trying fallback parameters; some families expose only one.
            }
        }
    }

    private static double[] GetOrderedBoundingBoxSpansMm(BoundingBox bb)
    {
        return new[]
        {
            Math.Abs(bb.MaxX - bb.MinX),
            Math.Abs(bb.MaxY - bb.MinY),
            Math.Abs(bb.MaxZ - bb.MinZ),
        }
        .OrderBy(v => v)
        .ToArray();
    }

    private static double ClampMm(double valueMm, double minMm)
    {
        if (double.IsNaN(valueMm) || double.IsInfinity(valueMm))
            return minMm;
        return Math.Max(valueMm, minMm);
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

    private WallType GetDefaultWallType() =>
        new FilteredElementCollector(_doc)
            .OfClass(typeof(WallType))
            .Cast<WallType>()
            .Where(wt => wt.Kind == WallKind.Basic && !IsFoundationOrFooting(wt))
            .FirstOrDefault()
        ?? new FilteredElementCollector(_doc)
            .OfClass(typeof(WallType))
            .Cast<WallType>()
            .FirstOrDefault(wt => wt.Kind == WallKind.Basic)
        ?? throw new InvalidOperationException(
            "No basic wall type found in the project. " +
            "Load a wall type before running the agent.");

    /// <summary>
    /// Picks the basic wall type whose width best matches the detected wall
    /// thickness. Excludes foundation/footing/retaining types and prefers
    /// standard architectural basic wall types.
    /// </summary>
    private WallType GetWallTypeForThickness(double thicknessMm)
    {
        var clampedMm = Math.Min(Math.Max(thicknessMm, 100.0), 500.0);
        var targetFt  = MmToFt(clampedMm);

        var basicTypes = new FilteredElementCollector(_doc)
            .OfClass(typeof(WallType))
            .Cast<WallType>()
            .Where(wt => wt.Kind == WallKind.Basic)
            .ToList();

        var nonFoundation = basicTypes
            .Where(wt => !IsFoundationOrFooting(wt))
            .ToList();

        var pool = nonFoundation.Count > 0 ? nonFoundation : basicTypes;

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
            .OrderBy(wt => Math.Abs(wt.Width - targetFt))
            .FirstOrDefault()
            ?? GetDefaultWallType();
    }

    private FloorType GetDefaultFloorType() =>
        new FilteredElementCollector(_doc)
            .OfClass(typeof(FloorType))
            .Cast<FloorType>()
            .FirstOrDefault()
        ?? throw new InvalidOperationException(
            "No floor type found in the project. " +
            "Load a floor type before running the agent.");

    private DuctType GetDefaultDuctType() =>
        new FilteredElementCollector(_doc)
            .OfClass(typeof(DuctType))
            .Cast<DuctType>()
            .FirstOrDefault()
        ?? throw new InvalidOperationException(
            "No duct type found in the project. " +
            "Load a rectangular duct type before running the agent.");

    private MechanicalSystemType GetDefaultMechanicalSystemType() =>
        new FilteredElementCollector(_doc)
            .OfClass(typeof(MechanicalSystemType))
            .Cast<MechanicalSystemType>()
            .FirstOrDefault()
        ?? throw new InvalidOperationException(
            "No mechanical system type found in the project.");

    private ConduitType GetDefaultConduitType() =>
        new FilteredElementCollector(_doc)
            .OfClass(typeof(ConduitType))
            .Cast<ConduitType>()
            .FirstOrDefault()
        ?? throw new InvalidOperationException(
            "No conduit type found in the project. " +
            "Load a conduit type before running the agent.");

    private PipeType GetDefaultPipeType() =>
        new FilteredElementCollector(_doc)
            .OfClass(typeof(PipeType))
            .Cast<PipeType>()
            .FirstOrDefault()
        ?? throw new InvalidOperationException(
            "No pipe type found in the project. " +
            "Load a pipe type before running the agent.");

    private PipingSystemType GetPipingSystemTypeByColor(ElementInstruction instruction)
    {
        var allSystems = new FilteredElementCollector(_doc)
            .OfClass(typeof(PipingSystemType))
            .Cast<PipingSystemType>()
            .ToList();

        if (allSystems.Count == 0)
            throw new InvalidOperationException("No piping system type found.");

        string? color = null;
        if (instruction.Parameters?.TryGetValue("trace_color_hex", out var traceColorObj) == true)
            color = traceColorObj?.ToString();
        if (string.IsNullOrWhiteSpace(color)
            && instruction.Parameters?.TryGetValue("dominant_color", out var colorObj) == true)
            color = colorObj?.ToString();

        if (!string.IsNullOrWhiteSpace(color))
        {
            color = color.ToUpperInvariant();
            // Fire Protection (Red)
            if (color == "#FF0000") 
                return allSystems.FirstOrDefault(s => s.Name.Contains("Fire")) ?? allSystems.First();
            // Hydronic / Chilled Water (Blue)
            if (color.StartsWith("#0000FF") || color.StartsWith("#3333FF"))
                return allSystems.FirstOrDefault(s => s.Name.Contains("Chilled") || s.Name.Contains("Hydronic")) ?? allSystems.First();
            // Process / gas mains are often yellow in industrial facilities.
            if (color == "#FFFF00" || color == "#FFC000")
                return allSystems.FirstOrDefault(s =>
                    s.Name.Contains("Process") ||
                    s.Name.Contains("Gas") ||
                    s.Name.Contains("Steam"))
                    ?? allSystems.First();
        }
        return allSystems.First();
    }

    private MechanicalSystemType GetMechanicalSystemTypeByColor(ElementInstruction instruction)
    {
        var allSystems = new FilteredElementCollector(_doc)
            .OfClass(typeof(MechanicalSystemType))
            .Cast<MechanicalSystemType>()
            .ToList();

        if (instruction.Parameters?.TryGetValue("dominant_color", out var colorObj) == true)
        {
            var colorText = colorObj?.ToString();
            if (string.IsNullOrWhiteSpace(colorText))
                return allSystems.FirstOrDefault() ?? throw new InvalidOperationException("No mechanical system type found.");

            var color = colorText.ToUpperInvariant();
            // Supply Air (Blue/Cyan in some standards)
            if (color.Contains("0000FF") || color.Contains("00FFFF"))
                return allSystems.FirstOrDefault(s => s.Name.Contains("Supply")) ?? allSystems.First();
            // Return/Exhaust (Yellow/Orange)
            if (color.Contains("FF") && color.Contains("AA")) 
                return allSystems.FirstOrDefault(s => s.Name.Contains("Return") || s.Name.Contains("Exhaust")) ?? allSystems.First();
        }
        return allSystems.FirstOrDefault() ?? throw new InvalidOperationException("No mechanical system type found.");
    }

    private PipingSystemType GetDefaultPipingSystemType() =>
        new FilteredElementCollector(_doc)
            .OfClass(typeof(PipingSystemType))
            .Cast<PipingSystemType>()
            .FirstOrDefault()
        ?? throw new InvalidOperationException(
            "No piping system type found in the project.");
    private CableTrayType GetDefaultCableTrayType() =>
        new FilteredElementCollector(_doc)
            .OfClass(typeof(CableTrayType))
            .Cast<CableTrayType>()
            .FirstOrDefault()
        ?? throw new InvalidOperationException(
            "No cable tray type found in the project. " +
            "Load a cable tray type before running the agent.");

    private FamilySymbol GetDefaultValveSymbol() =>
        new FilteredElementCollector(_doc)
            .OfClass(typeof(FamilySymbol))
            .OfCategory(BuiltInCategory.OST_PipeAccessory)
            .Cast<FamilySymbol>()
            .FirstOrDefault()
        ?? throw new InvalidOperationException(
            "No pipe accessory family loaded in the project. " +
            "Load a gate valve or ball valve family before running the agent.");

    private FamilySymbol GetDefaultMepEquipmentSymbol(BuiltInCategory category) =>
        new FilteredElementCollector(_doc)
            .OfClass(typeof(FamilySymbol))
            .OfCategory(category)
            .Cast<FamilySymbol>()
            .FirstOrDefault()
        ?? throw new InvalidOperationException(
            $"No family symbol found for category {category}. " +
            "Load the required MEP equipment family before running the agent.");

    private FamilySymbol GetDefaultDoorSymbol() =>
        new FilteredElementCollector(_doc)
            .OfClass(typeof(FamilySymbol))
            .OfCategory(BuiltInCategory.OST_Doors)
            .Cast<FamilySymbol>()
            .FirstOrDefault()
        ?? throw new InvalidOperationException(
            "No door family loaded in the project. " +
            "Load a door family before running the agent.");

    private FamilySymbol GetDefaultWindowSymbol() =>
        new FilteredElementCollector(_doc)
            .OfClass(typeof(FamilySymbol))
            .OfCategory(BuiltInCategory.OST_Windows)
            .Cast<FamilySymbol>()
            .FirstOrDefault()
        ?? throw new InvalidOperationException(
            "No window family loaded in the project. " +
            "Load a window family before running the agent.");

    private RailingType GetDefaultRailingType() =>
        new FilteredElementCollector(_doc)
            .OfClass(typeof(RailingType))
            .Cast<RailingType>()
            .FirstOrDefault()
        ?? throw new InvalidOperationException(
            "No railing type found in the project. " +
            "Load a railing type before running the agent.");

    /// <summary>
    /// Finds the wall closest to <paramref name="point"/> within tolerance.
    /// Returns null if no wall is close enough — callers fall back to unhosted placement.
    /// </summary>
    private Wall? FindNearestWall(XYZ point, double toleranceFt = 0.66)
    {
        return new FilteredElementCollector(_doc)
            .OfClass(typeof(Wall))
            .Cast<Wall>()
            .Where(w => w.Location is LocationCurve lc && lc.Curve.Distance(point) <= toleranceFt)
            .OrderBy(w => ((LocationCurve)w.Location).Curve.Distance(point))
            .FirstOrDefault();
    }

    private static bool IsYellowPipeColor(ElementInstruction instruction)
    {
        if (instruction.Parameters == null
            || !instruction.Parameters.TryGetValue("dominant_color", out var colorObj)
            || colorObj == null)
            return false;

        var color = colorObj.ToString()?.ToUpperInvariant();
        return color == "#FFFF00" || color == "#FFC000";
    }

    private static XYZ AlignPointToWallHost(Wall? hostWall, XYZ desiredPoint, double zFt)
    {
        if (hostWall == null)
            return new XYZ(desiredPoint.X, desiredPoint.Y, zFt);

        if (hostWall.Location is not LocationCurve lc)
            return new XYZ(desiredPoint.X, desiredPoint.Y, zFt);

        var projection = lc.Curve.Project(new XYZ(desiredPoint.X, desiredPoint.Y, lc.Curve.GetEndPoint(0).Z));
        var onCurve = projection?.XYZPoint ?? desiredPoint;

        var bb = hostWall.get_BoundingBox(null);
        if (bb == null)
            return new XYZ(onCurve.X, onCurve.Y, zFt);

        const double insetFt = 0.02; // ~6 mm
        double x = Math.Max(bb.Min.X + insetFt, Math.Min(bb.Max.X - insetFt, onCurve.X));
        double y = Math.Max(bb.Min.Y + insetFt, Math.Min(bb.Max.Y - insetFt, onCurve.Y));
        double z = Math.Max(bb.Min.Z + insetFt, Math.Min(bb.Max.Z - insetFt, zFt));
        return new XYZ(x, y, z);
    }


    private static bool IsWithinXYTolerance(XYZ point, BoundingBoxXYZ bb, double tolFt)
    {
        return point.X >= bb.Min.X - tolFt
            && point.X <= bb.Max.X + tolFt
            && point.Y >= bb.Min.Y - tolFt
            && point.Y <= bb.Max.Y + tolFt;
    }

    private void ApplySemanticMetadata(Element element, ElementInstruction instruction)
    {
        var label = instruction.ElementType.ToString().Replace("_", " ").ToUpperInvariant();
        if (instruction.Parameters?.TryGetValue("component_label", out var rawLabel) == true)
        {
            var txt = rawLabel?.ToString();
            if (!string.IsNullOrWhiteSpace(txt))
                label = txt;
        }

        var typeText = instruction.ElementType.ToString();
        if (instruction.Parameters?.TryGetValue("component_type", out var rawType) == true)
        {
            var txt = rawType?.ToString();
            if (!string.IsNullOrWhiteSpace(txt))
                typeText = txt;
        }

        var embed = string.Empty;
        if (instruction.Parameters?.TryGetValue("semantic_embedding", out var rawEmbedding) == true)
            embed = rawEmbedding?.ToString() ?? string.Empty;

        var mark = $"{label}-{ShortId(instruction)}";
        var description = $"{label} ({typeText})";

        TrySetStringParam(element.get_Parameter(BuiltInParameter.ALL_MODEL_MARK), mark);
        TrySetStringParam(element.get_Parameter(BuiltInParameter.ALL_MODEL_DESCRIPTION), description);

        var commentsParam = element.get_Parameter(BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS);
        var existing = commentsParam?.AsString() ?? string.Empty;
        var semanticBlob = $"Label={label};Type={typeText};Embedding={embed}";
        var merged = string.IsNullOrWhiteSpace(existing) ? semanticBlob : $"{existing} | {semanticBlob}";
        TrySetStringParam(commentsParam, merged.Length > 950 ? merged[..950] : merged);

        // Optional custom shared params if the project has them loaded.
        TrySetStringParam(element.LookupParameter("ScanToBIM_ComponentLabel"), label);
        TrySetStringParam(element.LookupParameter("ScanToBIM_ComponentType"), typeText);
        TrySetStringParam(
            element.LookupParameter("ScanToBIM_ComponentEmbedding"),
            string.IsNullOrWhiteSpace(embed) ? string.Empty : (embed.Length > 950 ? embed[..950] : embed)
        );
    }

    private static void TrySetStringParam(Parameter? param, string value)
    {
        if (param == null || param.IsReadOnly || param.StorageType != StorageType.String)
            return;

        try
        {
            param.Set(value);
        }
        catch
        {
            // Metadata is best-effort and must not block element creation.
        }
    }

    private double ResolveInstructionZScale(ElementInstruction instruction)
    {
        bool isStairOrRamp = instruction.ElementType == ElementType.stair || instruction.ElementType == ElementType.ramp;

        string scope = Environment.GetEnvironmentVariable("STB_AUTO_Z_SCALE_SCOPE")?.Trim().ToLowerInvariant() ?? "all";
        if (TryGetParamString(instruction, "auto_z_scale_scope", out var scopeFromInstruction)
            && !string.IsNullOrWhiteSpace(scopeFromInstruction))
            scope = scopeFromInstruction.Trim().ToLowerInvariant();

        bool eligible = scope switch
        {
            "all" => true,
            "stairs" => isStairOrRamp,
            "stairs_only" => isStairOrRamp,
            _ => true,
        };

        if (!eligible)
            return 1.0;

        if (_envZScale.HasValue)
            return _envZScale.Value;

        // Auto recommendation is enabled by default, but can be disabled per-instruction.
        bool allowAutoScale = true;
        if (TryGetParamBool(instruction, "apply_auto_z_scale", out var applyAuto))
            allowAutoScale = applyAuto;

        if (!allowAutoScale)
            return 1.0;

        if (instruction.Parameters == null
            || !instruction.Parameters.TryGetValue("auto_z_scale_recommended", out var raw)
            || raw == null)
            return 1.0;

        double parsed;
        if (raw is System.Text.Json.JsonElement je)
        {
            if (je.ValueKind == System.Text.Json.JsonValueKind.Number && je.TryGetDouble(out var num))
                parsed = num;
            else if (je.ValueKind == System.Text.Json.JsonValueKind.String
                && double.TryParse(je.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var strNum))
                parsed = strNum;
            else
                return 1.0;
        }
        else if (!double.TryParse(raw.ToString(), NumberStyles.Float, CultureInfo.InvariantCulture, out parsed))
        {
            return 1.0;
        }

        return Math.Clamp(parsed, 1.0, 3.0);
    }

    private double ResolveInstructionZOffsetMm(ElementInstruction instruction)
    {
        bool isStairOrRamp = instruction.ElementType == ElementType.stair || instruction.ElementType == ElementType.ramp;

        string scope = Environment.GetEnvironmentVariable("STB_AUTO_Z_SCALE_SCOPE")?.Trim().ToLowerInvariant() ?? "all";
        if (TryGetParamString(instruction, "auto_z_scale_scope", out var scopeFromInstruction)
            && !string.IsNullOrWhiteSpace(scopeFromInstruction))
            scope = scopeFromInstruction.Trim().ToLowerInvariant();

        bool eligible = scope switch
        {
            "all" => true,
            "stairs" => isStairOrRamp,
            "stairs_only" => isStairOrRamp,
            _ => true,
        };

        if (!eligible)
            return 0.0;

        if (_envZOffsetMm.HasValue)
            return _envZOffsetMm.Value;

        bool allowAutoOffset = true;
        if (TryGetParamBool(instruction, "apply_auto_z_offset", out var applyOffset))
            allowAutoOffset = applyOffset;

        if (!allowAutoOffset)
            return 0.0;

        if (!TryGetParamDouble(instruction, "auto_z_offset_mm", out var offsetMm))
            return 0.0;

        return Math.Clamp(offsetMm, -20000.0, 20000.0);
    }

    private void TryJoinAndCutOverlappingWalls(Wall newWall)
    {
        var newBb = newWall.get_BoundingBox(null);
        if (newBb == null)
            return;

        var others = new FilteredElementCollector(_doc)
            .OfClass(typeof(Wall))
            .Cast<Wall>()
            .Where(w => w.Id != newWall.Id)
            .Where(w => BoundingBoxesOverlap(newBb, w.get_BoundingBox(null), MmToFt(20.0)))
            .ToList();

        foreach (var other in others)
        {
            if (!AreWallsNearCoincident(newWall, other))
                continue;

            try
            {
                if (!JoinGeometryUtils.AreElementsJoined(_doc, newWall, other))
                    JoinGeometryUtils.JoinGeometry(_doc, newWall, other);

                var newLen = TryGetWallLength(newWall);
                var otherLen = TryGetWallLength(other);

                // Prefer the longer wall as the host and cut/embed the shorter one.
                if (newLen <= otherLen && JoinGeometryUtils.IsCuttingElementInJoin(_doc, newWall, other))
                    JoinGeometryUtils.SwitchJoinOrder(_doc, newWall, other);
                else if (otherLen < newLen && !JoinGeometryUtils.IsCuttingElementInJoin(_doc, newWall, other))
                    JoinGeometryUtils.SwitchJoinOrder(_doc, newWall, other);
            }
            catch
            {
                // Best-effort cleanup; skip walls Revit refuses to join.
            }
        }
    }

    private bool ShouldPromoteRampToStair(ElementInstruction instruction, double riseFt, double runFt)
    {
        if (TryGetParamDouble(instruction, "stair_steps", out var taggedSteps) && taggedSteps >= 2.0)
            return true;

        if (runFt <= 1e-6)
            return false;

        var slope = riseFt / runFt;
        return riseFt >= MmToFt(700.0) && slope >= 0.35;
    }

    private bool AreWallsNearCoincident(Wall a, Wall b)
    {
        if (!TryGetWallLine2D(a, out var a0, out var a1, out var dirA, out var lenA))
            return false;
        if (!TryGetWallLine2D(b, out var b0, out var b1, out var dirB, out var lenB))
            return false;

        if (lenA < MmToFt(500.0) || lenB < MmToFt(500.0))
            return false;

        var alignment = Math.Abs(dirA.DotProduct(dirB));
        if (alignment < 0.97)
            return false;

        var bbA = a.get_BoundingBox(null);
        var bbB = b.get_BoundingBox(null);
        if (bbA == null || bbB == null)
            return false;

        var zOverlap = AxisOverlap(bbA.Min.Z, bbA.Max.Z, bbB.Min.Z, bbB.Max.Z);
        if (zOverlap < MmToFt(250.0))
            return false;

        var midA = new XYZ((a0.X + a1.X) * 0.5, (a0.Y + a1.Y) * 0.5, 0.0);
        var midB = new XYZ((b0.X + b1.X) * 0.5, (b0.Y + b1.Y) * 0.5, 0.0);
        var perp = new XYZ(-dirA.Y, dirA.X, 0.0);
        var centerGap = Math.Abs(perp.DotProduct(midB - midA));
        if (centerGap > MmToFt(220.0))
            return false;

        var aMin = Math.Min(dirA.DotProduct(a0), dirA.DotProduct(a1));
        var aMax = Math.Max(dirA.DotProduct(a0), dirA.DotProduct(a1));
        var bMin = Math.Min(dirA.DotProduct(b0), dirA.DotProduct(b1));
        var bMax = Math.Max(dirA.DotProduct(b0), dirA.DotProduct(b1));
        var overlap = AxisOverlap(aMin, aMax, bMin, bMax);
        var overlapRatio = overlap / Math.Max(Math.Min(lenA, lenB), 1e-6);

        return overlapRatio >= 0.65;
    }

    private static bool TryGetWallLine2D(Wall wall, out XYZ p0, out XYZ p1, out XYZ dir2D, out double length)
    {
        p0 = XYZ.Zero;
        p1 = XYZ.Zero;
        dir2D = XYZ.Zero;
        length = 0.0;

        if (wall.Location is not LocationCurve lc)
            return false;

        if (lc.Curve is not Line line)
            return false;

        var start = line.GetEndPoint(0);
        var end = line.GetEndPoint(1);
        var raw = end - start;
        var raw2D = new XYZ(raw.X, raw.Y, 0.0);
        var len2D = raw2D.GetLength();
        if (len2D <= 1e-6)
            return false;

        p0 = new XYZ(start.X, start.Y, 0.0);
        p1 = new XYZ(end.X, end.Y, 0.0);
        dir2D = raw2D.Normalize();
        length = len2D;
        return true;
    }

    private static double AxisOverlap(double aMin, double aMax, double bMin, double bMax)
    {
        return Math.Max(0.0, Math.Min(aMax, bMax) - Math.Max(aMin, bMin));
    }

    private void TryConnectPipeToNearbyNetwork(Pipe newPipe, XYZ startPoint, XYZ endPoint, double expectedDiameterFt)
    {
        var endConnectors = GetPipeEndConnectors(newPipe).ToList();
        if (endConnectors.Count == 0)
            return;

        var targetPoints = new[] { startPoint, endPoint };
        foreach (var targetPoint in targetPoints)
        {
            var source = endConnectors
                .Where(c => !c.IsConnected)
                .OrderBy(c => c.Origin.DistanceTo(targetPoint))
                .FirstOrDefault();
            if (source == null)
                continue;

            var nearby = FindNearestOpenPipeConnector(source, newPipe.Id, expectedDiameterFt, MmToFt(450.0));
            if (nearby == null)
                continue;

            TryCreatePipeFitting(source, nearby);
        }
    }

    private IEnumerable<Connector> GetPipeEndConnectors(Pipe pipe)
    {
        foreach (Connector connector in pipe.ConnectorManager.Connectors)
        {
            if (connector.ConnectorType == ConnectorType.End)
                yield return connector;
        }
    }

    private Connector? FindNearestOpenPipeConnector(
        Connector source,
        ElementId sourcePipeId,
        double expectedDiameterFt,
        double toleranceFt)
    {
        Connector? best = null;
        double bestDist = double.MaxValue;

        var candidatePipes = new FilteredElementCollector(_doc)
            .OfClass(typeof(Pipe))
            .Cast<Pipe>()
            .Where(p => p.Id != sourcePipeId);

        foreach (var pipe in candidatePipes)
        {
            foreach (var connector in GetPipeEndConnectors(pipe))
            {
                if (connector.IsConnected)
                    continue;

                if (!IsPipeDiameterCompatible(connector, expectedDiameterFt))
                    continue;

                var distance = source.Origin.DistanceTo(connector.Origin);
                if (distance > toleranceFt)
                    continue;

                if (distance < bestDist)
                {
                    bestDist = distance;
                    best = connector;
                }
            }
        }

        return best;
    }

    private static bool IsPipeDiameterCompatible(Connector connector, double expectedDiameterFt)
    {
        try
        {
            var connectorDiameter = connector.Radius * 2.0;
            if (connectorDiameter <= 1e-6 || expectedDiameterFt <= 1e-6)
                return true;

            var ratio = connectorDiameter / expectedDiameterFt;
            return ratio >= 0.55 && ratio <= 1.8;
        }
        catch
        {
            return true;
        }
    }

    private void TryCreatePipeFitting(Connector a, Connector b)
    {
        if (a.IsConnected || b.IsConnected)
            return;

        try
        {
            if (a.Origin.DistanceTo(b.Origin) <= MmToFt(35.0))
            {
                _doc.Create.NewUnionFitting(a, b);
                return;
            }
        }
        catch
        {
            // Fall through to elbow/connect fallback.
        }

        try
        {
            _doc.Create.NewElbowFitting(a, b);
            return;
        }
        catch
        {
            // Fall through to direct connector join fallback.
        }

        try
        {
            a.ConnectTo(b);
        }
        catch
        {
            // Best-effort connection only.
        }
    }

    private sealed class StairFailureSilencer : IFailuresPreprocessor
    {
        public FailureProcessingResult PreprocessFailures(FailuresAccessor failuresAccessor)
        {
            var messages = failuresAccessor.GetFailureMessages();
            foreach (var message in messages)
            {
                if (message.GetSeverity() == FailureSeverity.Warning)
                    failuresAccessor.DeleteWarning(message);
            }

            return FailureProcessingResult.Continue;
        }
    }

    private static bool BoundingBoxesOverlap(BoundingBoxXYZ? a, BoundingBoxXYZ? b, double tolFt)
    {
        if (a == null || b == null)
            return false;

        return a.Min.X <= b.Max.X + tolFt && a.Max.X >= b.Min.X - tolFt
            && a.Min.Y <= b.Max.Y + tolFt && a.Max.Y >= b.Min.Y - tolFt
            && a.Min.Z <= b.Max.Z + tolFt && a.Max.Z >= b.Min.Z - tolFt;
    }

    private static double TryGetWallLength(Wall wall)
    {
        if (wall.Location is LocationCurve lc)
            return lc.Curve.Length;

        return 0.0;
    }

    private static bool TryGetParamDouble(ElementInstruction instruction, string key, out double value)
    {
        value = 0.0;
        if (instruction.Parameters == null || !instruction.Parameters.TryGetValue(key, out var raw))
            return false;

        if (raw is System.Text.Json.JsonElement je)
        {
            if (je.ValueKind == System.Text.Json.JsonValueKind.Number && je.TryGetDouble(out var num))
            {
                value = num;
                return true;
            }
            if (je.ValueKind == System.Text.Json.JsonValueKind.String && double.TryParse(je.GetString(), out var parsed))
            {
                value = parsed;
                return true;
            }
            return false;
        }

        try
        {
            value = Convert.ToDouble(raw);
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool TryGetParamBool(ElementInstruction instruction, string key, out bool value)
    {
        value = false;
        if (instruction.Parameters == null || !instruction.Parameters.TryGetValue(key, out var raw))
            return false;

        if (raw is System.Text.Json.JsonElement je)
        {
            if (je.ValueKind == System.Text.Json.JsonValueKind.True)
            {
                value = true;
                return true;
            }
            if (je.ValueKind == System.Text.Json.JsonValueKind.False)
            {
                value = false;
                return true;
            }
            if (je.ValueKind == System.Text.Json.JsonValueKind.String
                && bool.TryParse(je.GetString(), out var parsedBool))
            {
                value = parsedBool;
                return true;
            }
            return false;
        }

        if (bool.TryParse(raw?.ToString(), out var parsed))
        {
            value = parsed;
            return true;
        }

        return false;
    }

    private static bool TryGetParamString(ElementInstruction instruction, string key, out string? value)
    {
        value = null;
        if (instruction.Parameters == null || !instruction.Parameters.TryGetValue(key, out var raw) || raw == null)
            return false;

        if (raw is System.Text.Json.JsonElement je)
        {
            if (je.ValueKind == System.Text.Json.JsonValueKind.String)
            {
                value = je.GetString();
                return !string.IsNullOrWhiteSpace(value);
            }

            value = je.ToString();
            return !string.IsNullOrWhiteSpace(value);
        }

        value = raw.ToString();
        return !string.IsNullOrWhiteSpace(value);
    }

    private IList<GeometryObject> BuildSteppedStairGeometry(ElementInstruction instruction)
    {
        var bb = instruction.BoundingBox;

        double minX = MmToFt(bb.MinX);
        double maxX = MmToFt(bb.MaxX);
        double minY = MmToFt(bb.MinY);
        double maxY = MmToFt(bb.MaxY);
        double minZ = MmToFtZ(bb.MinZ);
        double maxZ = MmToFtZ(bb.MaxZ);

        double runX = Math.Max(maxX - minX, MmToFt(900));
        double runY = Math.Max(maxY - minY, MmToFt(900));
        // Build stairs from detected extents; only apply extra rise when
        // explicitly requested by metadata.
        double rise = Math.Max(maxZ - minZ, MmToFt(120));
        if (TryGetParamDouble(instruction, "min_storey_rise_mm", out var minRiseMm) && minRiseMm > 0.0)
            rise = Math.Max(rise, MmToFt(minRiseMm));

        bool alongX = runX >= runY;
        double runLength = alongX ? runX : runY;
        double runWidth = Math.Max(alongX ? runY : runX, MmToFt(900));
        runWidth = Math.Min(runWidth, MmToFt(3500));

        int stepCount = 0;
        if (instruction.Parameters != null
            && instruction.Parameters.TryGetValue("stair_steps", out var stepsObj)
            && int.TryParse(stepsObj?.ToString(), out var parsed)
            && parsed > 0)
        {
            stepCount = parsed;
        }
        if (stepCount <= 0)
            stepCount = (int)Math.Round(rise / MmToFt(170.0));
        stepCount = Math.Max(3, Math.Min(stepCount, 40));

        double treadDepth = runLength / stepCount;
        double riser = rise / stepCount;

        var solids = new List<GeometryObject>(stepCount);
        for (int i = 0; i < stepCount; i++)
        {
            double x0 = alongX ? minX + i * treadDepth : minX;
            double x1 = alongX ? minX + (i + 1) * treadDepth : minX + runWidth;
            double y0 = alongX ? minY : minY + i * treadDepth;
            double y1 = alongX ? minY + runWidth : minY + (i + 1) * treadDepth;
            double z0 = minZ + i * riser;
            double z1 = z0 + riser;

            var solid = BuildBoxSolid(x0, y0, z0, x1, y1, z1);
            if (solid != null)
                solids.Add(solid);
        }

        // Safety fallback: keep a visible stair mass if step solids fail.
        if (solids.Count == 0)
        {
            var fallback = BuildBoxSolid(minX, minY, minZ, maxX, maxY, minZ + rise);
            if (fallback != null)
                solids.Add(fallback);
        }

        return solids;
    }

    private static Solid? BuildBoxSolid(double x0, double y0, double z0, double x1, double y1, double z1)
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

    private CeilingType GetDefaultCeilingType() =>
        new FilteredElementCollector(_doc)
            .OfClass(typeof(CeilingType))
            .Cast<CeilingType>()
            .FirstOrDefault()
        ?? throw new InvalidOperationException(
            "No ceiling type found in the project. " +
            "Load a ceiling type before running the agent.");

    private FamilySymbol GetDefaultBeamSymbol() =>
        new FilteredElementCollector(_doc)
            .OfClass(typeof(FamilySymbol))
            .OfCategory(BuiltInCategory.OST_StructuralFraming)
            .Cast<FamilySymbol>()
            .FirstOrDefault()
        ?? throw new InvalidOperationException(
            "No structural framing family loaded in the project. " +
            "Load a beam family (e.g. W-Wide Flange) before running the agent.");

    private StairsType GetDefaultStairType() =>
        new FilteredElementCollector(_doc)
            .OfClass(typeof(StairsType))
            .Cast<StairsType>()
            .FirstOrDefault()
        ?? throw new InvalidOperationException(
            "No stair type found in the project. " +
            "Load a stair type before running the agent.");

    private FamilySymbol GetDefaultColumnSymbol() =>
        new FilteredElementCollector(_doc)
            .OfClass(typeof(FamilySymbol))
            .OfCategory(BuiltInCategory.OST_StructuralColumns)
            .Cast<FamilySymbol>()
            .FirstOrDefault()
        ?? throw new InvalidOperationException(
            "No structural column family loaded in the project. " +
            "Load a column family (e.g. W-Wide Flange) before running the agent.");

    /// <summary>Millimetres to decimal feet (Revit internal unit).</summary>
    private static double MmToFt(double mm) => mm / 304.8;

    private double MmToFtZ(double mm) => MmToFt(mm * _instructionZScale + _instructionZOffsetMm);

    private static double? ReadVerticalScaleFromEnv()
    {
        var raw = Environment.GetEnvironmentVariable("STB_Z_SCALE");
        if (string.IsNullOrWhiteSpace(raw))
            return null;

        if (double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed))
            return Math.Clamp(parsed, 0.5, 10.0);

        return null;
    }

    private static double? ReadVerticalOffsetFromEnv()
    {
        var raw = Environment.GetEnvironmentVariable("STB_Z_OFFSET_MM");
        if (string.IsNullOrWhiteSpace(raw))
            return null;

        if (double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed))
            return Math.Clamp(parsed, -20000.0, 20000.0);

        return null;
    }

    /// <summary>Short ID for transaction names and logging.</summary>
    private static string ShortId(ElementInstruction i) =>
        i.InstructionId.Length >= 8 ? i.InstructionId[..8] : i.InstructionId;

    private XYZ AnchorPointToEnvelope(XYZ desiredPoint, double preferredZFt)
    {
        const double wallSnapTolFt = 0.66; // ~200 mm
        const double maxVerticalSnapFt = 1.64; // ~500 mm

        var hostWall = FindNearestWall(desiredPoint, wallSnapTolFt);
        if (hostWall != null)
            return AlignPointToWallHost(hostWall, desiredPoint, preferredZFt);

        double nearestHorizontalZ = preferredZFt;
        double bestDz = double.MaxValue;
        const double xyTolFt = 3.28; // ~1 m

        foreach (var floor in new FilteredElementCollector(_doc).OfClass(typeof(Floor)).Cast<Floor>())
        {
            var bb = floor.get_BoundingBox(null);
            if (bb == null) continue;
            if (!IsWithinXYTolerance(desiredPoint, bb, xyTolFt)) continue;

            double zCandidate = bb.Max.Z;
            double dz = Math.Abs(zCandidate - preferredZFt);
            if (dz > maxVerticalSnapFt) continue;
            if (dz < bestDz)
            {
                bestDz = dz;
                nearestHorizontalZ = zCandidate;
            }
        }

        foreach (var ceiling in new FilteredElementCollector(_doc).OfClass(typeof(Ceiling)).Cast<Ceiling>())
        {
            var bb = ceiling.get_BoundingBox(null);
            if (bb == null) continue;
            if (!IsWithinXYTolerance(desiredPoint, bb, xyTolFt)) continue;

            double zCandidate = bb.Min.Z;
            double dz = Math.Abs(zCandidate - preferredZFt);
            if (dz > maxVerticalSnapFt) continue;
            if (dz < bestDz)
            {
                bestDz = dz;
                nearestHorizontalZ = zCandidate;
            }
        }

        return new XYZ(desiredPoint.X, desiredPoint.Y, nearestHorizontalZ);
    }

    public static Solid? BuildCylinderSolid(XYZ start, XYZ end, double radius)
    {
        double height = start.DistanceTo(end);
        if (height <= 1e-6 || radius <= 1e-6)
            return null;

        var axis = (end - start).Normalize();
        XYZ refDir = Math.Abs(axis.DotProduct(XYZ.BasisZ)) > 0.99 ? XYZ.BasisX : XYZ.BasisZ;
        XYZ u = axis.CrossProduct(refDir).Normalize();
        XYZ v = axis.CrossProduct(u).Normalize();

        var loop = new CurveLoop();
        
        var center = start;
        var rVec = u.Multiply(radius);
        var normal = axis;

        var p0 = center + rVec;
        var p1 = center - rVec;
        var pMiddle1 = center + v.Multiply(radius);
        var arc1 = Arc.Create(p0, p1, pMiddle1);

        var pMiddle2 = center - v.Multiply(radius);
        var arc2 = Arc.Create(p1, p0, pMiddle2);

        loop.Append(arc1);
        loop.Append(arc2);

        try
        {
            var solid = GeometryCreationUtilities.CreateExtrusionGeometry(
                new List<CurveLoop> { loop },
                normal,
                height);
            return solid;
        }
        catch
        {
            return null;
        }
    }

    private static (double MinX, double MaxX, double MinZ, double MaxZ) GetRevitFloorAndWallBoundaries(Document doc)
    {
        double minX = double.MaxValue;
        double maxX = double.MinValue;
        double minZ = 0.0;
        double maxZ = 4000.0 * (1.0 / 304.8); // 4m fallback in feet

        var floors = new FilteredElementCollector(doc)
            .OfClass(typeof(Floor))
            .Cast<Floor>()
            .ToList();

        if (floors.Count > 0)
        {
            foreach (var floor in floors)
            {
                var bb = floor.get_BoundingBox(null);
                if (bb != null)
                {
                    minX = Math.Min(minX, bb.Min.X);
                    maxX = Math.Max(maxX, bb.Max.X);
                    minZ = Math.Min(minZ, bb.Min.Z);
                }
            }
        }

        var walls = new FilteredElementCollector(doc)
            .OfClass(typeof(Wall))
            .Cast<Wall>()
            .ToList();

        if (walls.Count > 0)
        {
            foreach (var wall in walls)
            {
                var bb = wall.get_BoundingBox(null);
                if (bb != null)
                {
                    minX = Math.Min(minX, bb.Min.X);
                    maxX = Math.Max(maxX, bb.Max.X);
                    maxZ = Math.Max(maxZ, bb.Max.Z);
                }
            }
        }

        if (minX == double.MaxValue || maxX == double.MinValue)
        {
            minX = -2000.0 * (1.0 / 304.8);
            maxX = 2000.0 * (1.0 / 304.8);
        }

        return (minX, maxX, minZ, maxZ);
    }

}
