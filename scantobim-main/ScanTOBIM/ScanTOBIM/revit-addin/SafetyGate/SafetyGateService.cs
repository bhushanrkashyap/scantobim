// ─────────────────────────────────────────────────────────────────────────────
// SafetyGateService.cs
//
// CRITICAL SAFETY CONTROLS — READ BEFORE MODIFYING.
//
// This class enforces the two non-negotiable safety rules for the system:
//
//   SC1 HARD BLOCK:
//     Safety-critical elements (NQA-1 SC1) MUST NEVER be auto-created by the
//     agent under any circumstances. AssertNotSC1() throws unconditionally for
//     any SC1 instruction. There is no override, no bypass, no exception path.
//
//   SC2 APPROVAL REQUIRED:
//     Safety-related elements (SC2) require a valid approval signature from a
//     qualified engineer before the element can be created. AssertSC2HasApproval()
//     blocks creation if the signature is absent or empty.
//
// Both methods MUST be called at the start of every create_* operation,
// before any Revit API transaction is opened. The order is:
//   1. AssertNotSC1(instruction)
//   2. AssertSC2HasApproval(instruction)
//
// Tests for these methods must achieve 100% coverage. See Tests/SafetyGateServiceTests.cs.
// ─────────────────────────────────────────────────────────────────────────────

using Autodesk.Revit.DB;
using ScanToBIM.Models;

namespace ScanToBIM.SafetyGate;

// ── Exceptions ────────────────────────────────────────────────────────────────

/// <summary>
/// Raised when the agent attempts to auto-create an SC1 safety-critical element.
/// This is a hard block — there is no recovery path. The element must be created
/// manually after documented NQA-1 review.
/// </summary>
public sealed class SafetyCategoryViolationException : Exception
{
    public string         SegmentId      { get; }
    public SafetyCategory SafetyCategory { get; }

    public SafetyCategoryViolationException(string segmentId, SafetyCategory category, string message)
        : base(message)
    {
        SegmentId      = segmentId;
        SafetyCategory = category;
    }
}

/// <summary>
/// Raised when an SC2 element instruction arrives without an approval signature.
/// The element remains pending until a qualified engineer approves via the agent API.
/// </summary>
public sealed class SC2ApprovalRequiredException : Exception
{
    public string SegmentId { get; }

    public SC2ApprovalRequiredException(string segmentId, string message)
        : base(message)
    {
        SegmentId = segmentId;
    }
}

// ── Service ───────────────────────────────────────────────────────────────────

public static class SafetyGateService
{
    // ── SC1 Hard Block ────────────────────────────────────────────────────────

    /// <summary>
    /// SC1 Hard Block gate.
    ///
    /// MUST be the very first call in every element-creation method.
    /// No Revit transaction should be opened before this check passes.
    ///
    /// Throws <see cref="SafetyCategoryViolationException"/> for any SC1 instruction.
    /// For SC2, SC3, NS: returns without throwing (caller may then check SC2 approval).
    /// </summary>
    public static void AssertNotSC1(ElementInstruction instruction)
    {
        if (instruction.SafetyCategory == SafetyCategory.SC1)
        {
            throw new SafetyCategoryViolationException(
                instruction.SegmentId,
                SafetyCategory.SC1,
                $"SC1 HARD BLOCK — Segment '{instruction.SegmentId}' is safety-critical (SC1). " +
                $"Element type '{instruction.ElementType}' in zone '{instruction.ZoneId}' " +
                "CANNOT be auto-created by the agent. " +
                "Manual NQA-1 review and documented approval required by a licensed engineer."
            );
        }
    }

    // ── SC2 Approval Gate ─────────────────────────────────────────────────────

    /// <summary>
    /// SC2 Approval gate.
    ///
    /// MUST be called after AssertNotSC1() and before opening any Revit transaction.
    ///
    /// For SC2 instructions: throws <see cref="SC2ApprovalRequiredException"/> if
    /// ApprovalSignature is null or whitespace.
    ///
    /// For SC1, SC3, NS: no-op (SC1 is already blocked by AssertNotSC1).
    /// </summary>
    public static void AssertSC2HasApproval(ElementInstruction instruction)
    {
        if (instruction.SafetyCategory != SafetyCategory.SC2)
            return;

        if (string.IsNullOrWhiteSpace(instruction.ApprovalSignature))
        {
            throw new SC2ApprovalRequiredException(
                instruction.SegmentId,
                $"SC2 APPROVAL REQUIRED — Segment '{instruction.SegmentId}' is safety-related (SC2). " +
                $"Element type '{instruction.ElementType}' requires a valid engineer approval signature. " +
                $"Approver UPN: {instruction.ApproverUpn ?? "(not set)"}. " +
                "Approve this gate via the agent API: POST /safety-gates/{gate_id}/decision"
            );
        }
    }

    // ── Compliance Parameter Writer ───────────────────────────────────────────

    /// <summary>
    /// Writes the 4 ScanToBIM compliance shared parameters to a Revit element.
    ///
    /// MUST be called inside an open Revit Transaction, after the element has
    /// been created and before the transaction is committed.
    ///
    /// Parameters written:
    ///   ScanToBIM_SafetyCategory  — SC1 / SC2 / SC3 / NS
    ///   ScanToBIM_ScanStationId   — source segment UUID (traceability)
    ///   ScanToBIM_SegmentId       — same as ScanStationId for PoV
    ///   ScanToBIM_AgentSession    — instruction UUID (agent session trace)
    /// </summary>
    public static void WriteComplianceParameters(Autodesk.Revit.DB.Element element, ElementInstruction instruction)
    {
        // ── Original 4 params ────────────────────────────────────────────────
        SetParam(element, SharedParamNames.SafetyCategory, instruction.SafetyCategory.ToString());
        SetParam(element, SharedParamNames.ScanStationId,  instruction.SegmentId);
        SetParam(element, SharedParamNames.SegmentId,      instruction.SegmentId);
        SetParam(element, SharedParamNames.AgentSession,   instruction.InstructionId);

        // ── P4 additional 5 params ───────────────────────────────────────────
        SetParam(element, SharedParamNames.DsearZone,
            instruction.DsearZone ?? "None");
        SetParam(element, SharedParamNames.ApprovalSignature,
            instruction.ApprovalSignature ?? string.Empty);
        SetParam(element, SharedParamNames.ApproverUpn,
            instruction.ApproverUpn ?? string.Empty);
        SetParam(element, SharedParamNames.ApprovedAtUtc,
            instruction.ApprovedAtUtc?.ToString("o") ?? string.Empty);
        SetParam(element, SharedParamNames.NcrRef, string.Empty); // populated later if NCR raised

        // ── Prepend systematic comment metadata ──────────────────────────────
        var labelPrefix = "Element";
        var elType = instruction.ElementType;
        if (elType == ScanToBIM.Models.ElementType.stair)
            labelPrefix = "Staircase";
        else if (elType == ScanToBIM.Models.ElementType.wall || elType == ScanToBIM.Models.ElementType.bund_wall)
            labelPrefix = "Wall";
        else if (elType == ScanToBIM.Models.ElementType.pipe)
            labelPrefix = "Pipe";
        else if (elType == ScanToBIM.Models.ElementType.conduit || elType == ScanToBIM.Models.ElementType.cable_tray)
            labelPrefix = "Wire";
        else if (elType == ScanToBIM.Models.ElementType.duct)
            labelPrefix = "Duct";
        else if (elType == ScanToBIM.Models.ElementType.floor || elType == ScanToBIM.Models.ElementType.grating)
            labelPrefix = "Floor";
        else if (elType == ScanToBIM.Models.ElementType.ceiling)
            labelPrefix = "Ceiling";
        else if (elType == ScanToBIM.Models.ElementType.beam)
            labelPrefix = "Beam";
        else if (elType == ScanToBIM.Models.ElementType.column)
            labelPrefix = "Column";
        else
            labelPrefix = elType.ToString();

        if (labelPrefix.Length > 0)
        {
            labelPrefix = char.ToUpper(labelPrefix[0]) + labelPrefix.Substring(1);
        }

        var commentsParam = element.get_Parameter(BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS);
        if (commentsParam != null && !commentsParam.IsReadOnly)
        {
            var existingVal = commentsParam.AsString() ?? "";
            var newVal = $"{labelPrefix} | SegmentId={instruction.SegmentId} | Confidence={instruction.SafetyCategory.ToString()}";
            if (!string.IsNullOrEmpty(existingVal))
            {
                newVal += " | " + existingVal;
            }
            commentsParam.Set(newVal);
        }
    }

    // ── Internal Helpers ──────────────────────────────────────────────────────

    private static void SetParam(Autodesk.Revit.DB.Element element, string paramName, string value)
    {
        var param = element.LookupParameter(paramName);
        if (param is { IsReadOnly: false })
            param.Set(value);
        // Silently skip missing params so the element is still created even if
        // shared params haven't been loaded yet. The agent panel will log a warning.
    }
}

// ── Shared Parameter Name Constants ───────────────────────────────────────────

/// <summary>
/// Canonical names for all ScanToBIM shared parameters.
/// Must match the entries in Resources/ScanToBIM.SharedParameters.txt exactly.
/// </summary>
public static class SharedParamNames
{
    // ── Original 4 (PoV) ────────────────────────────────────────────────────
    public const string SafetyCategory    = "ScanToBIM_SafetyCategory";
    public const string ScanStationId     = "ScanToBIM_ScanStationId";
    public const string SegmentId         = "ScanToBIM_SegmentId";
    public const string AgentSession      = "ScanToBIM_AgentSession";

    // ── P4 additions (9 total) ───────────────────────────────────────────────
    public const string DsearZone         = "ScanToBIM_DSEARZone";
    public const string ApprovalSignature = "ScanToBIM_ApprovalSignature";
    public const string ApproverUpn       = "ScanToBIM_ApproverUPN";
    public const string ApprovedAtUtc     = "ScanToBIM_ApprovedAtUtc";
    public const string NcrRef            = "ScanToBIM_NCRRef";

    // ── P2.2a USIBD LOA v3.1 additions (12 total) ────────────────────────────
    public const string LoaTier           = "ScanToBIM_LOATier";        // LOA10–LOA50
    public const string LoaSigmaMm        = "ScanToBIM_LOASigmaMm";     // numeric, 1σ residual
    public const string LoaStandardVersion = "ScanToBIM_LOAStandard";   // e.g. "USIBD LOA v3.1"

    /// <summary>All 12 parameter names — used by SharedParameterLoader.</summary>
    public static readonly string[] All =
    [
        SafetyCategory, ScanStationId, SegmentId, AgentSession,
        DsearZone, ApprovalSignature, ApproverUpn, ApprovedAtUtc, NcrRef,
        LoaTier, LoaSigmaMm, LoaStandardVersion,
    ];
}
