using System;
using System.Collections.Generic;
using System.Linq;
using Autodesk.Revit.DB;

namespace ScanToBIM.Services
{
    public static class GeometryUtils
    {
        // Simplifies a CurveLoop by removing duplicate and colinear points
        public static CurveLoop SimplifyCurveLoop(CurveLoop loop, double tolerance = 1e-6)
        {
            var points = loop.Select(c => c.GetEndPoint(0)).ToList();
            var simplified = new List<XYZ>();
            for (int i = 0; i < points.Count; i++)
            {
                var prev = points[(i - 1 + points.Count) % points.Count];
                var curr = points[i];
                var next = points[(i + 1) % points.Count];
                // Remove duplicate
                if (curr.IsAlmostEqualTo(prev, tolerance)) continue;
                // Remove colinear
                var v1 = (curr - prev).Normalize();
                var v2 = (next - curr).Normalize();
                if (v1.IsAlmostEqualTo(v2, tolerance)) continue;
                simplified.Add(curr);
            }
            // Rebuild CurveLoop
            var curves = new List<Curve>();
            for (int i = 0; i < simplified.Count; i++)
            {
                var a = simplified[i];
                var b = simplified[(i + 1) % simplified.Count];
                curves.Add(Line.CreateBound(a, b));
            }
            var result = new CurveLoop();
            foreach (var c in curves) result.Append(c);
            return result;
        }

        // Validates a CurveLoop (closed, non-self-intersecting, min 3 points)
        public static bool IsValidCurveLoop(CurveLoop loop, double tolerance = 1e-6)
        {
            var points = loop.Select(c => c.GetEndPoint(0)).ToList();
            if (points.Count < 3) return false;
            // Check closure
            if (!points.First().IsAlmostEqualTo(points.Last(), tolerance)) return false;
            // TODO: Add self-intersection check if needed
            return true;
        }

        // Subtracts opening loops from a main profile (for floors/ceilings)
        public static List<CurveLoop> SubtractOpenings(CurveLoop main, List<CurveLoop> openings)
        {
            var loops = new List<CurveLoop> { main };
            if (openings != null)
            {
                foreach (var o in openings)
                {
                    if (IsValidCurveLoop(o))
                        loops.Add(o);
                }
            }
            return loops;
        }
    }
}
