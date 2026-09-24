// ─────────────────────────────────────────────────────────────────────────────
// ScanResultStore.cs — Keeps latest scan results in memory and on disk
// ─────────────────────────────────────────────────────────────────────────────
using System.IO;
using System.Text.Json;
using ScanToBIM.Models;

namespace ScanToBIM.Services
{
    public static class ScanResultStore
    {
        public static SidecarResults? Latest { get; set; }
        public static string? LatestResultsPath { get; set; }

        // Loads results from disk if present
        public static void LoadFromDisk()
        {
            if (!string.IsNullOrEmpty(LatestResultsPath) && File.Exists(LatestResultsPath))
            {
                try
                {
                    var json = File.ReadAllText(LatestResultsPath);
                    Latest = JsonSerializer.Deserialize<SidecarResults>(json);
                }
                catch { /* ignore errors */ }
            }
        }
    }
}
