using Autodesk.Revit.UI;
using ScanToBIM.Services;
using System.Windows.Controls;

namespace ScanToBIM.UI;

/// <summary>
/// Code-behind for the ScanToBIM Agent dockable panel.
/// A single instance is created by the DockablePaneProvider in App.cs
/// and shown/hidden via Revit's dockable pane API.
/// </summary>
public partial class AgentPanelPage : Page
{
    // Singleton panel instance — Revit only allows one instance per pane ID
    private static AgentPanelPage?   _instance;
    private static AgentBridgeService? _bridge;

    public AgentPanelViewModel? ViewModel { get; private set; }

    public AgentPanelPage(UIApplication? uiApp)
    {
        InitializeComponent();
        // Do not initialize bridge or ViewModel yet if uiApp is null (created by DockablePaneProvider)
        if (uiApp != null)
            Initialize(uiApp);
    }

    // Safe initialization method
    public void Initialize(UIApplication uiApp)
    {
        if (_bridge == null)
        {
            _bridge = new AgentBridgeService(uiApp);
        }
        if (ViewModel == null)
        {
            ViewModel = new AgentPanelViewModel(_bridge);
            DataContext = ViewModel;
        }
    }

    /// <summary>
    /// Show the agent panel. Creates it on first call; shows it on subsequent calls.
    /// </summary>
    public static void ShowOrCreate(UIApplication uiApp)
    {
        if (uiApp == null)
        {
            System.Windows.MessageBox.Show(
                "Revit UIApplication is not available. Cannot open agent panel.",
                "ScanToBIM Agent — Startup Error",
                System.Windows.MessageBoxButton.OK,
                System.Windows.MessageBoxImage.Error);
            return;
        }

        if (_instance == null)
        {
            _instance = new AgentPanelPage(null);
        }
        _instance.Initialize(uiApp);

        var dpid = new DockablePaneId(App.PanelGuid);
        var dp = uiApp.GetDockablePane(dpid);
        if (dp == null)
        {
            System.Windows.MessageBox.Show(
                "ScanToBIM Agent panel is not registered. Please restart Revit or contact support.",
                "ScanToBIM Agent — Startup Error",
                System.Windows.MessageBoxButton.OK,
                System.Windows.MessageBoxImage.Error);
            return;
        }
        dp.Show();
    }

    /// <summary>The bridge service — exposed so App.cs can dispose on shutdown.</summary>
    public static AgentBridgeService? Bridge => _bridge;
}
