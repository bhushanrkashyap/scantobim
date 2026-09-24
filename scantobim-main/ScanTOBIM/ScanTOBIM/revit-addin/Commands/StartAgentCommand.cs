// ─────────────────────────────────────────────────────────────────────────────
// StartAgentCommand.cs
//
// IExternalCommands mapped to ribbon buttons in App.cs.
//   StartAgentCommand      → "Start Agent"  button — opens/shows the agent panel
//   LoadSharedParamsCommand → "Load Params" button — binds shared parameters
// ─────────────────────────────────────────────────────────────────────────────

using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using ScanToBIM.Services;
using ScanToBIM.UI;

namespace ScanToBIM.Commands;

[Transaction(TransactionMode.Manual)]
public class StartAgentCommand : IExternalCommand
{
    public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
    {
        AgentPanelPage.ShowOrCreate(commandData.Application);
        return Result.Succeeded;
    }
}

[Transaction(TransactionMode.Manual)]
public class LoadSharedParamsCommand : IExternalCommand
{
    public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
    {
        var loader = new SharedParameterLoader();
        return loader.Execute(commandData, ref message, elements);
    }
}
