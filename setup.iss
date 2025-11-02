[Setup]
AppName=FTMO AI Trading System
AppVersion=1.0
AppPublisher=FTMO Trader
DefaultDirName={pf}\FTMO AI Trading System
DefaultGroupName=FTMO AI Trading System
OutputDir=Output
OutputBaseFilename=FTMO_AI_Trading_System_Setup
SetupIconFile=ftmo_icon.ico
Compression=lzma
SolidCompression=yes

[Files]
Source: "bin\*"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "config\*"; DestDir: "{app}\config"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\FTMO AI Trading System"; Filename: "{app}\bin\start_enhanced_dashboard.bat"; IconFilename: "{app}\ftmo_icon.ico"
Name: "{autodesktop}\FTMO AI Trading System"; Filename: "{app}\bin\start_enhanced_dashboard.bat"; IconFilename: "{app}\ftmo_icon.ico"

[Run]
Filename: "{app}\bin\start_enhanced_dashboard.bat"; Description: "Launch FTMO AI Trading System"; Flags: nowait postinstall skipifsilent
