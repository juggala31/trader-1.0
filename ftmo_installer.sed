[Version]
Class=IEXPRESS
SEDVersion=3
[Options]
PackagePurpose=InstallApp
ShowInstallProgramWindow=1
HideExtractAnimation=1
UseLongFileName=1
InsideCompressed=0
CAB_FixedSize=0
CAB_ResvCodeSigning=0
RebootMode=I
InstallPrompt=%InstallPrompt%
DisplayLicense=%DisplayLicense%
FinishMessage=%FinishMessage%
TargetName=%TargetName%
FriendlyName=%FriendlyName%
AppLaunched=%AppLaunched%
PostInstallCmd=%PostInstallCmd%
AdminQuietInstalled=%AdminQuietInstalled%
UserQuietInstalled=%UserQuietInstalled%
SourceFiles=SourceFiles

[SourceFiles]
SourceFiles0=C:\exosatitraderpy\FTMO_Real_Installer\

[SourceFiles0]
%FILE0%=setup.bat

[Strings]
InstallPrompt=Do you want to install FTMO AI Trading System?
DisplayLicense=LICENSE.txt
FinishMessage=FTMO AI Trading System has been installed successfully!
TargetName=C:\exosatitraderpy\FTMO_AI_Trading_System_Setup.exe
FriendlyName=FTMO AI Trading System
AppLaunched=setup.bat
PostInstallCmd=<None>
AdminQuietInstalled=
UserQuietInstalled=
FILE0="setup.bat"
