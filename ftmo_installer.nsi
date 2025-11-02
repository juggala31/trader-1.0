; FTMO AI Trading System - Real Windows Installer
Name "FTMO AI Trading System"
OutFile "FTMO_AI_Trading_System_Setup.exe"
InstallDir "$PROGRAMFILES\FTMO AI Trading System"
RequestExecutionLevel admin
ShowInstDetails show

!include MUI2.nsh

!define MUI_ICON "ftmo_icon.ico"
!define MUI_UNICON "ftmo_icon.ico"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE.txt"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Section "Main Application"
    SetOutPath "$INSTDIR"
    
    ; Copy all Python files
    File /r "*.py"
    File "requirements.txt"
    
    ; Create start menu shortcut
    CreateDirectory "$SMPROGRAMS\FTMO AI Trading System"
    CreateShortcut "$SMPROGRAMS\FTMO AI Trading System\FTMO AI Trading System.lnk" "$INSTDIR\start_enhanced_dashboard.bat" "" "$INSTDIR\ftmo_icon.ico"
    CreateShortcut "$SMPROGRAMS\FTMO AI Trading System\Uninstall.lnk" "$INSTDIR\uninstall.exe"
    
    ; Create desktop shortcut
    CreateShortcut "$DESKTOP\FTMO AI Trading System.lnk" "$INSTDIR\start_enhanced_dashboard.bat" "" "$INSTDIR\ftmo_icon.ico"
    
    ; Write uninstaller
    WriteUninstaller "$INSTDIR\uninstall.exe"
    
    ; Write registry keys for uninstall
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FTMO_AI_Trading_System" "DisplayName" "FTMO AI Trading System"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FTMO_AI_Trading_System" "UninstallString" "$\"$INSTDIR\uninstall.exe$\""
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FTMO_AI_Trading_System" "DisplayIcon" "$\"$INSTDIR\ftmo_icon.ico$\""
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FTMO_AI_Trading_System" "Publisher" "FTMO Trading Technologies"
SectionEnd

Section "Post-Install"
    ; Run dependency installer silently
    ExecWait '"$INSTDIR\install_dependencies.bat"'
SectionEnd

Section "Uninstall"
    ; Remove shortcuts
    Delete "$DESKTOP\FTMO AI Trading System.lnk"
    Delete "$SMPROGRAMS\FTMO AI Trading System\*.*"
    RMDir "$SMPROGRAMS\FTMO AI Trading System"
    
    ; Remove installation directory
    RMDir /r "$INSTDIR"
    
    ; Remove registry keys
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FTMO_AI_Trading_System"
SectionEnd
