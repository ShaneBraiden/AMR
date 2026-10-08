; Inno Setup 6 script for the Antibiotic Resistance Forecast installers. build_app.ps1 compiles it after
; PyInstaller has built dist\<app name>\:
;   ISCC.exe /DAppVersion=1.0.0 packaging\installer.iss
; and for the Studio, with its own name, ID (so both can be installed side by side), icon and file name:
;   ISCC.exe /DAppVersion=1.0.0 "/DAppName=Antibiotic Resistance Forecast Studio" /DAppGuid=... /DIconFile=entry.ico
;            /DOutputName=AntibioticResistanceForecastStudio packaging\installer.iss
;
; Installs for the current user by default (no administrator rights needed); the first page offers an
; install for all users instead.

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#ifndef AppName
  #define AppName "Antibiotic Resistance Forecast"
#endif
#ifndef AppGuid
  #define AppGuid "7C1F3F2E-5B7A-4C1E-9A43-2E6B1F0D9A55"
#endif
#ifndef IconFile
  #define IconFile "app.ico"
#endif
#ifndef OutputName
  #define OutputName "AntibioticResistanceForecast"
#endif
#define AppExe AppName + ".exe"
#define WebView2Key "Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"

[Setup]
AppId={{{#AppGuid}}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\dist
OutputBaseFilename={#OutputName}-{#AppVersion}-Setup
SetupIconFile=..\assets\{#IconFile}
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
; The app is mostly large DLLs that compress well together.
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\{#AppName}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; The app's data folder holds only copies of loaded workbooks, trained-model caches and logs.
Type: filesandordirs; Name: "{localappdata}\{#AppName}"

[Code]
function HasWebView2(Root: Integer; Key: String): Boolean;
var
  Version: String;
begin
  Result := RegQueryStringValue(Root, Key, 'pv', Version) and (Version <> '') and (Version <> '0.0.0.0');
end;

function InitializeSetup: Boolean;
var
  ErrorCode: Integer;
begin
  Result := True;
  if not (HasWebView2(HKLM, 'SOFTWARE\WOW6432Node\{#WebView2Key}') or HasWebView2(HKLM, 'SOFTWARE\{#WebView2Key}')
          or HasWebView2(HKCU, 'Software\{#WebView2Key}')) then
    if SuppressibleMsgBox('{#AppName} needs the Microsoft Edge WebView2 Runtime, which is not installed on this PC.' + #13#10#13#10 +
                          'Open the Microsoft download page now? Install it before you start the app.',
                          mbConfirmation, MB_YESNO, IDNO) = IDYES then
      ShellExec('open', 'https://go.microsoft.com/fwlink/p/?LinkId=2124703', '', '', SW_SHOWNORMAL, ewNoWait, ErrorCode);
end;
