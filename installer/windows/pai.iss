; Inno Setup script for Pai on Windows.
;
; Build:
;   ISCC.exe /DAppVersion=1.0.3 installer\windows\pai.iss
; Override the source tree for a staged copy:
;   ISCC.exe /DAppSrc=C:\path\to\pai /DAppVersion=1.0.3 installer\windows\pai.iss
;
; Pai installs per-user under %LOCALAPPDATA%\Programs\Pai because the first-run
; bootstrap (setup_env.py) creates its private .venv inside the install
; directory. A stable AppId lets a newer installer upgrade an existing install
; in place. The .venv is not tracked by the installer, so it survives an
; upgrade and the wheels are not downloaded again; only a changed requirements
; file triggers a reinstall (see setup_env.py's hash marker). There is
; deliberately no [UninstallDelete] wildcard, so an upgrade cannot wipe .venv.

#define AppName "Pai"
#ifndef AppVersion
  #define AppVersion "1.0.3"
#endif
#ifndef AppSrc
  #define AppSrc "..\.."
#endif
#define AppIcon "..\..\pai_icon.ico"

[Setup]
AppId={{7E2F4C1A-9B3D-4A6E-8C21-5F0D9A7B3E64}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Pai
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\..\dist
OutputBaseFilename=pai-{#AppVersion}-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UsePreviousAppDir=yes

#if FileExists(AppIcon)
SetupIconFile={#AppIcon}
UninstallDisplayIcon={app}\pai_icon.ico
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "{#AppSrc}\*"; DestDir: "{app}"; Excludes: ".git,.git\*,*\.git\*,.venv,.venv\*,*\.venv\*,__pycache__,__pycache__\*,*\__pycache__\*,*.pyc,*.pyo,*.zip,dist,dist\*,*\.pytest_cache\*,*\.mypy_cache\*"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
#if FileExists(AppIcon)
Name: "{group}\{#AppName}"; Filename: "{app}\run.bat"; WorkingDir: "{app}"; IconFilename: "{app}\pai_icon.ico"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\run.bat"; WorkingDir: "{app}"; IconFilename: "{app}\pai_icon.ico"; Tasks: desktopicon
#else
Name: "{group}\{#AppName}"; Filename: "{app}\run.bat"; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\run.bat"; WorkingDir: "{app}"; Tasks: desktopicon
#endif

[Run]
Filename: "{app}\run.bat"; Parameters: "--setup-only"; Description: "Download Pai's dependencies now (needs internet)"; Flags: postinstall skipifsilent
Filename: "{app}\run.bat"; Description: "Launch Pai"; Flags: postinstall nowait skipifsilent

[Code]
function HasPython(): Boolean;
var
  Code: Integer;
begin
  Result :=
    (Exec('py', '-3.12 -c "import sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 15) else 1)"', '', SW_HIDE, ewWaitUntilTerminated, Code) and (Code = 0))
    or
    (Exec('python', '-c "import sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 15) else 1)"', '', SW_HIDE, ewWaitUntilTerminated, Code) and (Code = 0));
end;

function InitializeSetup(): Boolean;
begin
  Result := True;
  if not HasPython() then
  begin
    if MsgBox('Pai needs Python 3.12-3.14, but a supported version was not found on PATH.' + #13#10 + #13#10 +
              'Pai can download and install it for you on first launch. You can also install it now from https://www.python.org/downloads/ (or run "winget install Python.Python.3.12").' + #13#10 + #13#10 +
              'Continue the installation anyway?', mbConfirmation, MB_YESNO) = IDNO then
      Result := False;
  end;
end;
