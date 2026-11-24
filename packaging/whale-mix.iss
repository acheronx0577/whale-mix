#ifndef AppVersion
#define AppVersion "0.2.0"
#endif

[Setup]
AppId={{5A4F4A5E-4B02-4A73-9F91-729BEB76FA09}
AppName=Whale Mix
AppVersion={#AppVersion}
AppPublisher=acheronx0577
DefaultDirName={localappdata}\Programs\WhaleMix
DisableProgramGroupPage=yes
DefaultGroupName=Whale Mix
PrivilegesRequired=lowest
RedirectionGuard=no
OutputDir=..\dist
OutputBaseFilename=WhaleMix-{#AppVersion}-windows-x64-setup
SetupIconFile=..\app\assets\icons\whale-mix.ico
UninstallDisplayIcon={app}\whale-mix.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\WhaleMix\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Whale Mix"; Filename: "{app}\whale-mix.exe"; WorkingDir: "{app}"; IconFilename: "{app}\whale-mix.exe"
Name: "{group}\{cm:UninstallProgram,Whale Mix}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Whale Mix"; Filename: "{app}\whale-mix.exe"; WorkingDir: "{app}"; Tasks: desktopicon; IconFilename: "{app}\whale-mix.exe"

[Registry]
Root: HKCU; Subkey: "Software\Classes\Applications\whale-mix.exe"; ValueType: string; ValueName: "FriendlyAppName"; ValueData: "Whale Mix"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\Applications\whale-mix.exe\SupportedTypes"; ValueType: string; ValueName: ".csv"; ValueData: ""; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\Applications\whale-mix.exe\shell\open"; ValueType: string; ValueName: "FriendlyAppName"; ValueData: "Whale Mix"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\Applications\whale-mix.exe\shell\open\command"; ValueType: string; ValueName: """{app}\whale-mix.exe"" ""%1"""; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\.csv\OpenWithProgids"; ValueType: string; ValueName: "WhaleMix.Csv"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\WhaleMix.Csv"; ValueType: string; ValueData: "Blue A Score CSV"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\WhaleMix.Csv\DefaultIcon"; ValueType: string; ValueName: "{app}\whale-mix.exe,0"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\WhaleMix.Csv\shell\open\command"; ValueType: string; ValueName: """{app}\whale-mix.exe"" ""%1"""; Flags: uninsdeletekey

[Run]
Filename: "{app}\whale-mix.exe"; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,Whale Mix}"; Flags: nowait postinstall skipifsilent runascurrentuser

[UninstallDelete]
Type: filesandordirs; Name: "{app}\*"
Type: dirifempty; Name: "{app}"

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  AppDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    AppDir := ExpandConstant('{app}');
    if (Length(AppDir) > 3) and DirExists(AppDir) then
    begin
      DelTree(AppDir, True, True, True);
    end;
  end;
end;
