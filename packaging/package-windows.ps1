param(
    [string]$BuildDir = "build/windows-msvc",
    [string]$DistDir = "dist/WhaleMix",
    [string]$Version = "0.2.0"
)

$ErrorActionPreference = "Stop"

Write-Host "Installing to $DistDir..."
cmake --install $BuildDir --prefix $DistDir

# Find windeployqt
$windeploy = (Get-Command windeployqt, windeployqt6, windeployqt-qt6 -ErrorAction SilentlyContinue | Select-Object -First 1).Source
if ($windeploy) {
    Write-Host "Running $windeploy..."
    & $windeploy --release --no-translations "$DistDir/whale-mix.exe"
} else {
    Write-Warning "windeployqt not found!"
}

# Clean any test framework clutter if installed
if (Test-Path "$DistDir/include") { Remove-Item -Recurse -Force "$DistDir/include" }
if (Test-Path "$DistDir/lib") { Remove-Item -Recurse -Force "$DistDir/lib" }

# Copy GCC runtimes if compiled with MinGW / GCC
$gccBin = "C:\msys64\ucrt64\bin"
if (Test-Path "$gccBin\libgcc_s_seh-1.dll") {
    Copy-Item "$gccBin\libgcc_s_seh-1.dll" "$DistDir\" -Force
    Copy-Item "$gccBin\libstdc++-6.dll" "$DistDir\" -Force
    Copy-Item "$gccBin\libwinpthread-1.dll" "$DistDir\" -Force
}

# Ensure samples exist in dist
if (-not (Test-Path "$DistDir/samples")) {
    Copy-Item -Recurse -Force "samples" "$DistDir/samples"
}

# Create zip
$ZipPath = "dist/WhaleMix-$Version-windows-x64.zip"
if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
Write-Host "Creating $ZipPath..."
Compress-Archive -Path "$DistDir" -DestinationPath $ZipPath

Write-Host "Successfully created $ZipPath"
