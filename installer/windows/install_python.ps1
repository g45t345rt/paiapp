<#
.SYNOPSIS
    Download and install a supported Python (3.12-3.14) for Pai on Windows.

.DESCRIPTION
    Called by run.bat when no supported Python is on PATH and the user agreed to
    install one. Tries the Windows package manager (winget) first, then falls
    back to downloading the official python.org installer and running it in
    per-user mode (no admin prompt, no system-wide change). On success the path
    to a working python.exe is written to -OutFile (one line) and the exit code
    is 0; on failure the exit code is non-zero and no file is written.

    Progress is written to the host, so it does not interfere with -OutFile.

.PARAMETER OutFile
    Path to a file that receives the resolved python.exe path on success.
#>
[CmdletBinding()]
param(
    [string]$OutFile = ""
)

# The interpreter range Pai supports; keep in sync with setup_env.py
# (MIN_PYTHON/MAX_PYTHON) and the launchers.
$MinVersion = [version]"3.12"
$MaxVersion = [version]"3.15"
$FtpBase = "https://www.python.org/ftp/python/"
$WingetId = "Python.Python.3.12"

$ErrorActionPreference = "Stop"

function Write-Step($Message) {
    Write-Host "==> $Message"
}

function Test-SupportedPython([string]$Path) {
    if (-not $Path) { return $false }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    try {
        & $Path -c "import sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 15) else 1)" 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Find-SupportedPython {
    $candidates = New-Object System.Collections.Generic.List[string]

    # python.org and winget install per-user under %LOCALAPPDATA%\Programs\Python.
    $local = Join-Path $env:LOCALAPPDATA "Programs\Python"
    if (Test-Path -LiteralPath $local) {
        Get-ChildItem -LiteralPath $local -Directory -Filter "Python3*" -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            ForEach-Object {
                $exe = Join-Path $_.FullName "python.exe"
                if (Test-Path -LiteralPath $exe) { $candidates.Add($exe) }
            }
    }

    # The py launcher knows about installs that are not under %LOCALAPPDATA%.
    foreach ($tag in @("3.12", "3.13", "3.14", "3")) {
        try {
            $exe = (& py "-$tag" -c "import sys; print(sys.executable)" 2>$null)
            if ($exe) { $candidates.Add(($exe | Select-Object -First 1).Trim()) }
        } catch {
            # py has no such tag; ignore.
        }
    }

    foreach ($candidate in $candidates) {
        if (Test-SupportedPython $candidate) { return $candidate }
    }
    return $null
}

function Test-Url([string]$Url) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Method Head -Uri $Url `
            -TimeoutSec 30
        return ($response.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function Get-InstallerArchSuffix {
    if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { return "arm64" }
    return "amd64"
}

function Get-PythonOrgInstallerUrl {
    $suffix = Get-InstallerArchSuffix
    Write-Step "Looking up the newest Python 3.12-3.14 on python.org"
    $listing = (Invoke-WebRequest -UseBasicParsing -Uri $FtpBase -TimeoutSec 60).Content
    $versions = [regex]::Matches($listing, 'href="(\d+\.\d+\.\d+)/"') |
        ForEach-Object { [version]$_.Groups[1].Value } |
        Where-Object { $_ -ge $MinVersion -and $_ -lt $MaxVersion } |
        Sort-Object -Descending -Unique

    foreach ($version in $versions) {
        $url = "{0}{1}/python-{1}-{2}.exe" -f $FtpBase, $version, $suffix
        if (Test-Url $url) { return $url }
    }
    return $null
}

function Install-WithWinget {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        return $false
    }
    Write-Step "Trying winget ($WingetId)"
    try {
        & winget install --id $WingetId --exact --silent `
            --accept-package-agreements --accept-source-agreements
    } catch {
        Write-Host "winget failed: $($_.Exception.Message)"
        return $false
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "winget exited with code $LASTEXITCODE"
        return $false
    }
    return [bool](Find-SupportedPython)
}

function Install-WithPythonOrg {
    $url = Get-PythonOrgInstallerUrl
    if (-not $url) {
        Write-Host "Could not find a downloadable Python 3.12-3.14 installer."
        return $false
    }
    Write-Step "Downloading $url"
    $exe = Join-Path $env:TEMP ("pai-python-" + [guid]::NewGuid().ToString("N") + ".exe")
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $exe -TimeoutSec 600
        Write-Step "Installing Python for the current user"
        $installArgs = "/quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 " +
                "Include_pip=1 Include_test=0"
        $proc = Start-Process -FilePath $exe -ArgumentList $installArgs -Wait -PassThru
        if ($proc.ExitCode -ne 0) {
            Write-Host "The Python installer exited with code $($proc.ExitCode)."
            return $false
        }
    } finally {
        Remove-Item -LiteralPath $exe -Force -ErrorAction SilentlyContinue
    }
    return [bool](Find-SupportedPython)
}

try {
    $python = Find-SupportedPython
    if ($python) {
        Write-Step "Python is already available at $python"
    } elseif (Install-WithWinget) {
        $python = Find-SupportedPython
        Write-Step "winget installed Python at $python"
    } elseif (Install-WithPythonOrg) {
        $python = Find-SupportedPython
        Write-Step "Installed Python at $python"
    }

    if (-not $python) {
        Write-Error "Could not install Python 3.12-3.14 automatically. Install it from https://www.python.org/downloads/ and run Pai again."
        exit 1
    }

    if ($OutFile) {
        [System.IO.File]::WriteAllText($OutFile, $python)
    } else {
        Write-Output $python
    }
    exit 0
} catch {
    Write-Error "Python installation failed: $($_.Exception.Message)"
    exit 1
}
