param(
    [string]$Image = "sumo-rl:latest",
    [string]$Display = "host.docker.internal:0.0",
    [string]$Scenario = "sumo_rl/nets/3x3grid/3x3.sumocfg",
    [ValidateSet("", "all", "fixed-time", "max-pressure", "sotl", "dqn-ar", "spre-plus", "ppo")]
    [string]$Agent = "",
    [string]$ControlledTls = "",
    [string]$PpoModel = "",
    [string]$DqnModel = "",
    [int]$NumEpisodes = 1,
    [int]$SimEndTime = 3600,
    [string]$RunName = "",
    [switch]$Shell,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($repoRoot)) {
    $repoRoot = (Get-Location).Path
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker was not found on PATH. Start Docker Desktop, then try again."
}

$imageExists = docker images $Image --format "{{.Repository}}:{{.Tag}}"
if (-not $imageExists) {
    throw "Docker image '$Image' was not found. Build it first with: docker build -t sumo-rl ."
}

$vcxsrv = Get-Process vcxsrv -ErrorAction SilentlyContinue
if (-not $vcxsrv) {
    Write-Warning "VcXsrv does not appear to be running. Start XLaunch first: Multiple windows, then Disable access control."
}

$containerName = "sumo-rl-gui-$([Guid]::NewGuid().ToString('N').Substring(0, 8))"
$repoMount = "type=bind,source=$repoRoot,target=/workspace"

Write-Host "[INFO] Repo: $repoRoot"
Write-Host "[INFO] Image: $Image"
Write-Host "[INFO] DISPLAY: $Display"
Write-Host "[INFO] Container: $containerName"

$dockerArgs = @(
    "run",
    "--rm",
    "-it",
    "--name", $containerName,
    "--mount", $repoMount,
    "-w", "/workspace",
    "-e", "DISPLAY=$Display",
    "-e", "QT_X11_NO_MITSHM=1",
    $Image
)

if ($Shell) {
    $dockerArgs += "bash"
} elseif (-not [string]::IsNullOrWhiteSpace($Agent)) {
    $dockerArgs += @(
        "python3", "-m", "sumo_rl.experiments.eval_baselines",
        "--use-gui",
        "--agent", $Agent,
        "--sumo-cfg-file", $Scenario,
        "--num-episodes", "$NumEpisodes",
        "--sim-end-time", "$SimEndTime"
    )

    if (-not [string]::IsNullOrWhiteSpace($ControlledTls)) {
        $dockerArgs += @("--controlled-tls", $ControlledTls)
    }
    if (-not [string]::IsNullOrWhiteSpace($PpoModel)) {
        $dockerArgs += @("--ppo-model", $PpoModel)
    }
    if (-not [string]::IsNullOrWhiteSpace($DqnModel)) {
        $dockerArgs += @("--dqn-model", $DqnModel)
    }
    if (-not [string]::IsNullOrWhiteSpace($RunName)) {
        $dockerArgs += @("--run-name", $RunName)
    }
} else {
    $dockerArgs += @("sumo-gui", "-c", $Scenario)
}

if ($DryRun) {
    Write-Host "[DRY RUN] docker $($dockerArgs -join ' ')"
} else {
    docker @dockerArgs
}
