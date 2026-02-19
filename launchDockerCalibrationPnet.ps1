# docker run -it -p 8888:8888 -p 3000:3000 --mount type=bind,src="D:\OneDrive - UQAM\1 - Projets\Post-Doc - PnET Calibration\Calibration_PnET_DIVERSE",dst=/calibrationFolder landis_ii_v8_calibration_pnet

# Get the directory where the script is located
$calibrationPath = Split-Path -Parent $MyInvocation.MyCommand.Path

# Ensure the path exists
if (-Not (Test-Path $calibrationPath)) {
    Write-Error "Calibration folder not found at $calibrationPath"
    exit 1
}

# Run the Docker container with the resolved absolute path
docker run -it --rm -p 8888:8888 -p 3000:3000 --mount type=bind,src="$calibrationPath",dst=/calibrationFolder landis_ii_v8_calibration_pnet

pause