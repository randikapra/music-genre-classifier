@echo off
echo Starting Music Genre Classification Application...

:: Store the script's directory
set "SCRIPT_DIR=%~dp0"
echo Script located at: %SCRIPT_DIR%

:: Navigate to script directory first
cd /d "%SCRIPT_DIR%"

:: Check and install dependencies from requirements.txt
if exist requirements.txt (
    echo Checking for required dependencies...
    pip list | findstr "pyspark" > nul
    if errorlevel 1 (
        echo Installing dependencies from requirements.txt...
        pip install -r requirements.txt
    ) else (
        echo Dependencies already installed.
    )
) else (
    echo Warning: requirements.txt not found. Dependencies may be missing.
    :: Create a basic requirements file if it doesn't exist
    echo Creating a basic requirements.txt file...
    (
        echo flask==2.0.1
        echo pyspark==3.1.2
        echo numpy==1.21.0
        echo pandas==1.3.0
    ) > requirements.txt
    echo Installing basic dependencies...
    pip install -r requirements.txt
)

:: Set environment variables if needed
set PYSPARK_PYTHON=python
set PYSPARK_DRIVER_PYTHON=python

:: Check if app.py exists in current directory
if exist app.py (
    echo Found app.py in current directory
) else (
    echo app.py not found in current directory, checking subdirectories...
    
    :: Try to find app.py (Windows version of find)
    for /f "delims=" %%i in ('dir /s /b "%SCRIPT_DIR%app.py" 2^>nul') do (
        set "APP_PATH=%%i"
        goto :found_app
    )
    
    echo Error: Could not find app.py. Make sure it exists in this directory or subdirectories.
    goto :eof
    
    :found_app
    echo Found app.py at: %APP_PATH%
    for %%i in ("%APP_PATH%") do cd /d "%%~dpi"
    echo Changed directory to: %CD%
)

:: Verify Spark is available
where spark-shell > nul 2>&1
if errorlevel 1 (
    echo Warning: spark-shell not found in PATH. Make sure Spark is properly installed.
)

:: Run the Flask application
echo Starting Flask application...
start /B python app.py

:: Wait a moment for the app to start
timeout /t 3 > nul

:: Open the browser
echo Opening browser...
start http://localhost:5000 || echo Failed to open browser, please manually navigate to http://localhost:5000

echo Application started successfully!
echo The application is running in the background.
echo Close this window or press Ctrl+C to exit (note: the Flask server will continue running).
pause

:: @echo off
:: echo Starting Music Genre Classification Application...

:: Set environment variables if needed
:: set PYSPARK_PYTHON=python
:: set PYSPARK_DRIVER_PYTHON=python

:: Navigate to project directory
:: cd music-genre-classifier1

:: Activate virtual environment if you have one
:: call venv\Scripts\activate

:: Run the Flask application
:: echo Starting Flask application...
:: start "" python app.py

:: Wait a moment for the app to start
:: timeout /t 3

:: Open the browser
:: echo Opening browser...
:: start http://localhost:5000

:: echo Application started successfully!
