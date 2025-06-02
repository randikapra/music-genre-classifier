#!/bin/bash

echo "Starting Music Genre Classification Application..."

# Store the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Script located at: $SCRIPT_DIR"

# Navigate to script directory first
cd "$SCRIPT_DIR"

# Check and install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Checking for required dependencies..."
    if ! pip list | grep -q "pyspark"; then
        echo "Installing dependencies from requirements.txt..."
        pip install -r requirements.txt
    else
        echo "Dependencies already installed."
    fi
else
    echo "Warning: requirements.txt not found. Dependencies may be missing."
    # Create a basic requirements file if it doesn't exist
    echo "Creating a basic requirements.txt file..."
    cat > requirements.txt << EOF
flask==2.0.1
pyspark==3.1.2
numpy==1.21.0
pandas==1.3.0
EOF
    echo "Installing basic dependencies..."
    pip install -r requirements.txt
fi

# Set environment variables if needed
export PYSPARK_PYTHON=python
export PYSPARK_DRIVER_PYTHON=python

# Check if app.py exists in current directory
if [ -f "app.py" ]; then
    echo "Found app.py in current directory"
else
    echo "app.py not found in current directory, checking subdirectories..."
    
    # Try to find app.py
    APP_PATH=$(find "$SCRIPT_DIR" -name "app.py" -type f | head -n 1)
    
    if [ -z "$APP_PATH" ]; then
        echo "Error: Could not find app.py. Make sure it exists in this directory or subdirectories."
        exit 1
    else
        echo "Found app.py at: $APP_PATH"
        cd "$(dirname "$APP_PATH")"
        echo "Changed directory to: $(pwd)"
    fi
fi

# Verify Spark is available
if ! command -v spark-shell &> /dev/null; then
    echo "Warning: spark-shell not found in PATH. Make sure Spark is properly installed."
fi

# Run the Flask application
echo "Starting Flask application..."
python app.py &
APP_PID=$!

# Wait a moment for the app to start
sleep 3

# Open the browser (platform-dependent)
echo "Opening browser..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open http://localhost:5000 || (echo "Failed to open browser with xdg-open, please manually navigate to http://localhost:5000")
elif [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:5000 || (echo "Failed to open browser with open, please manually navigate to http://localhost:5000")
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    start http://localhost:5000 || (echo "Failed to open browser, please manually navigate to http://localhost:5000")
else
    echo "Please open a browser and navigate to: http://localhost:5000"
fi

echo "Application started successfully!"
echo "Press Ctrl+C to stop the application"

# Wait for the app to be stopped
wait $APP_PID


# #!/bin/bash

# echo "Starting Music Genre Classification Application..."

# # Set environment variables if needed
# export PYSPARK_PYTHON=python
# export PYSPARK_DRIVER_PYTHON=python

# # Navigate to project directory
# cd music-genre-classifier1

# # Activate virtual environment if you have one
# # source venv/bin/activate

# # Run the Flask application
# echo "Starting Flask application..."
# python app.py &
# APP_PID=$!

# # Wait a moment for the app to start
# sleep 3

# # Open the browser (platform-dependent)
# echo "Opening browser..."
# if [[ "$OSTYPE" == "linux-gnu"* ]]; then
#     xdg-open http://localhost:5000
# elif [[ "$OSTYPE" == "darwin"* ]]; then
#     open http://localhost:5000
# fi

# echo "Application started successfully!"
# echo "Press Ctrl+C to stop the application"

# # Wait for the app to be stopped
# wait $APP_PID