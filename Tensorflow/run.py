import os

# Build the Docker image
os.system('docker build -t emot .')

# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Run the Docker container with the correct command
os.system(f'docker run --rm -it --privileged -v {current_dir}:/app --device=/dev/video0 emot python display.py')
