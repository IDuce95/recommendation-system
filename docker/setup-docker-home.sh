

echo "🐳 Configuring Docker to use /home partition for data storage..."

DOCKER_DATA_DIR="/home/docker-data"
if [ ! -d "$DOCKER_DATA_DIR" ]; then
    echo "Creating Docker data directory: $DOCKER_DATA_DIR"
    sudo mkdir -p "$DOCKER_DATA_DIR"
    sudo chown root:docker "$DOCKER_DATA_DIR"
    sudo chmod 755 "$DOCKER_DATA_DIR"
fi

DOCKER_CONFIG_DIR="/etc/docker"
if [ ! -d "$DOCKER_CONFIG_DIR" ]; then
    echo "Creating Docker config directory: $DOCKER_CONFIG_DIR"
    sudo mkdir -p "$DOCKER_CONFIG_DIR"
fi

echo "Copying Docker daemon configuration..."
sudo cp docker/daemon.json /etc/docker/daemon.json

echo "Stopping Docker service..."
sudo systemctl stop docker

echo "Starting Docker service with new configuration..."
sudo systemctl start docker

if sudo systemctl is-active --quiet docker; then
    echo "✅ Docker service is running successfully!"
    echo "🏠 Docker data will now be stored in: $DOCKER_DATA_DIR"
    echo "📊 Current Docker info:"
    sudo docker info | grep "Docker Root Dir"
else
    echo "❌ Failed to start Docker service. Please check the configuration."
    exit 1
fi

echo "🎉 Docker configuration completed successfully!"
