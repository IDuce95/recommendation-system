

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE} $1 ${NC}"
    echo -e "${BLUE}================================${NC}"
}

load_environment() {
    if [ -f "$PROJECT_ROOT/.env" ]; then
        print_status "Loading environment from: .env"
        export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
    else
        print_error "Environment file .env not found!"
        exit 1
    fi
}

setup_directories() {
    print_status "Setting up required directories..."
    
    mkdir -p "$PROJECT_ROOT/logs/postgres"
    mkdir -p "$PROJECT_ROOT/logs/api"
    mkdir -p "$PROJECT_ROOT/logs/streamlit"
    mkdir -p "$PROJECT_ROOT/logs/mlflow"
    mkdir -p "$PROJECT_ROOT/logs/prometheus"
    
    mkdir -p "$PROJECT_ROOT/mlflow_artifacts"
    mkdir -p "$PROJECT_ROOT/embeddings"
    mkdir -p "$PROJECT_ROOT/models"
    
    if [ ! -d "/home/docker-volumes/postgres_data" ]; then
        print_status "Creating Docker volumes directory..."
        sudo mkdir -p /home/docker-volumes/postgres_data
        sudo mkdir -p /home/docker-volumes/prometheus_data
        sudo chmod 755 /home/docker-volumes/postgres_data
        sudo chmod 755 /home/docker-volumes/prometheus_data
    fi
}

get_docker_compose_cmd() {
    if command -v "docker compose" &> /dev/null; then
        echo "docker compose"
    elif command -v "docker-compose" &> /dev/null; then
        echo "docker-compose"
    else
        print_error "Neither 'docker compose' nor 'docker-compose' found!"
        exit 1
    fi
}

cmd_up() {
    print_header "Starting Services"
    load_environment
    setup_directories
    
    COMPOSE_CMD=$(get_docker_compose_cmd)
    cd "$PROJECT_ROOT"
    $COMPOSE_CMD up -d
    
    print_status "Services started successfully!"
    print_status "Access points:"
    print_status "  - API: http://localhost:5000"
    print_status "  - Streamlit: http://localhost:8501"
    print_status "  - MLflow: http://localhost:5555"
    print_status "  - Prometheus: http://localhost:9090"
    print_status "  - Grafana: http://localhost:3000 (admin/admin)"
    print_status "  - Database: localhost:5432"
}

cmd_down() {
    print_header "Stopping Services"
    load_environment
    
    COMPOSE_CMD=$(get_docker_compose_cmd)
    cd "$PROJECT_ROOT"
    $COMPOSE_CMD down
    
    print_status "Services stopped successfully!"
}

cmd_logs() {
    load_environment
    
    COMPOSE_CMD=$(get_docker_compose_cmd)
    cd "$PROJECT_ROOT"
    
    if [ -n "$2" ]; then
        $COMPOSE_CMD logs -f "$2"
    else
        $COMPOSE_CMD logs -f
    fi
}

cmd_build() {
    print_header "Building Services"
    load_environment
    setup_directories
    
    COMPOSE_CMD=$(get_docker_compose_cmd)
    cd "$PROJECT_ROOT"
    $COMPOSE_CMD build --no-cache
    
    print_status "Build completed successfully!"
}

cmd_restart() {
    print_header "Restarting Services"
    cmd_down
    cmd_up
}

cmd_status() {
    load_environment
    
    COMPOSE_CMD=$(get_docker_compose_cmd)
    cd "$PROJECT_ROOT"
    $COMPOSE_CMD ps
}

cmd_clean() {
    print_header "Cleaning Up"
    load_environment
    
    COMPOSE_CMD=$(get_docker_compose_cmd)
    cd "$PROJECT_ROOT"
    
    print_status "Stopping and removing containers..."
    $COMPOSE_CMD down -v --remove-orphans
    
    print_status "Removing unused images..."
    docker image prune -f
    
    print_status "Cleanup completed!"
}

show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  up         Start all services"
    echo "  down       Stop all services"
    echo "  restart    Restart all services"
    echo "  build      Build all Docker images"
    echo "  logs       Show logs (optionally specify service name)"
    echo "  status     Show service status"
    echo "  clean      Clean up containers and images"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 up"
    echo "  $0 logs api"
    echo "  $0 restart"
}

case "${1:-help}" in
    "up")
        cmd_up
        ;;
    "down")
        cmd_down
        ;;
    "restart")
        cmd_restart
        ;;
    "build")
        cmd_build
        ;;
    "logs")
        cmd_logs "$@"
        ;;
    "status")
        cmd_status
        ;;
    "clean")
        cmd_clean
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
