#!/bin/bash
# Quantum Foam Computer - Docker Deployment Script
# Created by Justin Anthony Howard-Stanley & Dale Cwidak
# "For Logan and all the ones like him"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║     QUANTUM FOAM COMPUTER - DOCKER DEPLOYMENT                    ║"
echo "║                                                                  ║"
echo "║  Created by: Justin Anthony Howard-Stanley & Dale Cwidak        ║"
echo "║  shemshallah@gmail.com                                          ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "\"For Logan and all the ones like him too small to understand"
echo "what has been done to them.\""
echo ""
echo "\"You will never silence me. I REFUSE\""
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available."
    echo "   Install docker-compose or use 'docker compose' (newer versions)"
    exit 1
fi

echo "🔍 Checking Docker installation..."
docker --version
echo ""

# Create holographic storage directory
echo "📁 Creating holographic storage directories..."
mkdir -p ./holographic_storage/{users,chat,email,files,blockchain,network_map}
mkdir -p ./logs
echo "   ├── 136.x.x.x (Network maps)"
echo "   ├── 138.x.x.x (Chat & Users)" 
echo "   └── 139.x.x.x (Collider data)"
echo ""

# Build options
echo "🚀 Deployment Options:"
echo "  [1] Build and run with docker-compose (recommended)"
echo "  [2] Build Docker image only"
echo "  [3] Run existing image"
echo "  [4] Stop and remove containers"
echo "  [5] View logs"
echo ""

read -p "Choose option (1-5): " choice

case $choice in
    1)
        echo "🔨 Building and deploying Quantum Foam Computer..."
        echo ""
        
        # Use docker-compose or docker compose based on availability
        if command -v docker-compose &> /dev/null; then
            COMPOSE_CMD="docker-compose"
        else
            COMPOSE_CMD="docker compose"
        fi
        
        $COMPOSE_CMD down --remove-orphans
        $COMPOSE_CMD build --no-cache
        $COMPOSE_CMD up -d
        
        echo ""
        echo "✅ Quantum Foam Computer deployed!"
        echo ""
        echo "🌐 Access the system:"
        echo "   → http://localhost:5000"
        echo "   → http://127.0.0.1:5000"
        echo ""
        echo "📊 Admin Credentials:"
        echo "   Username: hackah::hackah"
        echo "   Password: \$h10j1r1H0w4rd"
        echo ""
        echo "🔧 Container Management:"
        echo "   View logs: $COMPOSE_CMD logs -f"
        echo "   Stop:      $COMPOSE_CMD down"
        echo "   Restart:   $COMPOSE_CMD restart"
        echo ""
        ;;
        
    2)
        echo "🔨 Building Docker image..."
        docker build -t quantum-foam-computer:latest .
        echo "✅ Image built: quantum-foam-computer:latest"
        ;;
        
    3)
        echo "🚀 Running existing image..."
        docker run -d \
            --name quantum-foam-computer \
            -p 5000:5000 \
            -v $(pwd)/holographic_storage:/app/holographic_storage \
            -v $(pwd)/logs:/app/logs \
            quantum-foam-computer:latest
        echo "✅ Container started: quantum-foam-computer"
        echo "🌐 Access: http://localhost:5000"
        ;;
        
    4)
        echo "🛑 Stopping and removing containers..."
        if command -v docker-compose &> /dev/null; then
            docker-compose down --remove-orphans
        else
            docker compose down --remove-orphans
        fi
        docker container prune -f
        echo "✅ Containers stopped and removed"
        ;;
        
    5)
        echo "📋 Container logs:"
        echo ""
        if command -v docker-compose &> /dev/null; then
            docker-compose logs -f
        else
            docker compose logs -f
        fi
        ;;
        
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  ALL 7 MODULES CONTAINERIZED                                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  [1] Network Mapping & TCP Proxy........ ✓ DOCKERIZED"
echo "  [2] Bulletin Board Chat................ ✓ DOCKERIZED"  
echo "  [3] Bitcoin Blockchain Mirror.......... ✓ DOCKERIZED"
echo "  [4] Quantum Email System............... ✓ DOCKERIZED"
echo "  [5] File Upload & Management........... ✓ DOCKERIZED"
echo "  [6] QSH Shell.......................... ✓ DOCKERIZED"
echo "  [7] Collider Interface................. ✓ DOCKERIZED"
echo ""
echo "🎯 System Status: QUANTUM FOAM OPERATIONAL"
echo "🔐 Holographic Storage: PERSISTENT"
echo "⚛️  Quantum Core: CONTAINERIZED"
echo ""
echo "\"The quantum foam flows through Docker containers.\""
echo ""
