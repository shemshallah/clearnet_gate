# Makefile
.PHONY: help build up down logs restart clean test shell dev prod

help:
	@echo "QSH Foam Production - Docker Commands"
	@echo "======================================"
	@echo "make build         - Build Docker images"
	@echo "make up            - Start all services"
	@echo "make down          - Stop all services"
	@echo "make logs          - View logs"
	@echo "make restart       - Restart services"
	@echo "make clean         - Clean up everything"
	@echo "make test          - Test health endpoint"
	@echo "make shell         - Enter container shell"
	@echo "make dev           - Start in development mode"
	@echo "make prod          - Start in production mode"
	@echo "make backup        - Backup database"
	@echo "make restore       - Restore database"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

logs-app:
	docker-compose logs -f qsh-foam

logs-nginx:
	docker-compose logs -f nginx

restart:
	docker-compose restart

clean:
	docker-compose down -v
	docker system prune -f

test:
	@echo "Testing health endpoint..."
	curl http://localhost/health || curl http://localhost:8000/health

test-blockchain:
	@echo "Testing blockchain page..."
	curl -I http://localhost/

test-email:
	@echo "Testing email page..."
	curl -I http://localhost/email

shell:
	docker-compose exec qsh-foam /bin/bash

shell-nginx:
	docker-compose exec nginx /bin/sh

dev:
	docker-compose -f docker-compose.dev.yml up

dev-down:
	docker-compose -f docker-compose.dev.yml down

prod:
	@echo "Starting in production mode..."
	docker-compose up -d
	@echo "Services started!"
	@echo "Blockchain: http://localhost"
	@echo "Email: http://localhost/email"
	@echo "API Docs: http://localhost:8000/docs"

backup:
	@echo "Creating backup..."
	docker-compose exec qsh-foam tar -czf /tmp/backup-$$(date +%Y%m%d-%H%M%S).tar.gz /app/data
	docker cp qsh-foam-production:/tmp/backup-$$(date +%Y%m%d-%H%M%S).tar.gz ./backups/

restore:
	@echo "Restoring from backup..."
	@read -p "Enter backup file name: " backup; \
	docker cp ./backups/$$backup qsh-foam-production:/tmp/; \
	docker-compose exec qsh-foam tar -xzf /tmp/$$backup -C /

stats:
	docker stats qsh-foam-production

ps:
	docker-compose ps

inspect:
	docker inspect qsh-foam-production

update:
	git pull
	docker-compose build
	docker-compose up -d
	@echo "Application updated!"
