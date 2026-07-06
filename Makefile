# DivinityAI — Makefile
# =============================================================================
#
# Usage:
#   make up                    # Start in development mode (default)
#   DEVELOPMENT_MODE=false make up  # Start in production mode
#   make down                  # Stop all services
#   make help                  # Show all available commands
#
# =============================================================================

# ── Configuration ─────────────────────────────────────────────────────────────

DEVELOPMENT_MODE ?= true
DOCKER_COMPOSE := docker compose
COMPOSE_FILE_DEV := docker-compose.yml
COMPOSE_FILE_PROD := docker-compose.yml:docker-compose.prod.yml

ifeq ($(DEVELOPMENT_MODE),true)
	COMPOSE_FILES := $(subst :, -f ,$(COMPOSE_FILE_DEV))
	MODE_LABEL := 🔧 development
else
	COMPOSE_FILES := $(subst :, -f ,$(COMPOSE_FILE_PROD))
	MODE_LABEL := 🚀 production
endif

COMPOSE := $(DOCKER_COMPOSE) $(COMPOSE_FILES)
ENV_FILE := .env
ENV_EXAMPLE := .env.example

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help
	@echo "DivinityAI — Makefile"
	@echo ""
	@echo "Current mode: DEVELOPMENT_MODE=$(DEVELOPMENT_MODE)  $(MODE_LABEL)"
	@echo ""
	@echo "Usage:"
	@echo "  make [target] [DEVELOPMENT_MODE=true|false]"
	@echo ""
	@echo "Core:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────

$(ENV_FILE):
	@echo "📋 Creating $(ENV_FILE) from $(ENV_EXAMPLE)..."
	@cp $(ENV_EXAMPLE) $(ENV_FILE)
	@echo "✅ $(ENV_FILE) created — edit it with your keys."

.PHONY: setup-env
setup-env: $(ENV_FILE) ## Create .env from .env.example if missing

# ── Build ─────────────────────────────────────────────────────────────────────

.PHONY: build
build: setup-env ## Build Docker images
	@echo "🔨 Building images — $(MODE_LABEL)"
	$(COMPOSE) build

.PHONY: build-no-cache
build-no-cache: setup-env ## Build images without cache
	@echo "🔨 Building images from scratch — $(MODE_LABEL)"
	$(COMPOSE) build --no-cache

# ── Start / Stop ──────────────────────────────────────────────────────────────

.PHONY: up
up: setup-env ## Start all services (detached)
	@echo "🆙 Starting services — $(MODE_LABEL)"
	$(COMPOSE) up -d
	@echo "✅ Done. Run 'make logs' to tail logs, 'make ps' to see status."

.PHONY: down
down: ## Stop all services and remove containers
	@echo "🛑 Stopping all services..."
	$(COMPOSE) down

.PHONY: down-volumes
down-volumes: ## Stop all services and remove volumes (destructive)
	@echo "⚠️  Stopping services and removing volumes..."
	$(COMPOSE) down -v

.PHONY: restart
restart: ## Restart all services
	@echo "🔄 Restarting services — $(MODE_LABEL)"
	$(COMPOSE) restart

.PHONY: ps
ps: ## Show running containers
	$(COMPOSE) ps

# ── Logs ──────────────────────────────────────────────────────────────────────

.PHONY: logs
logs: ## Tail logs from all services
	$(COMPOSE) logs -f

.PHONY: logs-backend
logs-backend: ## Tail backend logs
	$(COMPOSE) logs -f backend

.PHONY: logs-frontend
logs-frontend: ## Tail frontend logs
	$(COMPOSE) logs -f frontend

.PHONY: logs-nginx
logs-nginx: ## Tail nginx logs
	$(COMPOSE) logs -f nginx

# ── Shell access ──────────────────────────────────────────────────────────────

.PHONY: shell-backend
shell-backend: ## Open a shell in the backend container
	$(COMPOSE) exec backend bash

.PHONY: shell-frontend
shell-frontend: ## Open a shell in the frontend container
	$(COMPOSE) exec frontend sh

# ── Django shortcuts ──────────────────────────────────────────────────────────

.PHONY: migrate
migrate: ## Run Django migrations
	$(COMPOSE) exec backend python manage.py migrate

.PHONY: makemigrations
makemigrations: ## Create new Django migrations
	$(COMPOSE) exec backend python manage.py makemigrations

.PHONY: createsuperuser
createsuperuser: ## Create a Django superuser
	$(COMPOSE) exec backend python manage.py createsuperuser

.PHONY: collectstatic
collectstatic: ## Run Django collectstatic
	$(COMPOSE) exec backend python manage.py collectstatic --noinput

.PHONY: shell-django
shell-django: ## Open Django shell
	$(COMPOSE) exec backend python manage.py shell

.PHONY: test
test: ## Run all tests (backend + frontend)
	@echo "🧪 Running backend tests..."
	$(COMPOSE) exec -T backend python manage.py test --verbosity=2 chroma corpus generation qa.tests qa.integration_tests retrieval router
	@echo ""
	@echo "🧪 Running frontend tests..."
	$(COMPOSE) exec -T frontend sh -c "npm ci && npm run test"

.PHONY: test-backend
test-backend: ## Run Django tests only
	$(COMPOSE) exec backend python manage.py test --verbosity=2

.PHONY: test-backend-unit
test-backend-unit: ## Run Django unit tests only
	$(COMPOSE) exec backend python manage.py test --verbosity=2 chroma corpus generation qa.tests retrieval router

.PHONY: test-backend-integration
test-backend-integration: ## Run Django integration tests only
	$(COMPOSE) exec backend python manage.py test --verbosity=2 qa.integration_tests

.PHONY: test-frontend
test-frontend: ## Run frontend tests only
	$(COMPOSE) exec frontend sh -c "npm ci && npm run test"

# ── Frontend ──────────────────────────────────────────────────────────────────

.PHONY: build-frontend
build-frontend: ## Build frontend for production (runs vite build locally)
	cd frontend && npm ci && npm run build

.PHONY: lint-frontend
lint-frontend: ## Lint frontend code
	$(COMPOSE) exec frontend npm run lint

# ── Utilities ─────────────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Remove containers, images, volumes, and __pycache__
	@echo "🧹 Cleaning up..."
	$(COMPOSE) down -v --rmi all 2>/dev/null || true
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	@echo "✅ Clean."

.PHONY: prune
prune: ## Remove unused Docker data (system-wide)
	@echo "⚠️  Pruning unused Docker data..."
	docker system prune -af

.PHONY: info
info: ## Show project info and URLs
	@echo "DivinityAI — $(MODE_LABEL)"
	@echo ""
	@echo "Development mode: $(DEVELOPMENT_MODE)"
	@echo "Compose files:    $(COMPOSE_FILES)"
	@echo ""
	@echo "URLs:"
ifeq ($(DEVELOPMENT_MODE),true)
	@echo "  Backend API:    http://localhost:8899"
	@echo "  Frontend:       http://localhost:5899"
else
	@echo "  App (nginx):    http://localhost:80"
	@echo "  Backend API:    http://localhost:80/api/"
endif
	@echo ""
	@echo "External dependencies (run on host):"
	@echo "  Ollama:         http://localhost:11434"
	@echo "  ChromaDB:       http://localhost:8040"