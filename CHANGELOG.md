# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-07-15

### Added
- Created `.env.example` to document required configuration environment keys.
- Added dynamic CORS origin configuration in `app/core/config.py` and `main.py` via the `CORS_ORIGINS` environment setting.
- Created open-source documentation templates: `README.md`, `LICENSE` (MIT), `CONTRIBUTING.md`, and `CHANGELOG.md`.

### Changed
- Refactored `process_stream.py` to be a command-line script accepting arguments and reading configuration defaults from environment variables.
- Configured RabbitMQ credentials in `docker-compose.yml` to utilize environment variables.
- Changed absolute local volumes path `/home/admin1/Zydus/Backend/uploads` in `docker-compose.yml` to a relative path `./uploads`.

### Removed
- Deleted redundant duplicate script `process_stream1.py`.
- Deleted temporary/scratch files: `temp.txt` and untracked profile pictures `Mahek_*.jpg`.
- Removed unused `Flask` and `flask-cors` packages from `requirements.txt`.
- Fixed the version pin typo `uvicorn==0.38.0lapx==0.5.5` in `requirements.txt`.
