migrate:
	set -o allexport; source .env; set +o allexport; python3 migrations.py

run-dev:
	set -o allexport; source .env; set +o allexport; uvicorn main:app --host $$SERVER_HOST --port $$SERVER_PORT --reload