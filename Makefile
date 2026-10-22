.PHONY: data eda train powerbi images app test all

# Never overwrite a user's existing input dataset implicitly.
data:
	python -c "from pathlib import Path; p=Path('data/raw/loans.csv'); print(f'Using existing {p}' if p.exists() else 'Generating synthetic demo data'); import subprocess,sys; sys.exit(subprocess.call([sys.executable,'-m','src.generate_data']) if not p.exists() else 0)"
eda:
	python -m src.eda
train:
	python -u -m src.train
powerbi:
	python -m src.export_powerbi
images:
	python -m src.dashboard_images
app:
	python -m streamlit run app/app.py
test:
	python -m pytest -q
all: data eda train powerbi images test
