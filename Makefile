.PHONY: data eda train powerbi images app test all
data:  ; python -m src.generate_data
eda:   ; python -m src.eda
train: ; python -m src.train
powerbi: ; python -m src.export_powerbi
images: ; python -m src.dashboard_images
app:   ; python -m streamlit run app/app.py
test:  ; python -m pytest -q
all: data eda train powerbi images test
