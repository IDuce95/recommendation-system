generate_data:
	- python data/data_generator.py

load_data_to_db:
	- python data/load_data_to_db.py

create_table:
	- PGPASSWORD=password psql -U postgres -d recommendation_system -f data/create_table.sql

preprocess_data:
	- python app/data_processing/data_preprocessor.py

api:
	- python app/api/fastapi_server.py

streamlit:
	- streamlit run app/streamlit_app/streamlit_ui.py