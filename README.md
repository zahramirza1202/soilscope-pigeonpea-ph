# soilscope-pigeonpea-ph
Streamlit app for soil pH prediction and pigeonpea suitability screening

## How to run
pip install streamlit pandas numpy scikit-learn altair
streamlit run main.py

## Dataset
Kaggle Crop Recommendation dataset (Crop_recommendation.csv).
Upload it in the app to start.

## Note
The pH models do not predict pH better than the dataset average, so the app recommends using a measured soil-test pH.