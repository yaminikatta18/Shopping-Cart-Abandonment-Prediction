# Shopping Cart Abandonment Prediction (Decision Tree + Random Forest)

Flask website + scikit-learn models. Machine Learning – 25SC2107E.

## Run in VS Code
1. Open this folder in VS Code (File > Open Folder).
2. Open a terminal (Ctrl + `) and create a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python generate_data.py`   (creates data/cart_data.csv)
5. `python train.py`           (trains models, saves models/ and metrics.json)
6. `python app.py`             then open http://127.0.0.1:5000

(If you skip steps 4-5, app.py trains automatically on first start.)

## Structure
- generate_data.py: synthetic dataset (5 features + `abandoned` target)
- train.py: Decision Tree, Random Forest, CV, metrics, feature importance
- app.py: Flask site and `/api/predict`, `/api/metrics`
- templates/, static/: the website (Chart.js loads from a CDN, so internet is needed)

## Using a real dataset
Replace data/cart_data.csv with your own, keeping columns: browsing_time,
products_viewed, cart_value, previous_purchases, discount_usage, abandoned (0/1),
then re-run `python train.py`.

Note: the bundled data is synthetic, so metrics demonstrate the pipeline, not real-world performance.
