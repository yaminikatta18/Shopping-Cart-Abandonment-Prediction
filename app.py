"""Flask app: serves the website and a JSON prediction API."""
import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).parent
FEATURES = ["browsing_time", "products_viewed", "cart_value",
            "previous_purchases", "discount_usage"]
LABELS = {"browsing_time": "Browsing time", "products_viewed": "Products viewed",
          "cart_value": "Cart value", "previous_purchases": "Previous purchases",
          "discount_usage": "Discount usage"}
LIMITS = {"browsing_time": (0, 240), "products_viewed": (0, 200),
          "cart_value": (0, 100000), "previous_purchases": (0, 500),
          "discount_usage": (0, 1)}

app = Flask(__name__)

if not (ROOT / "models" / "metrics.json").exists():
    import train
    train.main()

dt = joblib.load(ROOT / "models" / "decision_tree.joblib")
rf = joblib.load(ROOT / "models" / "random_forest.joblib")
METRICS = json.loads((ROOT / "models" / "metrics.json").read_text())


def tree_path(row):
    """Human-readable decision rules the Decision Tree followed for this customer."""
    t = dt.tree_
    steps = []
    for node in dt.decision_path(row).indices:
        if t.children_left[node] == t.children_right[node]:
            continue  # leaf
        f, thr = FEATURES[t.feature[node]], t.threshold[node]
        val = float(row.iloc[0][f])
        op = "<=" if val <= thr else ">"
        steps.append(f"{LABELS[f]} = {val:g} {op} {thr:.2f}")
    return steps


def advice(p):
    if p >= 0.7:
        return "High risk: send a reminder plus a personalised discount or free shipping now."
    if p >= 0.4:
        return "Medium risk: show a cart reminder or limited-time offer."
    return "Low risk: likely to complete the purchase, no intervention needed."


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/metrics")
def metrics():
    return jsonify(METRICS)


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        values = {f: float(data[f]) for f in FEATURES}
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Please provide all five numeric features."), 400
    for f, v in values.items():
        lo, hi = LIMITS[f]
        if not lo <= v <= hi:
            return jsonify(error=f"{LABELS[f]} must be between {lo} and {hi}."), 400

    row = pd.DataFrame([values], columns=FEATURES)
    out = {}
    for name, model in [("decision_tree", dt), ("random_forest", rf)]:
        p = float(model.predict_proba(row)[0, 1])
        out[name] = {"probability": p, "label": "Cart Abandoned" if p >= 0.5 else "Purchase Completed"}
    out["decision_tree"]["path"] = tree_path(row)
    out["advice"] = advice(out["random_forest"]["probability"])
    return jsonify(out)


if __name__ == "__main__":
    app.run(debug=True)
