# main.py
import pymongo
import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from scipy.sparse import coo_matrix
import pickle
import os
from datetime import datetime
import redis
import json
from fastapi import FastAPI
from pydantic import BaseModel

# Kết nối MongoDB
MONGODB_URI = "mongodb+srv://hnhu:hoainhu1234@webbuycake.asd8v.mongodb.net/?retryWrites=true&w=majority&appName=WebBuyCake"
client = pymongo.MongoClient(MONGODB_URI)
db = client['test']

# Kết nối Redis
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
except redis.ConnectionError:
    print("Redis không chạy, bỏ qua cache.")
    redis_client = None

# Chuẩn bị dữ liệu
def prepare_data(db, new_orders_only=False, last_timestamp=None):
    dataset = Dataset()
    query = {'createdAt': {'$gt': last_timestamp}} if new_orders_only and last_timestamp else {}
    users = set(str(order['userId']) for order in db.orders.find(query) if order.get('userId'))
    products = set(str(item['product']) for order in db.orders.find(query) for item in order.get('orderItems', []))
    if new_orders_only:
        dataset.fit_partial(users=users, items=products)
    else:
        dataset.fit(users=users, items=products)
    interactions = []
    for order in db.orders.find(query):
        if order.get('userId'):
            user_id = str(order['userId'])
            for item in order.get('orderItems', []):
                product_id = str(item['product'])
                quantity = item.get('quantity', 1)
                interactions.append((user_id, product_id, quantity))
    (interactions_matrix, weights) = dataset.build_interactions(interactions)
    print("Interactions matrix shape:", interactions_matrix.shape)
    return dataset, interactions_matrix

# Huấn luyện hoặc cập nhật mô hình
def train_or_update_model(db, model_path='model.pkl', dataset_path='dataset.pkl', last_timestamp=None):
    if os.path.exists(model_path) and os.path.exists(dataset_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        dataset, interactions_matrix = prepare_data(db, new_orders_only=True, last_timestamp=last_timestamp)
        model.fit_partial(interactions_matrix, epochs=10, num_threads=2, verbose=True)
    else:
        dataset, interactions_matrix = prepare_data(db)
        model = LightFM(loss='warp', random_state=42)
        model.fit(interactions_matrix, epochs=30, num_threads=2, verbose=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)
    db.model_metadata.update_one(
        {'type': 'last_update'},
        {'$set': {'timestamp': datetime.utcnow()}},
        upsert=True
    )
    return model, dataset

# Đánh giá mô hình
def evaluate_model(db, model, dataset, k=5):
    user_ids, item_ids = dataset.mapping()[0], dataset.mapping()[2]
    orders = list(db.orders.find().sort('createdAt', -1))
    test_size = max(1, int(0.1 * len(orders)))
    test_orders = orders[:test_size]
    precisions = []
    recalls = []
    for order in test_orders:
        user_id = str(order['userId'])
        if user_id not in user_ids:
            continue
        actual_items = [str(item['product']) for item in order.get('orderItems', [])]
        recommended = recommend(user_id, None, db, model, dataset, n_items=k)
        relevant = set(actual_items) & set(recommended)
        precision = len(relevant) / len(recommended) if recommended else 0
        recall = len(relevant) / len(actual_items) if actual_items else 0
        precisions.append(precision)
        recalls.append(recall)
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Recall@{k}: {avg_recall:.4f}")
    return avg_precision, avg_recall

# Hàm khuyến nghị
def recommend(user_id, current_product_id, db, model, dataset, n_items=5):
    user_ids, item_ids = dataset.mapping()[0], dataset.mapping()[2]
    cache_key = f"rec:{user_id}:{current_product_id}"
    if redis_client:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    if user_id not in user_ids:
        product = db.products.find_one({'_id': current_product_id})
        if product and 'productCategory' in product:
            category = product['productCategory']
            similar_products = list(db.products.find({'productCategory': category}).limit(n_items))
            recommendations = [str(p['_id']) for p in similar_products if str(p['_id']) != current_product_id]
            if redis_client:
                redis_client.setex(cache_key, 3600, json.dumps(recommendations))
            return recommendations
        return []
    user_idx = user_ids[user_id]
    scores = model.predict(user_idx, np.arange(len(item_ids)))
    top_items = np.argsort(-scores)[:n_items + 1]
    recommendations = [list(item_ids.keys())[i] for i in top_items if list(item_ids.keys())[i] != current_product_id]
    if not recommendations:
        product = db.products.find_one({'_id': current_product_id})
        if product and 'productCategory' in product:
            category = product['productCategory']
            similar_products = list(db.products.find({'productCategory': category}).limit(n_items))
            recommendations = [str(p['_id']) for p in similar_products if str(p['_id']) != current_product_id]
    if redis_client and recommendations:
        redis_client.setex(cache_key, 3600, json.dumps(recommendations))
    return recommendations[:n_items]

# Tính trước khuyến nghị
def precompute_recommendations(db, model, dataset):
    user_ids = dataset.mapping()[0]
    for user_id in user_ids:
        recommended = recommend(user_id, None, db, model, dataset)
        db.recommendations.update_one(
            {'userId': user_id},
            {'$set': {'recommended': recommended, 'updatedAt': datetime.utcnow()}},
            upsert=True
        )

# FastAPI
app = FastAPI()

class RecommendationRequest(BaseModel):
    user_id: str
    product_id: str

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    recommendations = recommend(request.user_id, request.product_id, db, model, dataset)
    return {"recommendations": recommendations}

@app.post("/update-model")
async def update_model():
    last_update = db.model_metadata.find_one({'type': 'last_update'})
    last_timestamp = last_update['timestamp'] if last_update else None
    global model, dataset
    model, dataset = train_or_update_model(db, last_timestamp=last_timestamp)
    precompute_recommendations(db, model, dataset)
    return {"status": "Model updated"}

@app.post("/interaction/log")
async def log_interaction(interaction: dict):
    db.user_interactions.insert_one(interaction)
    return {"status": "success"}

# Khởi tạo mô hình
try:
    model, dataset = train_or_update_model(db)
    precompute_recommendations(db, model, dataset)
except Exception as e:
    print(f"Lỗi khi khởi tạo mô hình: {e}")

# Test logic
def test_recommendation(db):
    try:
        evaluate_model(db, model, dataset)
        user_id = "6756e4441df899603742e267"
        product_id = "67643c2411d943b7bdecb7d3"
        recommended = recommend(user_id, product_id, db, model, dataset)
        print("Sản phẩm khuyến nghị:", recommended)
    except Exception as e:
        print(f"Lỗi khi test khuyến nghị: {e}")

test_recommendation(db)