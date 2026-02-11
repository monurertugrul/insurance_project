#!/bin/bash

echo "🧹 Starting cleanup..."

# Remove QLoRA and TinyLlama training artifacts
rm -rf adapter_tinyllama
rm -rf models/llama_adapter
rm -rf models/llama_qlora_insurance
rm -rf models/llama_qlora_structured
rm -f llama_agent.py
rm -f qlora_inference.py
rm -f qlora_modal_train.py
rm -f agents/qlora_agent.py

# Remove deep neural network experiments
rm -rf deep_neural_network
rm -f deep_neural_network.py
rm -f dnn_evaluator.py
rm -f modal_inference_dnn.py
rm -f modal_train_dnn.py
rm -f neural_network_agent.py
rm -f agents/deep_neural_network.py
rm -f agents/modal_train_dnn.py

# Remove Modal client and training scripts
rm -f modal_client.py
rm -f modal_train_dnn.py
rm -f modal_inference_dnn.py

# Remove duplicate datasets
rm -rf medical-insurance-cost-prediction
rm -f medical-insurance-cost-prediction.zip
rm -rf medical_insurance
rm -f train.csv
rm -f test.csv
rm -f validation.csv

# Remove virtual environments
rm -rf venv_local
rm -rf .venv

# Remove caches
find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove orphan files
rm -f 2
rm -f app.py
rm -f insurance_logs.txt
rm -f setup.py
rm -f xgb_model.json  # duplicate of rag/xgb_model.json

echo "✨ Cleanup complete!"
