#!/bin/bash

# Risk Model Neural Network Training Pipeline
# This script runs the complete training pipeline

echo "========================================"
echo "Risk Model Neural Network Training"
echo "========================================"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "Please install uv first"
    exit 1
fi

# Install required packages
echo "📦 Installing required packages..."
uv add torch torchvision talib-binary matplotlib scikit-learn --quiet

# Step 1: Generate training data
echo ""
echo "Step 1: Generating training data..."
echo "------------------------------------"
echo "This will download historical stock data and may take several minutes."
echo ""

read -p "Do you want to generate new training data? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    uv run python generate_training_data.py
    if [ $? -ne 0 ]; then
        echo "❌ Failed to generate training data"
        exit 1
    fi
else
    echo "Skipping data generation. Using existing data."
fi

# Check if training data exists
if [ ! -f "data/train_data.json" ] || [ ! -f "data/val_data.json" ]; then
    echo "❌ Training data not found. Please generate training data first."
    exit 1
fi

# Step 2: Train the model
echo ""
echo "Step 2: Training the neural network..."
echo "------------------------------------"
echo "This will train the risk model. Training may take 10-30 minutes."
echo ""

read -p "Do you want to train the model? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    uv run python train_risk_model.py
    if [ $? -ne 0 ]; then
        echo "❌ Training failed"
        exit 1
    fi
else
    echo "Skipping model training."
fi

# Step 3: Test the trained model
echo ""
echo "Step 3: Testing the trained model..."
echo "------------------------------------"

if [ -f "models/best_risk_model.pth" ]; then
    uv run python test_model.py
    if [ $? -ne 0 ]; then
        echo "⚠️  Testing failed, but model may still be usable"
    fi
else
    echo "❌ No trained model found. Please train the model first."
fi

echo ""
echo "========================================"
echo "✅ Training pipeline complete!"
echo "========================================"
echo ""
echo "Generated files:"
echo "  - data/train_data.json: Training dataset"
echo "  - data/val_data.json: Validation dataset"
echo "  - models/best_risk_model.pth: Best model checkpoint"
echo "  - models/final_risk_model.pth: Final model checkpoint"
echo "  - models/training_history.png: Loss curve plot"
echo "  - models/predictions_vs_actual.png: Prediction accuracy plot"
echo "  - models/metrics.json: Performance metrics"
echo ""
echo "The trained model can now be used in the main trading system."