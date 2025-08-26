"""Train the risk model neural network.

This script trains the risk model using generated training data.
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis_engine.risk_model import RiskModel, RiskNeuralNetwork

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RiskModelTrainer:
    """Trainer for risk model neural network."""
    
    def __init__(self, model_save_dir: Path = None) -> None:
        """Initialize trainer.
        
        Args:
            model_save_dir: Directory to save trained models
        """
        self.risk_model = RiskModel()
        self.model_save_dir = model_save_dir or Path(__file__).parent / 'models'
        self.model_save_dir.mkdir(exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        
    def load_data(self, data_path: Path) -> list:
        """Load training data from JSON file.
        
        Args:
            data_path: Path to data file
            
        Returns:
            List of training samples
        """
        with open(data_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples from {data_path}")
        return data
    
    def prepare_data_loader(self, data: list, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """Prepare PyTorch DataLoader from training data.
        
        Args:
            data: List of training samples
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader object
        """
        # Extract features and targets
        X = []
        y = []
        
        for sample in data:
            features = [
                sample['historical_volatility'],
                sample['atr'],
                sample['rsi'],
                sample['volume_ratio'],
                sample['ma_deviation'],
                sample['beta']
            ]
            X.append(features)
            y.append(sample['target_stop_loss'])
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        # Create dataset and loader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return loader
    
    def train_epoch(self, train_loader: DataLoader, 
                   model: nn.Module,
                   criterion: nn.Module,
                   optimizer: optim.Optimizer) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            model: Neural network model
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            Average loss for the epoch
        """
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            
            # Calculate loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def validate(self, val_loader: DataLoader,
                model: nn.Module,
                criterion: nn.Module) -> tuple:
        """Validate model on validation set.
        
        Args:
            val_loader: Validation data loader
            model: Neural network model
            criterion: Loss function
            
        Returns:
            Tuple of (average loss, predictions, targets)
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_preds.extend(outputs.numpy().flatten())
                all_targets.extend(batch_y.numpy().flatten())
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss, np.array(all_preds), np.array(all_targets)
    
    def train(self, train_data_path: Path, 
             val_data_path: Path,
             epochs: int = 100,
             batch_size: int = 32,
             learning_rate: float = 0.001) -> None:
        """Train the risk model.
        
        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        # Load data
        train_data = self.load_data(train_data_path)
        val_data = self.load_data(val_data_path)
        
        # Prepare data loaders
        train_loader = self.prepare_data_loader(train_data, batch_size, shuffle=True)
        val_loader = self.prepare_data_loader(val_data, batch_size, shuffle=False)
        
        # Initialize model, loss, and optimizer
        model = RiskNeuralNetwork(input_size=6, hidden_size=32, output_size=1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
        
        best_val_loss = float('inf')
        best_model_state = None
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader, model, criterion, optimizer)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_preds, val_targets = self.validate(val_loader, model, criterion)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                
                # Save best model
                model_path = self.model_save_dir / 'best_risk_model.pth'
                torch.save({
                    'model_state_dict': best_model_state,
                    'val_loss': best_val_loss,
                    'epoch': epoch,
                    'model_config': {
                        'input_size': 6,
                        'hidden_size': 32,
                        'output_size': 1
                    }
                }, model_path)
                
            else:
                patience_counter += 1
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}, "
                          f"Best Val Loss: {best_val_loss:.6f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Final validation
        final_val_loss, final_preds, final_targets = self.validate(val_loader, model, criterion)
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        
        # Save final model
        final_model_path = self.model_save_dir / 'final_risk_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'final_val_loss': final_val_loss,
            'model_config': {
                'input_size': 6,
                'hidden_size': 32,
                'output_size': 1
            }
        }, final_model_path)
        
        # Plot training history
        self.plot_training_history()
        
        # Plot predictions vs targets
        self.plot_predictions(final_preds, final_targets)
        
        # Calculate and display metrics
        self.calculate_metrics(final_preds, final_targets)
    
    def plot_training_history(self) -> None:
        """Plot training and validation loss history."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', alpha=0.7)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = self.model_save_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved training history plot to {plot_path}")
    
    def plot_predictions(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Plot predictions vs actual targets.
        
        Args:
            predictions: Model predictions
            targets: Actual target values
        """
        plt.figure(figsize=(10, 6))
        
        # Scatter plot
        plt.scatter(targets, predictions, alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel('Actual Stop-Loss %')
        plt.ylabel('Predicted Stop-Loss %')
        plt.title('Predictions vs Actual Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = self.model_save_dir / 'predictions_vs_actual.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved predictions plot to {plot_path}")
    
    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> dict:
        """Calculate performance metrics.
        
        Args:
            predictions: Model predictions
            targets: Actual target values
            
        Returns:
            Dictionary of metrics
        """
        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print("\n" + "="*60)
        print("Model Performance Metrics")
        print("="*60)
        print(f"Mean Squared Error (MSE): {mse:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"R-squared (R²): {r2:.4f}")
        print("="*60)
        
        # Save metrics
        metrics_path = self.model_save_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics


def main():
    """Main training function."""
    trainer = RiskModelTrainer()
    
    # Data paths
    data_dir = Path(__file__).parent / 'data'
    train_data_path = data_dir / 'train_data.json'
    val_data_path = data_dir / 'val_data.json'
    
    # Check if data exists
    if not train_data_path.exists() or not val_data_path.exists():
        logger.error("Training data not found. Please run generate_training_data.py first.")
        return
    
    # Training parameters
    config = {
        'epochs': 200,
        'batch_size': 32,
        'learning_rate': 0.001
    }
    
    print("\n" + "="*60)
    print("Risk Model Neural Network Training")
    print("="*60)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Train model
    trainer.train(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        **config
    )
    
    print("\nTraining complete! Check the 'models' directory for saved models and plots.")


if __name__ == "__main__":
    main()