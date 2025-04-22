import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# Load your CSV file
csv_path = "mse_training.csv"  # <-- change this path
df = pd.read_csv(csv_path)

# Extract data
epochs = df['epoch'].values
losses = df['mse_loss'].values

# Polynomial fit (degree 2 or 3 for smooth trend)
degree = 3
coeffs = Polynomial.fit(epochs, losses, deg=degree)
smoothed_epochs = np.linspace(epochs.min(), epochs.max(), 500)
smoothed_losses = coeffs(smoothed_epochs)

# Plot original loss and trendline
plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, label="Original MSE Loss", alpha=0.5, linestyle='--')
plt.plot(smoothed_epochs, smoothed_losses, label=f"Fitted Trend (deg={degree})", linewidth=2)

plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("MSE Loss Trend with Polynomial Fit")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot
plot_save_path = "mse_loss_trendline.png"  # <-- change this path too
plt.savefig(plot_save_path)
plt.close()

print(f"Loss trendline plot saved to: {plot_save_path}")
