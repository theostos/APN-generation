import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.algebra.generator import draw_sparse_polynomials
from src.algebra.polynomial import evaluate_polynomials, load_power_table
from src.algebra.function import compute_derivative, compute_delta_max, load_add_table

exponent = 6
device = 'cpu'
NUM_ECH = 1_000

T = load_add_table(exponent)
X = load_power_table(exponent)
field_size = 2**exponent
delta_max_list = []
for k in tqdm(range(2,65)):
    P = draw_sparse_polynomials(field_size, k, NUM_ECH)
    F = evaluate_polynomials(P, X, T)
    DF = compute_derivative(F, T)
    delta_max = compute_delta_max(DF)
    delta_max_list.append(delta_max)
# Create the animation
fig, ax = plt.subplots()

# Initialize histogram
n_bins = 64
ax.set_xlim(0, 14)
ax.set_ylim(0, NUM_ECH)  # Adjust based on expected frequency
ax.set_xlabel("Delta Max")
ax.set_ylabel("Frequency")

# Update function for animation
def update(frame):
    ax.clear()
    ax.hist(delta_max_list[frame], bins=n_bins, color='blue', alpha=0.7)
    ax.set_title(f"Histogram of delta_max (sparse polynomial with {frame+1} non zero coefficients)")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, NUM_ECH)  # Adjust based on expected frequency
    ax.set_xlabel("Delta Max")
    ax.set_ylabel("Frequency")

# Create animation
anim = FuncAnimation(fig, update, frames=len(delta_max_list), interval=2000)

# Export as GIF
anim.save("export/delta_max_histogram.gif", writer=PillowWriter(fps=1/2))

