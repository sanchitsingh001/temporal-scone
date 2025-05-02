import pickle
import matplotlib.pyplot as plt

# Load the pickle file
file_path = 'scone_1_1_0.05_1_1_1.5_0.5_0.1.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

fpr = data['fpr95_test']

# Compute aligned differences every 10 epochs for City 1 vs City 0
city_1_vs_0_diff = [fpr[i] - fpr[i + 101] for i in range(0, 100, 10)]
city_2_vs_1_diff = [fpr[i + 101] - fpr[i + 201] for i in range(0, 100, 10)]


x_axis = list(range(0, 100, 10))  # Relative epoch index (0–90)

# Plot 1: City 1 vs City 0
plt.figure(figsize=(10, 6))
plt.plot(x_axis, city_1_vs_0_diff, marker='o', color='orange')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title('FPR95 Difference: City 1 - City 0 (every 10 epochs, aligned)')
plt.xlabel('Relative Epoch (0–100)')
plt.ylabel('FPR95 Difference')
plt.grid(True)
plt.tight_layout()
plt.savefig('aligned_city1_vs_city0.png')
print("Saved: 'aligned_city1_vs_city0.png'")

# Plot 2: City 2 vs City 1
plt.figure(figsize=(10, 6))
plt.plot(x_axis, city_2_vs_1_diff, marker='s', color='green')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title('FPR95 Difference: City 2 - City 1 (every 10 epochs, aligned)')
plt.xlabel('Relative Epoch (0–100)')
plt.ylabel('FPR95 Difference')
plt.grid(True)
plt.tight_layout()
plt.savefig('aligned_city2_vs_city1.png')
print("Saved: 'aligned_city2_vs_city1.png'")
