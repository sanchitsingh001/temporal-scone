import pickle
import matplotlib.pyplot as plt

# Load the pickle file
file_path = 'scone_1_1_0.05_1_1_1.5_0.5_0.1.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

fpr = data['fpr95_test']

# City ranges
city0_fpr = fpr[0:101]     # Epochs 0–100
city1_fpr = fpr[101:201]   # Epochs 101–200
city2_fpr = fpr[201:301]   # Epochs 201–300

# Compute means
mean_city0 = sum(city0_fpr) / len(city0_fpr)
mean_city1 = sum(city1_fpr) / len(city1_fpr)

# Sample every 10 epochs
city1_sampled = [fpr[i] for i in range(101, 200, 10)]
city2_sampled = [fpr[i] for i in range(201, 300, 10)]

x_axis = list(range(0, 100, 10))  # Relative epoch index

# --------- Plot 1: City 1 values vs. mean of City 0 ---------
plt.figure(figsize=(10, 6))
plt.plot(x_axis, city1_sampled, marker='o', linestyle='-', label='City 1 (every 10 epochs)', color='orange')
plt.axhline(mean_city0, color='blue', linestyle='--', label=f'Mean of City 0: {mean_city0:.4f}')
plt.title('City 1 FPR vs. Mean FPR of City 0')
plt.xlabel('Relative Epoch (0–100)')
plt.ylabel('FPR95')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('city1_vs_mean_city0.png')
print("Saved: 'city1_vs_mean_city0.png'")

# --------- Plot 2: City 2 values vs. mean of City 1 ---------
plt.figure(figsize=(10, 6))
plt.plot(x_axis, city2_sampled, marker='s', linestyle='-', label='City 2 (every 10 epochs)', color='green')
plt.axhline(mean_city1, color='purple', linestyle='--', label=f'Mean of City 1: {mean_city1:.4f}')
plt.title('City 2 FPR vs. Mean FPR of City 1')
plt.xlabel('Relative Epoch (0–100)')
plt.ylabel('FPR95')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('city2_vs_mean_city1.png')
print("Saved: 'city2_vs_mean_city1.png'")
