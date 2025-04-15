import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Contoh data matriks konfusi (ganti dengan data Anda sendiri)
confusion_matrix_data = np.array([
    [170  ,11   ,11   ,6   ,4],
    [  38,136  ,8   ,1   ,2],
    [  30,4 ,126   ,5   ,4],
    [  0   ,0   ,0 ,154   ,0],
    [  0   ,0   ,0   ,0 ,176]
])

# Label untuk setiap kelas
class_labels = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]

# Fungsi untuk memvisualisasikan matriks konfusi
def plot_confusion_matrix(conf_matrix, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

# Panggil fungsi untuk menampilkan matriks konfusi
plot_confusion_matrix(confusion_matrix_data, class_labels)
