import os

# Path ke folder yang berisi file
folder_path = 'C:/Users/renat/OneDrive/Documents/skripsi/Code/IMAGES/HasilCrop299/'

# Loop melalui semua file di folder
for filename in os.listdir(folder_path):
    # Cek apakah _512 ada di dalam nama file
    if '_299' in filename:
        # Buat nama file baru tanpa _512
        new_filename = filename.replace('_299', '')
        # Dapatkan path lengkap dari file lama dan file baru
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)
        # Ganti nama file
        os.rename(old_file, new_file)
        print(f"Renamed: {old_file} -> {new_file}")
