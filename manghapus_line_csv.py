import pandas as pd
import os

# Path ke file CSV dan folder gambar
csv_file_path = 'C:/Users/renat/OneDrive/Documents/skripsi/Code/messidor_data.csv'
images_folder_path = 'C:/Users/renat/OneDrive/Documents/skripsi/Code/IMAGES/HasilCrop512/'

# Membaca file CSV
df = pd.read_csv(csv_file_path)

# Daftar nama file gambar di folder
image_files = set(os.listdir(images_folder_path))

# Memastikan nama file gambar hanya mengandung ekstensi gambar yang diizinkan
image_files = set([f.lower().strip() for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

# Pastikan kolom 'image_id' ada di CSV
if 'image_id' not in df.columns:
    raise ValueError("Kolom 'image_id' tidak ditemukan dalam file CSV")

# Membersihkan dan menormalkan nilai pada kolom 'image_id'
df['image_id'] = df['image_id'].str.lower().str.strip()

# Menghapus baris yang tidak memiliki gambar yang sesuai di folder
df_cleaned = df[df['image_id'].isin(image_files)]

# Menghapus duplikasi berdasarkan kolom 'image_id'
df_cleaned = df_cleaned.drop_duplicates(subset=['image_id'])

# Menyimpan CSV yang telah diperbarui
df_cleaned.to_csv('C:/Users/renat/OneDrive/Documents/skripsi/Code/IMAGES/HasilCrop512/cleaned_file.csv', index=False)

print("File CSV telah diperbarui.")
