import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import io
import os

# Inisialisasi state untuk menandakan bahwa aplikasi baru saja dimulai
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.page = 'Home'
    st.session_state.show_label_column = True  # Menandakan apakah kolom label harus ditampilkan
    st.session_state.cluster_1 = None
    st.session_state.cluster_2 = None
    
# Fungsi untuk membaca atau membuat dataset
def load_or_create_dataset():
    try:
        df = pd.read_csv("data/customers.csv")
    except FileNotFoundError: 
        df = pd.DataFrame(columns=['label', 'stok_awal', 'stok_akhir', 'terjual'])
    return df

# Fungsi untuk Elbow Method
def elbow_method(data):
    clusters = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i).fit(data)
        clusters.append(km.inertia_)

    # Mencari indeks elbow (titik di mana penurunan inersia tidak signifikan)
    elbow_index = -1
    for i in range(1, len(clusters) - 1):
        if (clusters[i - 1] - clusters[i]) / (clusters[i] - clusters[i + 1]) > 0.1:
            elbow_index = i
            break

    # Plot hasil Elbow Method dengan penanda elbow
    fig_elbow, ax_elbow = plt.subplots(figsize=(12, 8))
    sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax_elbow, marker='o')
    ax_elbow.set_title("Elbow Method")
    ax_elbow.set_xlabel("Jumlah Cluster")
    ax_elbow.set_ylabel("Inertia")
    
    # Menambahkan penanda elbow
    optimal_clusters = elbow_index + 1
    ax_elbow.annotate('Elbow Point', xy=(optimal_clusters, clusters[elbow_index]), xytext=(optimal_clusters, clusters[elbow_index] + 100), arrowprops=dict(facecolor='red', shrink=0.05))
    
    st.pyplot(fig_elbow)

    return optimal_clusters


# Fungsi untuk melakukan clustering
def perform_clustering(data, n_clusters):
    original_data = data.copy()  # Salin data asli sebelum clustering

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data['cluster'] = kmeans.fit_predict(data[['stok_awal', 'stok_akhir', 'terjual']])
    center_points = pd.DataFrame(kmeans.cluster_centers_, columns=['stok_awal', 'stok_akhir', 'terjual'])
    
    return data, kmeans.inertia_, center_points  # Tambahkan return untuk data asli

def clustering(original_data) :
    
    clean_data = original_data.drop(['label'], axis=1)
    
    st.subheader("1. Menentukan jumlah cluster optimal")
    # Elbow Method
    optimal_clusters = elbow_method(clean_data)
    
    st.subheader("2. Melakukan Perhitungan K-Means")
    # Plot hasil clustering
    data, _, center_points = perform_clustering(clean_data, optimal_clusters)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x='stok_awal', y='stok_akhir', hue='cluster', data=data, palette='viridis', ax=ax)
    sns.scatterplot(x=center_points['stok_awal'], y=center_points['stok_akhir'], color='red', marker='X', s=300, label='Center Point', ax=ax)
    for i, center in center_points.iterrows():
        ax.text(center['stok_awal'], center['stok_akhir'], f'Cluster {i}', fontsize=12, color='black', ha='center', va='top')
    ax.set_title("Hasil Clustering dengan KMeans")
    ax.set_xlabel("Stok Awal")
    ax.set_ylabel("Stok Akhir")
    st.pyplot(fig)

    st.write("Inertia (Jumlah total jarak kuadrat antar data dengan center cluster):", int(_))
    
    st.session_state.show_label_column = True
    
    # Tampilkan data sesuai kluster dalam tabel
    st.subheader("Hasil Clustering :")
    for cluster_id in range(optimal_clusters):
        cluster_data = data[data['cluster'] == cluster_id]
        st.write(f"Cluster {cluster_id}:")

        # Tampilkan kolom label sesuai indeks data dalam cluster
        if 'label' in original_data.columns:
            # Dapatkan label dari data asli berdasarkan indeks
            cluster_data['Nama Pakaian'] = original_data.loc[cluster_data.index, 'label'].values
        else:
            # Jika kolom 'label' tidak ada, gunakan indeks data sebagai label
            cluster_data['Nama Pakaian'] = cluster_data.index.astype(str)

        if 'Nama Pakaian' in cluster_data.columns:
            st.table(cluster_data[['Nama Pakaian', 'stok_awal', 'stok_akhir', 'terjual']].style.format(precision=0))
        else:
            st.table(cluster_data[['stok_awal', 'stok_akhir', 'terjual']].style.format(precision=0))

        # Simpan hasil clustering ke dalam session state
        st.session_state[f"cluster_{cluster_id + 1}"] = cluster_data
    
    # Tombol "Kembali ke Data Awal"
    if st.button("Kembali ke Data Awal"):
        st.session_state.show_label_column = True
        st.experimental_rerun()  # Merender ulang aplikasi
    
# Halaman Home
def home():
    st.title("Customer Clustering App")
    st.write("Laporan dan Analisis Data Customer")
    st.write("Current Working Directory:", os.getcwd())
    
    st.header("5 Data Terbaru")
    original_data = load_or_create_dataset()
    
    # Data terbaru bersama dengan kolom label
    latest_data = load_or_create_dataset().tail(5)
    
    # Ganti nama kolom sesuai dengan yang diinginkan
    latest_data.columns = ['Stok Awal', 'Stok Akhir', 'Terjual','Nama Pakaian']

    st.table(latest_data.style.format(precision=0))

    # Tombol "Mulai Clustering"
    if st.button("Mulai Clustering"):
        clustering(original_data)

# Halaman Input Data
def input_data():
    st.title("Input Data Baru")
    new_data = st.text_input("Nama Pakaian")
    stok_awal = st.number_input("Stok Awal")
    stok_akhir = st.number_input("Stok Akhir")
    terjual = st.number_input("Terjual")

    if st.button("Tambahkan Data"):
        new_row = {'label': new_data, 'stok_awal': stok_awal, 'stok_akhir': stok_akhir, 'terjual': terjual}
        df = pd.concat([load_or_create_dataset(), pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv("data/customers.csv", index=False)
        st.success("Data berhasil ditambahkan!")

@st.cache_data
def convert_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

# Halaman Cetak Laporan
def cetak_laporan():
    st.title("Cetak Laporan")
    st.write("Hasil Clustering") 
    
    if st.session_state.cluster_1 is None:
        st.warning("Silahkan lakukan clustering terlebih dulu")
    else:
        # Create a Pandas DataFrame to combine all cluster data
        combined_data = pd.concat([
            st.session_state.cluster_1,
            st.session_state.cluster_2
        ], ignore_index=True)

        # menampilkan seluruh data cluster
        st.subheader("Hasil data clustering :")
        st.table(combined_data[['Nama Pakaian', 'stok_awal', 'stok_akhir', 'terjual','cluster']].style.format(precision=0))

        # Tombol untuk download seluruh data cluster format Excel 
        if st.button("Download Combined Data as Excel"):
            output = BytesIO()
            combined_data.to_excel(output, index=False, sheet_name='Combined_Data')
            output.seek(0)
            st.write("Combined data has been saved as an Excel file. Please [download it here](data/Combined_Cluster_Results.xlsx)", unsafe_allow_html=True)

    for cluster_id in range(1, 3):
        if st.session_state[f"cluster_{cluster_id}"] is None or st.session_state[f"cluster_{cluster_id}"].empty:
            st.warning(f"Cluster {cluster_id} belum ditemukan. Silahkan lakukan clustering terlebih dulu.")
        else:        
            st.subheader(f"Cluster {cluster_id}")
            st.table(st.session_state[f"cluster_{cluster_id}"][['Nama Pakaian', 'stok_awal', 'stok_akhir', 'terjual']].style.format(precision=0))
            
            # Tombol Download CSV
            csv_data = convert_to_csv(st.session_state[f"cluster_{cluster_id}"])
            download_csv_button = st.download_button(
                label=f"Download Data Cluster {cluster_id} format CSV",
                data=csv_data,
                file_name=f'Cluster_{cluster_id}_Results.csv',
                mime='text/csv'
            )

            # Tombol Download Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                st.session_state[f"cluster_{cluster_id}"].to_excel(writer, sheet_name=f'Cluster_{cluster_id}_Results', index=False)

            download_excel_button = st.download_button(
                label=f"Download Data Cluster {cluster_id} format Excel",
                data=excel_buffer,
                file_name=f'Cluster_{cluster_id}_Results.xlsx',
                mime='application/vnd.ms-excel'
            )

def main():
    st.sidebar.title("Menu")
    
    # Inisialisasi state untuk menandakan bahwa aplikasi baru saja dimulai
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.page = 'Home'
    else:
        home_page = st.sidebar.button("Home")
        input_data_page = st.sidebar.button("Input Data")
        cetak_laporan_page = st.sidebar.button("Cetak Laporan")

        if home_page:
            st.session_state.page = 'Home'
        elif input_data_page:
            st.session_state.page = 'Input Data'
        elif cetak_laporan_page:
            st.session_state.page = 'Cetak Laporan'

    # Menampilkan halaman berdasarkan state
    if st.session_state.page == 'Home':
        home()
    elif st.session_state.page == 'Input Data':
        input_data()
    elif st.session_state.page == 'Cetak Laporan':
        cetak_laporan()

if __name__ == "__main__":
    main()
