# Pengaruh Indikator terhadap Kemiskinan di Kab. Kotawaringin Timur

Dashboard Streamlit berbasis data panel 97 kab/kota (2014-2024). Indikator yang dianalisis:
**Ketimpangan (Gini)** dan **DBH SDA**.

Isi dashboard: gambaran kemiskinan Kotim, pengaruh per indikator (data Kotim), regresi panel
(Pooled, Fixed Effect, Random Effect, uji Hausman), simulasi, dan data.

## Struktur file
```
.
├── app.py                # aplikasi Streamlit
├── dataset_fiskal.csv    # dataset panel
├── requirements.txt      # daftar library
├── README.md
└── .gitignore
```

## Jalankan lokal
```
pip install -r requirements.txt
streamlit run app.py
```

## Push ke GitHub
```
git init
git add .
git commit -m "Dashboard kemiskinan Kotawaringin Timur"
git branch -M main
git remote add origin https://github.com/<username>/<nama-repo>.git
git push -u origin main
```

## Deploy ke Streamlit Community Cloud
1. Buka https://share.streamlit.io lalu login dengan GitHub.
2. Klik **Create app**, pilih repo, branch `main`, Main file path `app.py`.
3. Klik **Deploy**.
