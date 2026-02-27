from flask import Flask, render_template, request
from model.regression import process_csv
import os
import glob

app = Flask(__name__)

def clear_static_png():
    """Hapus semua file PNG di folder static"""
    png_files = glob.glob('static/*.png')
    for file in png_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

@app.route('/')
def index():
    # Hapus semua PNG saat mengakses halaman utama
    clear_static_png()
    return render_template("index.html")

@app.route('/process', methods=['POST'])
def process():
    file = request.files.get('file')

    if not file:
        return "File tidak ditemukan"
    
    # Hapus semua PNG sebelum memproses file baru
    clear_static_png()

    # Proses file
    (a, b, a_full, b_full,
     mse, r2, mse_full, r2_full,
     preview_awal,
     preview_regresi,
     info_awal,
     info_setelah,
     df_pred,
     statistik) = process_csv(file)

    return render_template(
        "result.html",

        # koefisien model split
        a=round(a, 4),
        b=round(b, 2),

        # koefisien model tanpa split
        a_full=round(a_full, 4),
        b_full=round(b_full, 2),

        # evaluasi split 80:20
        mse=round(mse, 2),
        r2=round(r2, 4),

        # evaluasi tanpa split
        mse_full=round(mse_full, 2),
        r2_full=round(r2_full, 4),

        # STATISTIK DESKRIPTIF
        stat=statistik,

        table_awal=preview_awal.to_html(
            index=False,
            classes="table table-bordered"
        ),

        table_regresi=preview_regresi.to_html(
            index=False,
            classes="table table-bordered"
        ),

        table_prediksi=df_pred.to_html(
            index=False,
            classes="table table-bordered"
        ),

        info_awal=info_awal,
        info_setelah=info_setelah,

        # Nama file gambar sesuai dengan regression.py
        img_harian="daily.png",
        img_bulanan="monthly.png",
        img_tahunan="yearly.png",
        img_regresi_split="regression_split.png",
        img_regresi_tanpa_split="regression_no_split.png",
        img_regresi="regression.png"
    )

if __name__ == "__main__":
    # Buat folder static jika belum ada
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)