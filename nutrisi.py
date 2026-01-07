# nutrisi.py

# --- 1. DAFTAR 50 KELAS (Urutan Alfabetis MUTLAK - Sesuai Dataset Folder) ---
# JANGAN MENGUBAH URUTAN INI karena model AI menggunakan indeks posisi.
CLASS_NAMES = [
    'Apple', 'Apricot', 'Avocado', 'Banana', 'Beans', 'Beetroot', 'Blackberrie', 
    'Blueberry', 'Cabbage red', 'Cactus fruit', 'Caju seed', 'Cantaloupe', 
    'Carambula', 'Carrot', 'Cauliflower', 'Cherimoya', 'Cherry', 'Chestnut', 
    'Clementine', 'Cocos', 'Corn', 'Cucumber', 'Dates', 'Eggplant', 'Fig', 
    'Ginger', 'Gooseberry', 'Granadilla', 'Grape Blue', 'Grapefruit Pink', 
    'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 
    'Lemon', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mangostan', 'Maracuja', 
    'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Nut', 'Onion', 'Orange'
]

# --- 2. DATABASE NUTRISI ---
# Kunci (Key) di sini disesuaikan agar bisa dicocokkan dengan CLASS_NAMES
NUTRISI_DATA = {
    'Apple': {"Kalori": "52 kcal", "Vitamin": "C", "Serat": "2.4g", "Manfaat": "Kesehatan jantung & pencernaan."},
    'Apricot': {"Kalori": "48 kcal", "Vitamin": "A, C", "Serat": "2.0g", "Manfaat": "Kesehatan mata & kulit."},
    'Avocado': {"Kalori": "160 kcal", "Vitamin": "K, E", "Serat": "6.7g", "Manfaat": "Lemak sehat untuk otak."},
    'Banana': {"Kalori": "89 kcal", "Vitamin": "B6, C", "Serat": "2.6g", "Manfaat": "Sumber energi & Kalium."},
    'Beans': {"Kalori": "31 kcal", "Vitamin": "C, K", "Serat": "3.4g", "Manfaat": "Protein nabati & tulang."},
    'Beetroot': {"Kalori": "43 kcal", "Vitamin": "Folat", "Serat": "2.8g", "Manfaat": "Stamina & tekanan darah."},
    'Blackberrie': {"Kalori": "43 kcal", "Vitamin": "C, K", "Serat": "5.3g", "Manfaat": "Antioksidan tinggi."},
    'Blueberry': {"Kalori": "57 kcal", "Vitamin": "C, K", "Serat": "2.4g", "Manfaat": "Fungsi memori otak."},
    'Cabbage red': {"Kalori": "31 kcal", "Vitamin": "C, K", "Serat": "2.1g", "Manfaat": "Anti-inflamasi alami."},
    'Cactus fruit': {"Kalori": "41 kcal", "Vitamin": "C", "Serat": "3.6g", "Manfaat": "Meningkatkan imun tubuh."},
    'Caju seed': {"Kalori": "553 kcal", "Vitamin": "E, K", "Serat": "3.3g", "Manfaat": "Kacang mete, sumber mineral."},
    'Cantaloupe': {"Kalori": "34 kcal", "Vitamin": "A, C", "Serat": "0.9g", "Manfaat": "Hidrasi & kesehatan kulit."},
    'Carambula': {"Kalori": "31 kcal", "Vitamin": "C", "Serat": "2.8g", "Manfaat": "Belimbing, rendah kalori."},
    'Carrot': {"Kalori": "41 kcal", "Vitamin": "A", "Serat": "2.8g", "Manfaat": "Kesehatan mata & imun."},
    'Cauliflower': {"Kalori": "25 kcal", "Vitamin": "C, K", "Serat": "2.0g", "Manfaat": "Kesehatan sel tubuh."},
    'Cherimoya': {"Kalori": "75 kcal", "Vitamin": "C, B6", "Serat": "3g", "Manfaat": "Kesehatan mood & imun."},
    'Cherry': {"Kalori": "50 kcal", "Vitamin": "C, A", "Serat": "1.6g", "Manfaat": "Kualitas tidur & sendi."},
    'Chestnut': {"Kalori": "196 kcal", "Vitamin": "C, B6", "Serat": "8.1g", "Manfaat": "Energi karbohidrat sehat."},
    'Clementine': {"Kalori": "47 kcal", "Vitamin": "C", "Serat": "1.7g", "Manfaat": "Kekebalan tubuh."},
    'Cocos': {"Kalori": "354 kcal", "Vitamin": "Mangan", "Serat": "9.0g", "Manfaat": "Elektrolit & lemak sehat."},
    'Corn': {"Kalori": "86 kcal", "Vitamin": "B", "Serat": "2.7g", "Manfaat": "Energi & kesehatan mata."},
    'Cucumber': {"Kalori": "15 kcal", "Vitamin": "K", "Serat": "0.5g", "Manfaat": "Hidrasi tubuh."},
    'Dates': {"Kalori": "277 kcal", "Vitamin": "B6", "Serat": "6.7g", "Manfaat": "Serat tinggi & energi."},
    'Eggplant': {"Kalori": "25 kcal", "Vitamin": "K", "Serat": "3.0g", "Manfaat": "Kesehatan jantung."},
    'Fig': {"Kalori": "74 kcal", "Vitamin": "B6", "Serat": "2.9g", "Manfaat": "Kesehatan pencernaan."},
    'Ginger': {"Kalori": "80 kcal", "Vitamin": "C", "Serat": "2.0g", "Manfaat": "Anti-mual & radang."},
    'Gooseberry': {"Kalori": "44 kcal", "Vitamin": "C", "Serat": "4.3g", "Manfaat": "Kesehatan ginjal."},
    'Granadilla': {"Kalori": "97 kcal", "Vitamin": "A, C", "Serat": "10.4g", "Manfaat": "Sangat tinggi serat."},
    'Grape Blue': {"Kalori": "69 kcal", "Vitamin": "K", "Serat": "0.9g", "Manfaat": "Antioksidan resveratrol."},
    'Grapefruit Pink': {"Kalori": "42 kcal", "Vitamin": "A, C", "Serat": "1.6g", "Manfaat": "Metabolisme tubuh."},
    'Guava': {"Kalori": "68 kcal", "Vitamin": "C", "Serat": "5.4g", "Manfaat": "Vitamin C sangat tinggi."},
    'Hazelnut': {"Kalori": "628 kcal", "Vitamin": "E", "Serat": "9.7g", "Manfaat": "Kesehatan jantung."},
    'Huckleberry': {"Kalori": "37 kcal", "Vitamin": "C", "Serat": "2.4g", "Manfaat": "Antioksidan."},
    'Kaki': {"Kalori": "70 kcal", "Vitamin": "A", "Serat": "3.6g", "Manfaat": "Kesehatan penglihatan."},
    'Kiwi': {"Kalori": "61 kcal", "Vitamin": "C, K", "Serat": "3.0g", "Manfaat": "Pencernaan & Imun."},
    'Kohlrabi': {"Kalori": "27 kcal", "Vitamin": "C", "Serat": "3.6g", "Manfaat": "Metabolisme sel."},
    'Kumquats': {"Kalori": "71 kcal", "Vitamin": "C", "Serat": "6.5g", "Manfaat": "Bisa dimakan kulitnya."},
    'Lemon': {"Kalori": "29 kcal", "Vitamin": "C", "Serat": "2.8g", "Manfaat": "Detoksifikasi tubuh."},
    'Limes': {"Kalori": "30 kcal", "Vitamin": "C", "Serat": "2.8g", "Manfaat": "Kesehatan kulit & imun."},
    'Lychee': {"Kalori": "66 kcal", "Vitamin": "C", "Serat": "1.3g", "Manfaat": "Antioksidan flavonoid."},
    'Mandarine': {"Kalori": "53 kcal", "Vitamin": "C", "Serat": "1.8g", "Manfaat": "Menjaga daya tahan."},
    'Mango': {"Kalori": "60 kcal", "Vitamin": "A, C", "Serat": "1.6g", "Manfaat": "Sistem kekebalan."},
    'Mangostan': {"Kalori": "73 kcal", "Vitamin": "B9, C", "Serat": "1.8g", "Manfaat": "Anti-inflamasi kuat."},
    'Maracuja': {"Kalori": "97 kcal", "Vitamin": "C", "Serat": "10g", "Manfaat": "Rileksasi & Serat."},
    'Melon Piel de Sapo': {"Kalori": "36 kcal", "Vitamin": "C", "Serat": "1.0g", "Manfaat": "Sangat menghidrasi."},
    'Mulberry': {"Kalori": "43 kcal", "Vitamin": "C, Besi", "Serat": "1.7g", "Manfaat": "Kesehatan darah."},
    'Nectarine': {"Kalori": "44 kcal", "Vitamin": "A, C", "Serat": "1.7g", "Manfaat": "Kesehatan kulit."},
    'Nut': {"Kalori": "600 kcal", "Vitamin": "E", "Serat": "7.0g", "Manfaat": "Energi tahan lama."},
    'Onion': {"Kalori": "40 kcal", "Vitamin": "C, B6", "Serat": "1.7g", "Manfaat": "Antibakteri alami."},
    'Orange': {"Kalori": "47 kcal", "Vitamin": "C", "Serat": "2.4g", "Manfaat": "Vitamin C harian."}
}