# Artificial Neural Network on Arduino Uno

![Platform](https://img.shields.io/badge/platform-Arduino%20Uno-00979D?logo=arduino&logoColor=white)
![Language](https://img.shields.io/badge/language-C++-00599C?logo=cplusplus&logoColor=white)
![Status](https://img.shields.io/badge/status-experimental-orange)
![Memory](https://img.shields.io/badge/SRAM-2KB-critical)

> Membangun autograd engine dan jaringan syaraf tiruan dari nol — langsung di atas mikrokontroler 8-bit dengan 2 KB SRAM.

---

## Ringkasan

Proyek ini merupakan eksperimen untuk menjalankan jaringan syaraf tiruan (JST) sepenuhnya di atas Arduino Uno — tanpa library machine learning eksternal, tanpa floating-point unit, dan dengan memori yang sangat terbatas.

Tujuannya bukan efisiensi, melainkan pemahaman: seberapa jauh konsep dasar deep learning dapat diturunkan ke perangkat keras paling sederhana sekalipun?

---

## Cara Kerja

Inti dari proyek ini adalah **computational graph** — struktur data yang merepresentasikan seluruh alur komputasi JST sebagai graph terarah.

```
(a)----
        \
          (*)------>(c)
        /
(b)----

c = a * b
∂c/∂a = b  →  a.grad += c.grad * b
∂c/∂b = a  →  b.grad += c.grad * a
```

Setiap operasi menghasilkan node baru yang menyimpan nilainya sekaligus referensi ke node-node sebelumnya. Ketika backpropagation dijalankan, gradien mengalir mundur melalui seluruh graph mengikuti aturan chain rule — sepenuhnya secara otomatis.

### Komponen Utama

Setiap node direpresentasikan oleh objek **`Tensor`** dengan dua properti inti:

| Properti | Fungsi |
|---|---|
| `_value` | Menyimpan hasil komputasi forward pass |
| `_grad` | Menyimpan gradien yang diperoleh saat backpropagation |

JST dibangun dari tiga lapisan abstraksi: **Tensor → Neuron → Layer**, dengan optimisasi berbasis gradient descent yang diimplementasikan seluruhnya secara manual.

---

## Arsitektur Model

<img width="2048" height="926" alt="Arsitektur JST" src="https://github.com/user-attachments/assets/7c0890dd-f361-455f-96d5-fcb091cfae54" />

Model yang digunakan pada pengujian ini memiliki arsitektur minimal:
- **Input layer** — menerima fitur input
- **Hidden layer** — 2 neuron dengan fungsi aktivasi
- **Output layer** — 1 neuron untuk prediksi regresi

---

## Hasil Pengujian

Pengujian dilakukan pada permasalahan **regresi linear** untuk memverifikasi bahwa alur forward pass, gradient computation, dan weight update berjalan benar di lingkungan Arduino Uno.

### 100 Epochs
<img width="817" height="947" alt="Hasil 100 Epochs" src="https://github.com/user-attachments/assets/15467b42-b908-4f4c-b2e7-61db464a1eff" />

### 1000 Epochs
<img width="803" height="926" alt="Hasil 1000 Epochs" src="https://github.com/user-attachments/assets/487738de-a804-436f-a871-ccb7d0994187" />

Model menunjukkan konvergensi yang konsisten seiring bertambahnya epoch, membuktikan bahwa mekanisme gradient descent berfungsi dengan benar meski berjalan di atas prosesor 16 MHz dengan memori 2 KB.

---

## Batasan Memori

Arduino Uno hanya memiliki **2 KB SRAM** — kira-kira cukup untuk menyimpan 500 bilangan bulat 32-bit. Ini menjadi constraint paling krusial dalam proyek ini.

Setiap `Tensor` dalam graph membutuhkan alokasi heap. Tanpa pengelolaan memori yang cermat, program akan crash akibat heap corruption atau stack overflow bahkan sebelum forward pass selesai. Seluruh siklus hidup objek — alokasi, penggunaan, dan dealokasi — dikelola secara eksplisit untuk memastikan footprint memori tetap dalam batas yang tersedia.

---

## Kesimpulan

Menjalankan JST di Arduino Uno **sangat mungkin dilakukan**, namun dengan batasan yang tidak bisa diabaikan. Ukuran model yang dapat berjalan jauh lebih kecil dari standar modern, dan setiap byte memori harus diperhitungkan.

Meski begitu, eksperimen ini membuktikan bahwa konsep inti machine learning — autograd, backpropagation, gradient descent — dapat diturunkan ke mikrokontroler 8-bit tanpa bantuan framework apapun. Lebih dari sekadar hasil teknisnya, proses membangun setiap komponen dari nol memberikan pemahaman yang jauh lebih dalam tentang cara kerja JST dibandingkan sekadar menggunakan PyTorch atau TensorFlow.

---

## Catatan

Proyek ini bukan alternatif untuk framework ML modern. Ini adalah latihan pemahaman untuk membangun Tensor, computational graph, forward pass, hingga backpropagation dari tingkat paling dasar, di atas hardware yang paling terbatas yang tersedia, justru untuk memaksa pemahaman yang benar-benar fundamental.
