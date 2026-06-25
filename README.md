# Artificial Neural Network pada Arduino Uno
![Platform](https://img.shields.io/badge/platform-Arduino%20Uno-blue)
![Programming Language](https://img.shields.io/badge/C++-blue)
## Ringkasan
Proyek ini merupakan langkah eksperimental untuk menjalankan algoritma Jaringan Syaraf Tiruan pada perangkat mikrokontroler seperti Arduino Uno. Tujuan proyek ini adalah untuk menguji seberapa mungkin perangkat mikrokontroler mampu menjalankan algoritma Jaringan Syaraf Tiruan.

## Rincian
Proyek ini menggunakan struktur data graph komputasional untuk menghitung setiap langkah operasi Jaringan Syaraf Tiruan serta membangun pelacakan diferensiasi untuk mencatat gradien sehingga memudahkan proses backpropagation.
```
(a)----
        \
          (*)------>(c)
        /
(b)----

c = a * b
a.grad = c.grad * b
b.grad = c.grad * a
```
Setiap node diwakili oleh objek Tensor yang berisikan properti penting berupa _value untuk menyimpan nilai hasil komputasi dan _grad untuk menyimpan nilai gradien yang diperoleh selama proses backpropagation. Selain itu, setiap Tensor juga menyimpan referensi ke node-node sebelumnya yang digunakan untuk membentuk graph komputasional. Sehingga ketika operasi matematika dilakukan, sistem tidak hanya menghitung nilai hasil operasi tetapi juga membangun hubungan antar node sehingga seluruh proses perhitungan dapat direpresentasikan sebagai graph terarah.

Sebagai contoh, operasi:

```
c = a * b
```

akan menghasilkan node baru c yang menyimpan nilai hasil perkalian serta referensi ke node a dan b. Saat proses backpropagation dijalankan, gradien akan disebarkan dari node output menuju seluruh node yang terhubung berdasarkan aturan turunan masing-masing operasi.

Implementasi Jaringan Syaraf Tiruan dibangun menggunakan beberapa komponen utama, yaitu layer, neuron, serta algoritma optimisasi berbasis gradient descent. Seluruh proses forward propagation dan backpropagation dilakukan secara manual tanpa menggunakan pustaka machine learning eksternal. Sebagai pengujian awal, model digunakan untuk mempelajari regresi linear yang merupakan salah satu permasalahan klasik dalam Jaringan Syaraf Tiruan. Pengujian ini bertujuan untuk memverifikasi bahwa proses perhitungan gradien dan pembaruan bobot telah berjalan dengan benar pada lingkungan Arduino Uno.

## Batasan
Karena Arduino Uno hanya memiliki 2 KB SRAM, pengelolaan memori menjadi aspek yang sangat penting. Setiap objek Tensor yang dibuat selama pembentukan graph komputasional dikelola dengan hati-hati agar penggunaan memori tetap berada dalam batas yang tersedia dan tidak menyebabkan kegagalan program akibat kehabisan memori.

## Arsitektur
<img width="2048" height="926" alt="nn" src="https://github.com/user-attachments/assets/7c0890dd-f361-455f-96d5-fcb091cfae54" />
Arsitektur yang digunakan pada proyek ini menggunakan 1 hidden layer dengan 2 neuron dan output layer dengan 1 neuron.

## Hasil
### Pengujian dengan 100 Epochs
<img width="817" height="947" alt="Cuplikan layar 2026-06-24 225429" src="https://github.com/user-attachments/assets/15467b42-b908-4f4c-b2e7-61db464a1eff" />

### Pengujian dengan 1000 Epochs
<img width="803" height="926" alt="Cuplikan layar 2026-06-24 221136" src="https://github.com/user-attachments/assets/487738de-a804-436f-a871-ccb7d0994187" />

## Kesimpulan
Membangun sebuah algoritma Jaringan Syaraf Tiruan pada Arduino Uno sangat mungkin dilakukan, namun dengan banyak keterbatasan yang krusial terutama pada batasan memori yang hanya memiliki 2 KB SRAM. 

Meskipun ukuran model yang dapat dijalankan jauh lebih kecil dibandingkan implementasi pada komputer modern, pendekatan ini membuktikan bahwa konsep dasar pembelajaran mesin tetap dapat diterapkan pada mikrokontroler 8-bit. Hasil eksperimen ini juga memberikan gambaran mengenai tantangan utama dalam pengembangan kecerdasan buatan pada perangkat embedded, seperti keterbatasan RAM, kapasitas penyimpanan program, serta efisiensi penggunaan memori selama pembentukan graph komputasional.

## Catatan
Proyek ini tidak ditujukan untuk menggantikan framework pembelajaran mesin modern, melainkan sebagai sarana eksplorasi dan pembelajaran untuk memahami cara kerja internal Jaringan Syaraf Tiruan dari tingkat paling dasar. Dengan membangun seluruh komponen dari awal, mulai dari Tensor, graph komputasional, propagasi maju (forward propagation), hingga propagasi balik (backpropagation), proses pembelajaran menjadi lebih transparan dan mudah dipahami dibandingkan hanya menggunakan pustaka yang sudah jadi.
