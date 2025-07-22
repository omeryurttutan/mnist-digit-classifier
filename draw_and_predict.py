# draw_and_predict.py
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Modeli yükle
model = tf.keras.models.load_model("mnist_model.h5")

# Tkinter penceresi
WIDTH, HEIGHT = 280, 280
WHITE = (255, 255, 255)

class App:
    def __init__(self, root):
        self.clear_button = tk.Button(root, text="Sil", command=self.clear_canvas)
        self.clear_button.pack()
        self.root = root
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg='white')
        self.canvas.pack()
        self.button = tk.Button(root, text="Tahmin Et", command=self.predict_digit)
        self.button.pack()
        self.label = tk.Label(root, text="Çiz ve 'Tahmin Et' tıkla")
        self.label.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.image = Image.new("L", (WIDTH, HEIGHT), 255)
        self.draw_image = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8  # fırça kalınlığı
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
        self.draw_image.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")  # Ekrandaki çizimi sil
        self.image = Image.new("L", (WIDTH, HEIGHT), 255)  # Boş resim oluştur
        self.draw_image = ImageDraw.Draw(self.image)  # Yeni çizim objesi
        self.label.config(text="Çiz ve 'Tahmin Et' tıkla")  # Tahmin yazısını sıfırla

    def predict_digit(self):
        from PIL import ImageChops
        import matplotlib.pyplot as plt

        # 1. Orijinal çizimi ters çevir
        img = self.image.copy()
        img = ImageOps.invert(img)

        # 2. Rakamın etrafındaki boşluğu kırp
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

        # 3. Biraz boşluk bırak: padding (kenarlardan %20 kadar)
        pad_percent = 0.2
        w, h = img.size
        pad_w = int(w * pad_percent)
        pad_h = int(h * pad_percent)

        padded_img = Image.new('L', (w + 2 * pad_w, h + 2 * pad_h), 0)
        padded_img.paste(img, (pad_w, pad_h))

        # 4. Kare yap ve ortala
        max_side = max(padded_img.size)
        square_img = Image.new('L', (max_side, max_side), 0)
        offset = ((max_side - padded_img.size[0]) // 2, (max_side - padded_img.size[1]) // 2)
        square_img.paste(padded_img, offset)

        # 5. 28x28'e küçült
        img_resized = square_img.resize((28, 28))

        # 6. Göster (debug amaçlı)
        plt.imshow(img_resized, cmap='gray')
        plt.title("Ortalanmış + Kenarlı Görsel")
        plt.axis('off')
        plt.show()

        # 7. Tahmin
        img_array = np.array(img_resized).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28)
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)

        self.label.config(text=f"Tahmin: {digit}")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Rakam Tanıyıcı")
    app = App(root)
    root.mainloop()
