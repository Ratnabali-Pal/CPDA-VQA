import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(image_path, title=None):
    img = mpimg.imread(image_path)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()
