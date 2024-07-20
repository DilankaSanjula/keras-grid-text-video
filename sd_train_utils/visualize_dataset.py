import matplotlib.pyplot as plt
from textwrap import wrap
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer

tokenizer = SimpleTokenizer()

def save_sample_batch_images(sample_batch, save_path):
    plt.figure(figsize=(15, 5))  # Adjust the figure size as needed

    for i in range(3):
        ax = plt.subplot(1, 4, i + 1)
        plt.imshow((sample_batch["images"][i] + 1) / 2)

        text = tokenizer.decode(sample_batch["tokens"][i].numpy().squeeze())
        text = text.replace("<|startoftext|>", "")
        text = text.replace("<|endoftext|>", "")
        text = "\n".join(wrap(text, 12))
        plt.title(text, fontsize=15)

        plt.axis("off")

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # Close the plot to free up memory

