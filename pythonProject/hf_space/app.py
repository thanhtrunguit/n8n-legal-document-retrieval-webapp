from sentence_transformers import CrossEncoder
import py_vncorenlp
import os

# Initialize vncorenlp (point to the JAR file in your folder)
vncorenlp_path = "hf_space/vncorenlp"
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_path)

# Function to preprocess text with Vietnamese word segmentation
def preprocess_text(text):
    if not text:
        return text
    segmented_text = rdrsegmenter.word_segment(text)
    # Join tokenized sentences into a single string
    return " ".join([" ".join(sentence) for sentence in segmented_text])

text = 'Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây.'
print(preprocess_text(text))