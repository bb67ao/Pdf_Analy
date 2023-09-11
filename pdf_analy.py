import PyPDF2
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import torch.nn.functional as F
import faiss

class Pdf_Analy():
    def __init__(self, file_path) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.pdf_pages = 0
        self.chapter_contents = []
        self.read_pdf(file_path)

    def read_pdf(self, file_path):
        with open(file_path, "rb") as file:
            pdf = PyPDF2.PdfReader(file)
            self.pdf_pages = len(pdf.pages)
            chapter_contents = []
            for page in pdf.pages:
                text = page.extract_text()
                chapter_contents.append(text)
        self.chapter_contents = chapter_contents

    def write2txt(self, write_txt_path):
        if(write_txt_path != None):
            for contents in self.chapter_contents:
                save_to_txt_file(contents, write_txt_path)
        # return chapter_contents



class Summary_Model():
    def __init__(self, model_name='bart', inputs_max_length=1024, summary_max_length=100) -> None:
        self.model_name = model_name
        self.inputs_max_length = inputs_max_length
        self.summary_max_length = summary_max_length
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()
        self.model.eval()
        # self.d_model = self.get_model().config.d_model

    def get_tokenizer(self):
        return BartTokenizer.from_pretrained(self.model_name)

    def get_str_encoder(self, str):
        return BartTokenizer.from_pretrained(self.model_name).encode(str, add_special_tokens=True)

    def get_emb_decode(self, emb):
        return BartTokenizer.from_pretrained(self.model_name).decode(emb, skip_special_tokens=True)

    def get_model(self):
        return BartForConditionalGeneration.from_pretrained(self.model_name)

    def summarize_content(self, contents, index):
        inputs = self.tokenizer(contents[index],
                                truncation=True, max_length=self.inputs_max_length, return_tensors='pt')
        summary_ids = self.model.generate(
            inputs['input_ids'], num_beams=4, max_length=self.summary_max_length, early_stopping=True)

        summary = self.get_emb_decode(summary_ids[0])
        return summary, summary_ids


class Summary2Faiss():
    def __init__(self, dmodel) -> None:
        self.index = faiss.IndexFlatL2(dmodel)

    def add2faiss(self, summary_ids):
        self.index.add(summary_ids)

    def write2faiss(self, write_path):
        faiss.write_index(self.index, write_path)

    def read_faiss(self, read_path):
        self.index = faiss.read_index(read_path)

    def get_topk(self, query, k):
        return self.index.search(query, k)

    def get_vec(self, query_index):
        return self.index.reconstruct(query_index)

def save_to_txt_file(data, file_path):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(data)

def build_index_file(pdf_analy, summary_model, summary2faiss, faiss_path):
    contents = pdf_analy.chapter_contents
    pdf_pages = pdf_analy.pdf_pages
    for index in range(pdf_pages):
        summary, summary_ids = summary_model.summarize_content(
            contents, index)
        summary_ids = summary_ids[:,1:]
        padded_tensor = F.pad(
            summary_ids, (0, d_summary_max_length - summary_ids.shape[1]))
        summary2faiss.add2faiss(padded_tensor)
        save_to_txt_file(str(summary_ids) + '\n' + str(summary) + '\n','summary_ids.txt')
        # print(summary)
    summary2faiss.write2faiss(faiss_path)


def get_topk_relate_content(src_content, summary_model, summary2faiss, faiss_path, k):
    summary2faiss.read_faiss(faiss_path)

    src_vector = torch.tensor([summary_model.get_str_encoder(src_content)])
    src_vector = F.pad(
        src_vector, (0, d_summary_max_length - src_vector.shape[1]))

    # print(src_vector)
    dis, query_index = summary2faiss.get_topk(src_vector, k)

    for i in range(k):
        index_id = int(query_index[0][i])
        vector = summary2faiss.get_vec(index_id)
        res = summary_model.get_emb_decode(vector)
        print(f"Res {i+1} : {res}")


if __name__ == "__main__":
    pdf_path = "./test.pdf"
    txt_path = "./txt_file.txt"
    faiss_path = "./faiss.index"
    model = "bart"
    d_inputs_max_length = 1024
    d_summary_max_length = 100
    k = 5

    pdf_analy = Pdf_Analy(pdf_path)

    summary_model = Summary_Model(
        model, d_inputs_max_length, d_summary_max_length)

    summary2faiss = Summary2Faiss(d_summary_max_length)

    # build_index_file(pdf_analy,summary_model,summary2faiss,faiss_path)

    src_content = "We propose lightweight self-attentive network (LSAN), a novel solution to memory-efficient sequential recommendation. LSAN aggressively replaces the item embedding matrix with base embedding matrices, each of which contains substantially fewer embedding vectors (i.e., base embeddings) than the total amount of items. Results for each user are purely conditioned on the static user-itemaffinity instead of her/his interest dynamics."

    get_topk_relate_content(src_content,
                            summary_model, summary2faiss, faiss_path, k)

