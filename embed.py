from openai import OpenAI
from FlagEmbedding import BGEM3FlagModel


# def embed(content, options):
#     base_url = options['gpt-baseurl']
#     api_key = options['gpt-apikey']
#     model = options['embedding-model']
#     client = OpenAI(base_url=base_url,
#         api_key=api_key)
#     response = client.embeddings.create(input=content, model=model).data[0].embedding
#     return response

def embed(content): #using bge
    model = BGEM3FlagModel('../bge-m3', use_fp16=True) # BAAI/bge-m3

    response = model.encode(content, max_length = 8192)['dense_vecs']

    return response