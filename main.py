from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings

from load import read_markdown_files,parse_node_md,read_github
from llama_index.core import VectorStoreIndex,StorageContext, load_index_from_storage 
import openai
from llama_index.llms.openai import OpenAI
from tqdm import tqdm
from utils import count_lines_in_jsonl, read_jsonl, write_jsonl, write_csv, calculate_avg
from score import get_score


markdown_files_path = './data/markdowns/zh-CN/source'
persist_dir = "./db"
Settings.embed_model = HuggingFaceBgeEmbeddings(model_name="../bge-m3")
openai.api_key = 'sk-xfovpV3O7IwdmDDJBb05Ff03E5014c14Ab5e935715Fe90D3'
openai.base_url = 'https://api.gptapi.us/v1'
openai_model = 'gpt-4o-mini'
github_token = ''

store = 1
test_path = './test/test1.jsonl' 
val_path = './test/val.jsonl'
test_out_path = './result/answer_test.jsonl'
val_out_path = './result/answer_val.jsonl'
score_out_path = './result/score.csv'

temperature = 0.1
top_k = 50
similarity_cutoff = 0.5
system_prompt = '你是一个蚂蚁集团的TuGraph数据库专家，\
                            擅长使用与TuGraph数据库相关的知识来回答用户的问题，\
                            针对用户的提问，你会得到一些文本材料辅助回答，如果某些辅助文本与提问关联性不强，则可以忽略，\
                            结合有用的部分以及你的知识，回答用户的提问。如果可以直接给出答案,则只回答最关键的部分,做到尽可能简洁。\
                            注意：问题中的数据库一律指代TuGraph,\
                            请仿照下面的样例答案格式进行后续的回答,给出答案.\
                            样例问题1：RPC 及 HA 服务中，verbose 参数的设置有几个级别？, 样例答案:  三个级别（0，1，2)。 \
                            样例问题2: 如果成功修改一个用户的描述，应返回什么状态码？样例答案：200 '



llm = OpenAI(model=openai_model, temperature=temperature, system_prompt=system_prompt)



if store:
    documents = []
    md_knowledges = read_markdown_files(markdown_files_path)
    repo_documents = read_github('Sun-HR02','tugraph-db')
    documents += md_knowledges
    documents += repo_documents
    # nodes = parse_node_md(knowledges) 
    print('start embedding....')
    index = VectorStoreIndex(md_knowledges)
    print('embedding done!')
    # persist
    index.storage_context.persist(persist_dir=persist_dir)  
else: 
    # load
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir) 
    index = load_index_from_storage(storage_context)

# Assuming 'index' is your constructed index object 
query_engine = index.as_query_engine(llm=llm, similarity_cutoff=similarity_cutoff, top_k=top_k) 

print('正在对 val.jsonl 进行生成检索.....')
answers_val = []
with tqdm(total=count_lines_in_jsonl(val_path)) as pbar:
    for obj in read_jsonl(val_path):
        query = obj.get('input_field')
        answer = query_engine.query(query) # 可以用source_nodes获得相关chunk
        answers_val.append(dict(id=obj.get('id'), output_field = answer.response))
        # answers_val.append(dict(id=obj.get('id'), output_field = query_engine.query(query)))
        pbar.update(1)
write_jsonl(answers_val, val_out_path )
print('val.jsonl 已生成答案！\n \n')

print('正在计算分数.....')
score_output = get_score(val_path,val_out_path)
write_csv(score_output,score_out_path)
print('分数平均为{}! \n \n'.format(calculate_avg(score_output)))