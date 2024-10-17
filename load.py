from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import MarkdownReader
import os
from pathlib import Path
from llama_index.core.node_parser import SimpleNodeParser 
from llama_index.core.text_splitter import CodeSplitter 
from llama_index.readers.github import GithubRepositoryReader, GithubClient


parser = MarkdownReader()
file_extractor = {".md": parser}

# 读取指定路径下的所有 Markdown 文件，并保留文件夹结构信息
def read_markdown_files(markdown_files_path):
    print('Reading markdown files...')
    markdown_knowledge = []
    # 遍历指定路径下的所有文件和文件夹
    for root, dirs, files in os.walk(markdown_files_path,topdown=True):
        for file in files:
            if file.endswith('.md'):
                file_path = Path(root) / file
                documents = SimpleDirectoryReader(
                    input_files=[file_path], file_extractor=file_extractor ).load_data() 
                markdown_knowledge += documents
    print('Reading markdown files done!')
    return markdown_knowledge

def parse_node_md(markdown_knowledge):
    # Assuming documents have already been loaded 
    print('Starting parsing to nodes.....')
    # Initialize the parser 
    parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20) 

    # Parse documents into nodes 
    nodes = parser.get_nodes_from_documents(markdown_knowledge)
    print('Parsing to nodes  done!')

    return nodes


def parse_node_code(code_knowledge):
    # Assuming documents have already been loaded 

    text_splitter = CodeSplitter( 
    language="python", chunk_lines=40, chunk_lines_overlap=15, max_chars=1500, 
    ) 
    parser = SimpleNodeParser.from_defaults(text_splitter=text_splitter)
    # Parse documents into nodes 
    nodes = parser.get_nodes_from_documents(code_knowledge)
    return nodes

def read_github(owner, repo, github_token):
    github_reader = GithubRepositoryReader(
    github_client=GithubClient(github_token=github_token, verbose=False),
    owner=owner,
    repo=repo,
    concurrent_requests = 1,
    retries = 5,
    use_parser=False,
    verbose=True,
    filter_directories=(
        ["docs",".github",".msvc"],
        GithubRepositoryReader.FilterType.EXCLUDE,
    ),
    filter_file_extensions=(
        [
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".ico",
            "json",
            ".ipynb",
        ],
        GithubRepositoryReader.FilterType.EXCLUDE,
    ),
)
    documents = github_reader.load_data(branch="master")
    return documents
