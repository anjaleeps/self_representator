import logging
import os
import requests
from requests import RequestException
from typing import List
from pathlib import Path
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import  MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode
from llama_index.readers.file import MarkdownReader, PDFReader
from models import UserDetails, RepoInfo, RepoData, ArticleInfo


load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class Parser(ABC):

    @abstractmethod
    def parse(self, documents: List) -> List[BaseNode]:
        pass

class DataLoader(ABC):

    @abstractmethod
    def load(self) -> List[BaseNode]:
        pass


################################################## GitHub ##################################################

class GitHubClient:

    def __init__(self, username: str):
        self._username = username
        self._github_url = os.getenv("GITHUB_API_URL")
        self._headers = self._make_headers()

    def _get_json(self, url: str, desc: str):
        try:
            response = requests.get(url, headers=self._headers, timeout=15)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error("GitHub %s failed: %s", desc, e)
            raise

    def get_user_details(self) -> UserDetails:
        url = self._github_url + f"/users/{self._username}"
        logger.info("Fetching user details for %s", self._username)
        result = self._get_json(url, "user details")
        if not result:
            return ValueError("No user details found")
        return UserDetails(**result)
    
    def list_all_public_repos(self) -> List[RepoInfo]:
        has_next_page = True
        results = []
        url = self._github_url + f"/users/{self._username}/repos"
        logger.info("Listing public repos for %s", self._username)

        while has_next_page:
            try:
                response = requests.get(url, headers=self._headers, timeout=15)
                response.raise_for_status()
            except RequestException as e:
                logger.error("GitHub list repos failed: %s", e)
                break

            next_page_url = response.links.get("next", {}).get("url", None)
            has_next_page = next_page_url is not None
            url = next_page_url

            data = response.json()
            if data:
                results.extend(data)

        return [RepoInfo(**repo) for repo in results]
    
    def get_repo_topics(self, repo_name: str) -> List[str]:
        url = self._github_url + f"/repos/{self._username}/{repo_name}/topics"
        logger.debug("Fetching topics for repo %s", repo_name)
        result = self._get_json(url, "repo topics")
        return result.get("names", [])
    
    def get_repo_languages(self, repo_name: str) -> List[str: int]:
        url = self._github_url + f"/repos/{self._username}/{repo_name}/languages"
        logger.debug("Fetching languages for repo %s", repo_name)
        result = self._get_json(url, "repo languages")
        return result
    
    def get_repo_readme(self, repo_name: str) -> str:
        url = self._github_url + f"/repos/{self._username}/{repo_name}/readme"
        logger.debug("Fetching README for repo %s", repo_name)
        result = self._get_json(url, "repo readme")
        download_url = result.get("download_url", None)
        if not download_url:
            return None
        
        # Read markdown file
        try:
            response = requests.get(download_url, timeout=15)
            response.raise_for_status()
            return response.text
        except RequestException as e:
            logger.error("GitHub README download failed for %s: %s", repo_name, e)
            return None
    
    def get_all_repo_details(self):
        details = []
        repos = self.list_all_public_repos()
        total_repos = len(repos)
        non_forks = [r for r in repos if not r.fork]
        logger.info("Discovered %d repos (%d non-forks)", total_repos, len(non_forks))
        for repo in non_forks:
            topics = self.get_repo_topics(repo.name)
            languages = self.get_repo_languages(repo.name)
            readme = self.get_repo_readme(repo.name)

            repo_data = RepoData(
                id=repo.id,
                name=repo.name,
                full_name=repo.full_name,
                url=repo.url,
                description=repo.description,
                topics=topics,
                languages=languages,
                readme=readme
            )
            details.append(repo_data)

        logger.info("Collected detailed data for %d repos", len(details))
        return details

    def _make_headers(self):
        headers = {
            "Authorization": f"Bearer {os.getenv("GITHUB_API_TOKEN")}",
            "Accept": "application/vnd.github+json",
            "User-Agent": self._username
        }
        return headers
    

class GitHubParser(Parser):

    def __init__(self):
        super().__init__()
        self.parser = MarkdownNodeParser.from_defaults()

    def parse(self, documents: List[Document]) -> TextNode:
        return self.parser.get_nodes_from_documents(documents)

    def repo_data_to_doc(self, repo_data: RepoData) -> Document:
        text = f"""## Repository description
        {repo_data.description if repo_data.description else "None"}
        ## Repository readme
        {repo_data.readme if repo_data.readme else "None"}
        """
        
        return Document(
            text=text,
            metadata = {
                "id": repo_data.id,
                "name": repo_data.name,
                "full_name": repo_data.full_name,
                "url": repo_data.url,
                "topics": ", ".join(repo_data.topics),
                "languages": ", ".join(f"{k}: {v}" for k, v in repo_data.languages.items()),
            }
        )
    
    def user_details_to_doc(self, user_details: UserDetails) -> Document:
        text = f"""## Name
        {user_details.name}
        ## Company
        {user_details.company if user_details.company else "Not Mentioned"}
        ## Bio
        {user_details.bio if user_details.bio else "None"}
        """

        return Document(
            text=text,
            metadata={
                "username": user_details.name,
                "following": user_details.following,
                "followers": user_details.followers
            }
        )
    

class GitHubDataLoader(DataLoader):

    def __init__(self):
        self.client = GitHubClient(os.getenv("GITHUB_USERNAME"))
        self.parser = GitHubParser()

    def load(self):
        user_details = self.client.get_user_details()
        repo_list = self.client.get_all_repo_details()
        logger.info("Building documents for %d repos", len(repo_list))
        
        user_doc = self.parser.user_details_to_doc(user_details)
        repo_docs = [self.parser.repo_data_to_doc(repo) for repo in repo_list]
        nodes = self.parser.parse([*repo_docs, user_doc])
        logger.info("Parsed %d nodes from GitHub", len(nodes))
        
        return nodes
    
    
################################## Medium ############################

class MediumClient:

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._medium_api_url = os.getenv("MEDIUM_API_URL")
        self._headers = self._make_headers()

    def _get_json(self, url: str, desc: str):
        try:
            response = requests.get(url, headers=self._headers, timeout=15)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error("Medium %s failed: %s", desc, e)
            raise

    def list_user_articles(self) -> List[str]:
        url = self._medium_api_url + f"/user/{self.user_id}/articles"
        logger.debug("Listing all user articles for user %s", self.user_id)
        result = self._get_json(url, "list articles")
        return result.get("associated_articles", [])
    
    def get_article_info(self, article_id: str) -> str:
        url = self._medium_api_url + f"/article/{article_id}"
        logger.debug("Retrieving info for article %s", article_id)
        result = self._get_json(url, "article info")
        return ArticleInfo(**result)
    
    def get_article_markdown(self, article_id: str) -> str:
        url = self._medium_api_url + f"/article/{article_id}/markdown"
        logger.debug("Retrieving markdown for article %s", article_id)
        result = self._get_json(url, "article markdown")
        return result.get("markdown", None)
    
    def _make_headers(self):
        return {
            "x-rapidapi-key": os.getenv("RAPID_API_KEY")
        }
    
class MediumParser(Parser):

    def __init__(self):
        super().__init__()
        self.parser = MarkdownNodeParser.from_defaults()

    def parse(self, documents: List[Document]) -> List[TextNode]:
        return self.parser.get_nodes_from_documents(documents)

    def article_to_doc(self, article_info: ArticleInfo, markdown: str) -> Document:
        return Document(
            text=markdown,
            metadata={
                "id": article_info.id,
                "title": article_info.title,
                "subtitle": article_info.subtitle,
                "url": article_info.url,
                "tags": ", ".join(article_info.tags),
                "topics": ", ".join(article_info.topics)
            }
        )
    
    
class MediumDataLoader(DataLoader):

    def __init__(self):
        self.client = MediumClient(os.getenv("MEDIUM_USER_ID"))
        self.parser = MediumParser()

    def load(self):
        articles_ids = self.client.list_user_articles()
        logger.info("Building documents for %d articles", len(articles_ids))

        docs = []
        for article_id in articles_ids:
            info = self.client.get_article_info(article_id)
            markdown = self.client.get_article_markdown(article_id)
            docs.append(self.parser.article_to_doc(info, markdown))

        nodes = self.parser.parse(docs)
        logger.info("Parsed %d nodes from Medium", len(nodes))
        return nodes
    

################################ Local Docs ################################

class LocalDocumentLoader(DataLoader):

    def __init__(self):
        self.markdown_parser = MarkdownNodeParser.from_defaults()
        self.sentence_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)

    def extract_file_metadata(self, file_path: str) -> dict:
        file_name = Path(file_path).name
        title, ext = file_name.split(".")
        return {
            "file_name": file_name,
            "title": title,
            "ext": ext
        }

    def load(self):
        file_extractor = {
            ".md": MarkdownReader(),
            ".pdf": PDFReader()
        }
        docs = SimpleDirectoryReader(
            input_dir=os.getenv("LOCAL_DOC_FOLDER_PATH"),
            required_exts=[".md", ".pdf", ".txt"],
            file_extractor=file_extractor,
            file_metadata=self.extract_file_metadata,
            recursive=True
        ).load_data()

        md_docs = [d for d in docs if d.metadata["ext"] == "md"]
        other_docs = [d for d in docs if d.metadata["ext"] != "md"]

        logger.info(f"Loaded {len(md_docs)} Markdown documents and {len(other_docs)} other documents from local documents")

        md_nodes = self.markdown_parser.get_nodes_from_documents(md_docs)
        other_nodes = self.sentence_splitter.get_nodes_from_documents(other_docs)
        all_nodes = [*md_nodes, *other_nodes]
        logger.info(f"Parsed {len(all_nodes)} nodes from Local Documents")
        return all_nodes
