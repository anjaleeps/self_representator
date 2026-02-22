from pydantic import BaseModel
from typing import Optional, List


class UserDetails(BaseModel):
    name: str
    company: Optional[str] = None
    bio: Optional[str] = None
    public_repos: int
    followers: int
    following: int


class RepoInfo(BaseModel):
    id: int
    name: str
    full_name: str
    private: bool
    fork: bool
    description: Optional[str] = None
    url: str
    language: Optional[str] = None


class RepoData(BaseModel):
    id: int
    name: str
    full_name: str
    description: Optional[str] = None
    url: str
    topics: list
    languages: dict
    readme: Optional[str] = None


class ArticleInfo(BaseModel):
    id: str
    title: str
    subtitle: str
    tags: list[str]
    topics: list[str]
    url: str
    