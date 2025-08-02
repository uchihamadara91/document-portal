from pydantic import BaseModel, Field, RootModel
from typing import Optional, List, Dict, Any, Union

# For output Parser of Data Analysis
class Metadata(BaseModel):
    Summary: List[str] = Field(default_factory=list, description="Summary of the document")
    Title: str
    Author: str
    DateCreated: str   
    LastModifiedDate: str
    Publisher: str
    Language: str
    PageCount: Union[int, str]  # Can be "Not Available"
    SentimentTone: str

# For Output Parser of Data Compare
class ChangeFormat(BaseModel):
    Page: str
    changes: str


class SummaryResponse(RootModel[list[ChangeFormat]]):
    pass