from typing import List, Optional
from pydantic import BaseModel

class ImageInfo(BaseModel):
    product_name: str
    image_path: str
    category: str

class RecommendationResponse(BaseModel):
    success: bool
    system_message: str
    recommendations_text: str
    images: Optional[List[ImageInfo]] = []
    product_count: Optional[int] = 0
