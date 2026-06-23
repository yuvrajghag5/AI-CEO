from typing import List, Literal, Annotated
from pydantic import BaseModel, Field
 
PriorityLevel = Literal["High", "Medium", "Low"]
 
 
MIN_DESCRIPTION_CHARS = 180  # roughly 30-40 words, enforced by outlines via the JSON schema
MIN_ACTION_ITEM_CHARS = 150  # each action plan item needs real specificity, not one vague sentence
 
# A string type with a per-item minimum length, for use inside a List[...]
# (List[str] alone only constrains how many items are in the list, not
# how long each individual item's content is).
ActionItem = Annotated[str, Field(min_length=MIN_ACTION_ITEM_CHARS)]
 
 
class Opportunity(BaseModel):
    opportunity: str = Field(description="Short title of the opportunity")
    business_impact: str = Field(
        min_length=MIN_DESCRIPTION_CHARS,
        description="A detailed 30-40 word explanation of the business impact — NOT a severity word like High/Medium/Low",
    )
    supporting_evidence: List[str] = Field(min_length=1, max_length=3)
 
 
class Risk(BaseModel):
    risk: str = Field(description="Short title of the risk")
    why_it_matters: str = Field(
        min_length=MIN_DESCRIPTION_CHARS,
        description="A detailed 30-40 word explanation of why this risk matters — NOT a severity word like High/Medium/Low",
    )
    supporting_evidence: List[str] = Field(min_length=1, max_length=3)
 
 
class CompetitorActivityItem(BaseModel):
    competitor_activity: str = Field(description="Short title of the competitor activity")
    strategic_meaning: str = Field(
        min_length=MIN_DESCRIPTION_CHARS,
        description="A detailed 30-40 word explanation of the strategic meaning — NOT a severity word like High/Medium/Low",
    )
    supporting_evidence: List[str] = Field(min_length=1, max_length=3)
 
 
class Trend(BaseModel):
    trend: str = Field(description="Short title of the trend")
    strategic_meaning: str = Field(
        min_length=MIN_DESCRIPTION_CHARS,
        description="A detailed 30-40 word explanation of the strategic meaning — NOT a severity word like High/Medium/Low",
    )
    supporting_evidence: List[str] = Field(min_length=1, max_length=3)
 
 
class Recommendation(BaseModel):
    recommendation: str = Field(description="Short title of the recommendation")
    priority: PriorityLevel
    supporting_evidence: List[str] = Field(min_length=1, max_length=3)
    expected_impact: str = Field(
        min_length=MIN_DESCRIPTION_CHARS,
        description="A detailed 30-40 word explanation of the expected impact",
    )
    risk_level: PriorityLevel
 
 
class CEOBriefing(BaseModel):
    executive_summary: str = Field(
        min_length=350,
        description=(
            "A substantive narrative covering what is actually happening with "
            "NVIDIA (the real events/developments from the evidence), why it "
            "matters for the business, and the main strategic message. "
            "Do NOT write meta-commentary like 'I will focus on three "
            "opportunities (S3, S4)' — describe the actual content and "
            "implications, not which sections you are about to cover."
        ),
    )
    key_opportunities: List[Opportunity] = Field(min_length=3, max_length=3)
    key_risks: List[Risk] = Field(min_length=3, max_length=3)
    competitor_activity: List[CompetitorActivityItem] = Field(min_length=2, max_length=2)
    emerging_trends: List[Trend] = Field(min_length=2, max_length=2)
    strategic_recommendations: List[Recommendation] = Field(min_length=3, max_length=3)
    ceo_action_plan: List[ActionItem] = Field(
        min_length=3,
        max_length=3,
        description=(
            "Each item must be a specific, concrete action (who does what, "
            "referencing the relevant evidence/recommendation) — not a vague "
            "one-line statement like 'Allocate resources to R&D.'"
        ),
    )