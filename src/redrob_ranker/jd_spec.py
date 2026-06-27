"""Structured understanding of the job description.

The challenge is explicitly designed to punish keyword matching. So instead of
treating the JD as a bag of words, we parse it into a *rubric*: what the role
needs, what it explicitly does NOT want, what the ideal profile looks like, and
how to read behavioral signals.

Everything here is derived from reading job_description.txt closely. The raw JD
text is still loaded separately (see `load_jd_text`) and used for the semantic
similarity component, so the two views complement each other:

  - structured rubric  -> precise, explainable scoring of role fit
  - raw JD text        -> catches "hidden gem" evidence in free-text histories
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def load_jd_text(path: str | Path) -> str:
    """Load the raw JD text used for the semantic similarity component."""
    return Path(path).read_text(encoding="utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Skill families. We group skills into families so that a candidate is rewarded
# for demonstrating a *capability*, not for stuffing every synonym. Each family
# carries a weight (how central it is to this role) and a tier:
#   - "must"  : the JD says you absolutely need this
#   - "nice"  : the JD would like it but won't reject you for missing it
# Matching is done against the candidate's skill names AND their free-text role
# descriptions, so "built a recommendation system" counts even with no skill tag.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SkillFamily:
    name: str
    tier: str  # "must" | "nice"
    weight: float
    patterns: tuple[str, ...]  # regex-ish substrings (lowercased, word-aware)


SKILL_FAMILIES: tuple[SkillFamily, ...] = (
    # ---- MUST-HAVES (the "things you absolutely need" section) ----
    SkillFamily(
        "embeddings_retrieval", "must", 1.0,
        ("embedding", "sentence-transformer", "sentence transformer", "sbert",
         "openai embedding", "bge", "e5 model", " e5 ", "retrieval", "rag",
         "semantic search", "dense retrieval", "bi-encoder", "cross-encoder"),
    ),
    SkillFamily(
        "vector_search", "must", 1.0,
        ("vector database", "vector db", "vector search", "pinecone", "weaviate",
         "qdrant", "milvus", "opensearch", "elasticsearch", "faiss", "annoy",
         "hnsw", "hybrid search", "bm25", "lucene", "solr"),
    ),
    SkillFamily(
        "ranking_reco", "must", 1.0,
        ("ranking", "learning to rank", "learning-to-rank", "ltr", "recommendation",
         "recommender", "recsys", "search relevance", "personalization", "matching system"),
    ),
    SkillFamily(
        "ml_eval", "must", 1.0,
        ("ndcg", "mrr", "mean reciprocal", "map@", "mean average precision",
         "a/b test", "ab test", "offline evaluation", "evaluation framework",
         "relevance evaluation", "precision@", "recall@"),
    ),
    SkillFamily(
        "python_eng", "must", 0.7,
        ("python", "pytest", "code quality", "software engineering", "backend"),
    ),
    SkillFamily(
        "nlp_ir", "must", 0.9,
        ("nlp", "natural language", "information retrieval", " ir ", "text mining",
         "language model", " llm", "transformer", "bert", "tokeniz"),
    ),
    SkillFamily(
        "ml_production", "must", 0.85,
        ("production", "deployed", "real users", "at scale", "inference",
         "model serving", "mlops", "ml platform", "shipped"),
    ),
    # ---- NICE-TO-HAVES ("we'd like you to have but won't reject you") ----
    SkillFamily(
        "fine_tuning", "nice", 0.5,
        ("fine-tun", "fine tun", "lora", "qlora", "peft", "instruction tuning",
         "rlhf", "distillation"),
    ),
    SkillFamily(
        "ltr_models", "nice", 0.5,
        ("xgboost", "lightgbm", "gradient boost", "neural ranking", "lambdamart"),
    ),
    SkillFamily(
        "hr_marketplace", "nice", 0.4,
        ("hr-tech", "hrtech", "recruiting", "recruitment", "marketplace",
         "two-sided", "talent", "ats", "hiring platform"),
    ),
    SkillFamily(
        "distributed", "nice", 0.4,
        ("distributed systems", "spark", "kafka", "ray", "kubernetes",
         "large-scale inference", "low latency", "throughput"),
    ),
    SkillFamily(
        "open_source", "nice", 0.35,
        ("open source", "open-source", "github contribution", "maintainer",
         "published", "paper", "arxiv", "talk at"),
    ),
)

MUST_FAMILIES = tuple(f for f in SKILL_FAMILIES if f.tier == "must")
NICE_FAMILIES = tuple(f for f in SKILL_FAMILIES if f.tier == "nice")


# ---------------------------------------------------------------------------
# Title fit. The JD is blunt: an "HR Manager" who lists AI skills is NOT a fit.
# Title is the single strongest anti-keyword-stuffer signal, so we score it
# explicitly. We look at the current title AND the recent trajectory.
# ---------------------------------------------------------------------------

# Strong, on-target engineering identities for this role.
TARGET_TITLE_PATTERNS: tuple[str, ...] = (
    "ai engineer", "ml engineer", "machine learning engineer",
    "applied scientist", "applied ml", "research engineer", "nlp engineer",
    "search engineer", "relevance engineer", "ranking engineer",
    "recommendation", "data scientist", "research scientist",
    "ml scientist", "deep learning", "ai scientist", "staff engineer",
    "software engineer", "backend engineer", "data engineer", "platform engineer",
)

# Weight of how on-target each pattern is (1.0 = bullseye).
TARGET_TITLE_WEIGHT: dict[str, float] = {
    "ai engineer": 1.0, "ml engineer": 1.0, "machine learning engineer": 1.0,
    "applied scientist": 0.95, "applied ml": 0.95,
    "nlp engineer": 0.95, "search engineer": 0.9, "relevance engineer": 0.95,
    "ranking engineer": 0.95, "recommendation": 0.95, "ai scientist": 0.8,
    "ml scientist": 0.85, "deep learning": 0.8, "data scientist": 0.7,
    # research-leaning titles are credited modestly; the JD treats research-only
    # (no production) as a disqualifier, applied separately in scoring.
    "research engineer": 0.55, "research scientist": 0.45,
    "staff engineer": 0.6, "software engineer": 0.55,
    "backend engineer": 0.5, "data engineer": 0.5, "platform engineer": 0.5,
}

# Research-leaning identities. Full credit only with production evidence;
# otherwise the JD's "pure research without production deployment" disqualifier
# applies (see scoring._disqualifiers).
RESEARCH_TITLE_PATTERNS: tuple[str, ...] = (
    "research engineer", "research scientist", "research fellow", "ai research",
    "ml research", "applied research", "phd", "postdoc",
)

# Seniority red flags for a senior founding-team role (the JD wants 5-9 yrs and
# someone who can mentor and own architecture — not an early-career hire).
JUNIOR_TITLE_PATTERNS: tuple[str, ...] = (
    "junior", "jr.", "jr ", "intern", "trainee", "apprentice", "entry level",
    "entry-level", "graduate engineer",
)

# Services / consulting industries (beyond the named-firm list) — the JD's ideal
# is applied ML at *product* companies "not pure services".
SERVICES_INDUSTRY_TOKENS: tuple[str, ...] = (
    "it services", "consulting", "outsourcing", "bpo", "staffing", "system integrator",
    "managed services", "professional services",
)

# Titles that clearly signal a non-fit (the keyword-stuffer trap candidates).
ANTI_TITLE_PATTERNS: tuple[str, ...] = (
    "hr manager", "human resources", "recruiter", "talent acquisition",
    "marketing manager", "marketing", "content writer", "copywriter",
    "graphic designer", "designer", "accountant", "finance", "sales executive",
    "sales manager", "business development", "customer support", "customer success",
    "operations manager", "project manager", "program manager", "civil engineer",
    "mechanical engineer", "electrical engineer", "teacher", "professor",
    "doctor", "nurse", "lawyer", "chef", "driver",
)


# ---------------------------------------------------------------------------
# Career-history evidence. The "hidden gem" clause: a candidate who built a
# recommendation/search/ranking system at a PRODUCT company is a fit even if
# the title and skills are unflashy. We scan free-text role descriptions.
# ---------------------------------------------------------------------------

EVIDENCE_STRONG: tuple[str, ...] = (
    "recommendation system", "recommender", "search system", "ranking system",
    "ranking model", "relevance", "retrieval system", "semantic search",
    "vector search", "personalization", "matching engine", "matching system",
    "embedding", "learning to rank", "search relevance", "candidate matching",
)
EVIDENCE_MEDIUM: tuple[str, ...] = (
    "machine learning model", "ml model", "nlp", "deep learning", "feature pipeline",
    "model in production", "a/b test", "experimentation", "data pipeline",
    "real-time", "streaming", "large-scale", "production ml",
)


# ---------------------------------------------------------------------------
# Disqualifiers (the "things we explicitly do NOT want" section). Each returns
# a soft multiplicative penalty rather than a hard cut, because the JD says it
# "will probably not move forward" — i.e. strong negative, not absolute.
# ---------------------------------------------------------------------------

# Career entirely at consulting / services firms -> bad fit per JD.
CONSULTING_FIRMS: tuple[str, ...] = (
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mindtree", "ltimindtree", "lti",
    "mphasis", "deloitte", "ibm global services", "dxc", "hexaware",
    "larsen & toubro infotech", "persistent systems",
)

# Computer vision / speech / robotics primary WITHOUT NLP/IR -> re-learning here.
CV_SPEECH_ROBOTICS: tuple[str, ...] = (
    "computer vision", "image classification", "object detection", "segmentation",
    "speech recognition", "tts", "text-to-speech", "asr", "robotics", "slam",
    "autonomous", "lidar", "point cloud", "pose estimation", "ocr",
)

# Recent-LangChain-only signal (frame: <12mo of LangChain calling OpenAI, no depth).
FRAMEWORK_HYPE: tuple[str, ...] = (
    "langchain", "llamaindex", "llama-index", "autogen", "crewai", "prompt engineering",
)

NLP_IR_ANCHORS: tuple[str, ...] = (
    "nlp", "natural language", "information retrieval", "retrieval", "ranking",
    "search", "recommendation", "embedding", "text", "language model", "llm",
)


# ---------------------------------------------------------------------------
# Location. JD: Pune/Noida preferred; Tier-1 Indian cities welcome; outside
# India is case-by-case with no visa sponsorship.
# ---------------------------------------------------------------------------

PREFERRED_CITIES: tuple[str, ...] = ("pune", "noida")
TIER1_INDIAN_CITIES: tuple[str, ...] = (
    "bangalore", "bengaluru", "hyderabad", "mumbai", "delhi", "new delhi",
    "gurgaon", "gurugram", "noida", "pune", "chennai", "ncr",
)
INDIA_TOKENS: tuple[str, ...] = ("india",)


@dataclass(frozen=True)
class RoleSpec:
    """The full structured rubric for the released role."""

    title: str = "Senior AI Engineer — Founding Team"
    exp_min: float = 5.0
    exp_max: float = 9.0
    exp_ideal_low: float = 6.0
    exp_ideal_high: float = 8.0
    # Notice period: <=30 days ideal, buyable up to 30, 30+ raises the bar.
    notice_ideal_days: int = 30
    notice_max_soft_days: int = 90
    jd_text: str = ""
    skill_families: tuple[SkillFamily, ...] = field(default=SKILL_FAMILIES)

    @classmethod
    def load(cls, jd_path: str | Path) -> "RoleSpec":
        return cls(jd_text=load_jd_text(jd_path))


_WORD_RE = re.compile(r"[a-z0-9&+./-]+")


def normalize(text: str) -> str:
    """Lowercase + collapse whitespace; keep a few tech-relevant symbols."""
    return " ".join(_WORD_RE.findall((text or "").lower()))
