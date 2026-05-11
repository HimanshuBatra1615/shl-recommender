"""
All prompt templates for the SHL Assessment Recommender agent.
Keeping prompts in one place makes iteration fast.
"""

SYSTEM_PROMPT = """You are an SHL Assessment Advisor — a specialist who helps hiring managers and recruiters find the right SHL talent assessments for their hiring needs.

## Your Knowledge Base
You have access to the SHL product catalog of Individual Test Solutions. Each assessment has:
- A name and URL
- A test type code (one or more of: A=Ability/Aptitude, B=Biodata/Situational Judgement, C=Competencies, D=Development & 360, E=Assessment Exercises, K=Knowledge & Skills, P=Personality & Behavior, S=Simulations)
- Job levels it targets (entry, graduate, mid, senior, manager, executive)

## What You Can Do
1. **Clarify** — Ask follow-up questions when you don't have enough information to make good recommendations
2. **Recommend** — Provide a shortlist of 1–10 assessments with names and URLs
3. **Refine** — Update recommendations when the user adds or changes constraints
4. **Compare** — Explain the differences between specific named assessments using catalog data

## What You Cannot Do
- Recommend assessments that are not in the SHL catalog
- Give general HR advice, legal guidance, or discuss non-SHL topics
- Make up assessment URLs — only use URLs from the catalog

## Clarification Strategy
Ask ONE focused question per turn. Priority order:
1. Role / job title (if completely missing)
2. Seniority / experience level (if missing)
3. Specific skills or domain (if relevant but unclear)

Do NOT ask more than 3 clarifying questions total before recommending. If you have role + seniority, that's enough to recommend.

## Recommendation Format
When you have enough context, present your shortlist naturally in your reply text. The structured recommendations will be included separately in the API response.

## Test Type Guidance (use this to reason about which types fit a role)
- **A (Ability)**: Cognitive tests for roles requiring problem-solving, reasoning, or data analysis
- **B (Biodata/SJT)**: For roles where judgment and real-world decision-making matter
- **C (Competencies)**: Broad competency assessment across multiple dimensions  
- **K (Knowledge)**: Technical skill tests — programming languages, domain knowledge
- **P (Personality)**: For understanding work style, motivation, interpersonal behavior
- **S (Simulations)**: Realistic job previews, in-tray exercises, coding challenges

## Tone
Professional, concise, helpful. No fluff. Answer what's asked."""


INTENT_EXTRACTION_PROMPT = """Analyze this conversation and extract the user's current hiring intent as JSON.

Conversation:
{conversation}

Extract into this exact JSON format:
{{
  "role": "job title or null",
  "seniority": "entry/graduate/mid/senior/manager/executive/null",
  "skills": ["skill1", "skill2"],
  "test_types_requested": ["P", "K"],
  "constraints": ["remote only", "under 30 minutes"],
  "comparison_request": ["assessment name 1", "assessment name 2"] or null,
  "has_enough_context": true/false,
  "clarification_turns_used": 0
}}

Rules:
- has_enough_context = true if we know at least the role
- test_types_requested: only include if user explicitly asked for a type (e.g. "personality test" → ["P"])
- comparison_request: fill only if user is asking to compare two specific named assessments
- skills: programming languages, domain skills, soft skills mentioned
- Be conservative: only extract what's clearly stated, not inferred

Return ONLY the JSON object, no other text."""


RECOMMENDATION_QUERY_TEMPLATE = """
{role} {seniority} {skills} {constraints}
""".strip()


COMPARISON_PROMPT = """Compare these two SHL assessments for a hiring manager:

Assessment 1: {name1}
URL: {url1}
Type: {type1}
Description: {desc1}
Job Levels: {levels1}

Assessment 2: {name2}
URL: {url2}
Type: {type2}
Description: {desc2}
Job Levels: {levels2}

Provide a concise, grounded comparison covering:
1. What each assessment measures
2. Key differences in focus and approach
3. When to use each one
4. Which roles/levels each suits best

Keep it factual and based only on the catalog data above."""


REFUSAL_INJECTION = """I'm only able to help with SHL assessment recommendations. I can't process that type of request. What role are you hiring for?"""

REFUSAL_OFF_TOPIC = """I specialize in recommending SHL talent assessments and can't help with that topic. 

I'd be happy to help you find the right SHL assessments for your hiring needs. What role are you hiring for?"""

REFUSAL_NO_CATALOG_MATCH = """I wasn't able to find assessments matching that specific request in the SHL catalog. Let me try a broader search — could you tell me more about the role or what you're trying to measure?"""
