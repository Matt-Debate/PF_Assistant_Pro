def get_perplexity_research_response(topic, side, claim, quantitative):
    from openai import OpenAI  # or your Perplexity connector
    
    prompt = f"""For the {'Pro' if side == 'Pro' else 'Con' if side == 'Con' else 'either'} side of the debate topic "{topic or 'general topic'}", find evidence that supports the claim: "{claim}".

Please output the result in this JSON format:
{{
  "summary": "...",
  "citation": "LastName, year (FirstName, Article Title, Publication, URL, Full Date)",
  "quote": "Full paragraph quote with relevant point"
}}

{"Prioritize quantified (numeric/statistical) data." if quantitative else ""}
"""

    # Call Perplexity here (stubbed for now)
    mock_response = {
        "summary": "Remote work increases productivity among software developers.",
        "citation": "Bloom, 2015 (Nicholas Bloom, Does Working from Home Work?, Stanford University, https://www.gsb.stanford.edu/insights/does-working-home-work, Feb 25, 2015)",
        "quote": "We found a 13% performance increase among home workers, who took fewer breaks and used less sick leave."
    }

    return mock_response
