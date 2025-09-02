"""
Guidance-based schema enforcement utilities.
"""
from typing import Dict, Any
import time
import guidance as gd

def require_schema(schema: Dict[str, Any], prompt: str, *, model_name: str = "gpt-4o-mini", max_attempts: int = 3) -> Dict[str, Any]:
    """Generate JSON adhering to schema, retrying on failure."""
    llm = gd.llms.OpenAI(model_name)
    template = gd(
        """
You must output only JSON that validates against this schema:

{{json schema}}

{{#json schema}}  
{{/json}}

User prompt:
{{prompt}}
"""
    )
    last_err = None
    for attempt in range(1, max_attempts+1):
        out = template(llm, schema=schema, prompt=prompt, temperature=0)
        try:
            data = out.json()
            return data
        except Exception as e:
            last_err = e
            time.sleep(0.2 * attempt)
    raise ValueError(f"Schema validation failed after {max_attempts} attempts: {last_err}")