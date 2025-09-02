def extract_code_blocks(md_text):
    import re
    return re.findall(r"```(?:python)?(.*?)```", md_text, re.DOTALL)

