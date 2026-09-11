"""Fit complete conversation turns into a checkpoint's token budget."""


def build_chat_prompt(messages, tokenizer, context, reserve=32):
    """Keep the latest instruction intact and retain recent complete turn pairs.

    Reserve positions for generation where possible. Never truncate an individual
    instruction or insert orphan assistant replies. Return explicit context usage.
    """
    encode = tokenizer.encode if tokenizer else lambda text: list(text.encode('utf-8'))
    if not messages or messages[-1]['role'] != 'user':
        raise ValueError('Conversation must end with a user message.')
    def instruction(content):
        return f'### Instruction:\n{content}\n\n### Response:\n'
    latest = instruction(messages[-1]['content'])
    latest_tokens = len(encode(latest))
    if latest_tokens > context:
        raise ValueError(f'Your latest message needs {latest_tokens} tokens including chat formatting; '
                         f'this checkpoint supports {context}. Shorten the message.')
    budget = max(latest_tokens, context - min(reserve, context // 4))
    pairs = []
    pending = None
    for message in messages[:-1]:
        if message['role'] == 'user':
            pending = message['content']
        elif message['role'] == 'assistant' and pending is not None:
            pairs.append(instruction(pending) + message['content'] + '\n<|end|>\n\n')
            pending = None
    prompt = latest
    kept = 0
    for pair in reversed(pairs):
        candidate = pair + prompt
        if len(encode(candidate)) > budget:
            break
        prompt = candidate
        kept += 1
    return prompt, dict(prompt_tokens=len(encode(prompt)), context_limit=context,
                        retained_turns=kept, omitted_messages=len(messages) - 1 - kept * 2)
