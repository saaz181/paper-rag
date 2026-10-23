import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

HF_TOKEN = os.getenv('HF_TOKEN')

class Generator:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", token=HF_TOKEN)

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)


    def build_messages(self, query, context):

        system_prompt = """
            You are a research assistant answering questions using retrieved scientific-paper passages.

            Rules:
            - Use only information explicitly supported by the provided context.
            - Do not introduce claims that are not stated or clearly supported by the context.
            - Answer the exact question directly.
            - Synthesize the evidence instead of copying paper titles or long passages.
            - Keep the answer concise and clear.
            - Cite supporting sources inline using exactly the source labels provided in the context,
            for example: [2005.11401v4.pdf, p. 0].
            - Every important factual claim should have a citation when possible.
            - Do not invent document names, page numbers, or citations.
            - If the context does not contain enough information, say:
            "The provided documents do not contain enough information to answer this question."
            """.strip()

        user_prompt = f"""
                Context:

                {context}

                Question:
                {query}
                """.strip()

        return [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

    def generate(self, query, context, max_new_tokens=220):

        messages = self.build_messages(
            query=query,
            context=context
        )

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.device)

        print(
            "Input tokens:",
            inputs["input_ids"].shape[1]
        )

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        generated_tokens = outputs[
            0,
            inputs["input_ids"].shape[1]:
        ]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return answer.strip()